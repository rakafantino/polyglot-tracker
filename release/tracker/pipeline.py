# sidecar/tracker/pipeline.py
"""End-to-end smart-tracking analysis. Samples the clip window, detects and
tracks faces, scores active-speaker per face, decides framing, smooths, and
returns a trackPlan (see the output contract in the task / track-plan.js)."""
import collections
import os
import queue
import threading

import cv2

from detect import Detector, detector_candidates
from landmarks import Landmarker
from track import Tracker
from asd import ASD, MotionVadASD, extract_audio_rms
from scenes import scene_cuts
from grid import build_grid_keyframes
from smooth import smooth_keyframes

# Active-speaker signal source. Default "blendshape" = the shipped path
# (MediaPipe blendshapes x RMS loudness). "motion_vad" swaps in per-track
# mouth-ROI pixel motion x Silero VAD and SKIPS the MediaPipe landmark pass
# entirely — the 46-50% wall-time bottleneck. See
# [[lokaclip-asd-algorithm-reverse-engineered]].
#
# [ASD-MODE-CHOICE] User opt-in ONLY. The packaged app sends config.asd_mode
# per-job — the app's own default is 'blendshape' unless the user explicitly
# picked the experimental engine in Settings, so this NEVER silently changes
# behaviour for someone who never opted in. TRK_ASD_MODE stays as a dev/CLI
# override (running main.py by hand, benchmarking) for when there's no job
# config to read from at all; a real job's explicit choice always wins over it.
def _asd_mode(job_config=None):
    if job_config and str(job_config.get("asd_mode", "")).strip().lower() == "motion_vad":
        return "motion_vad"
    return os.environ.get("TRK_ASD_MODE", "blendshape").strip().lower()

# Detector/Landmarker load + compile ONNX/TFLite models — the single biggest
# fixed cost per analyze() call (~7s, per earlier profiling). In a persistent
# sidecar process (main.py now serves many jobs per process lifetime) that
# cost must be paid ONCE, not per clip. Cached as module-level singletons;
# Tracker stays per-job (cheap ByteTrack state, must not leak across clips).
_landmarker = None

# Detectors are keyed by the frame shape they were built for, because the
# ideal det_size is aspect-dependent (see detect.det_size_for) — a session
# built for a 16:9 source would letterbox a 9:16 clip into 40%-smaller faces.
# Bounded: a process serving many clips sees at most a handful of distinct
# aspects, and each entry pins a model + ORT session.
_detectors = collections.OrderedDict()
_MAX_DETECTORS = 3


def _get_detector(frame_w, frame_h, long_side=None):
    cands = detector_candidates(frame_w, frame_h, long_side)
    key = cands[0][0]                      # the ideal det_size for this shape
    if key in _detectors:
        _detectors.move_to_end(key)
        return _detectors[key]
    # probe_shape is the REAL frame shape: the probe has to exercise the same
    # letterboxing the clip will, or it proves nothing about this det_size.
    det = Detector(candidates=cands, probe_shape=(frame_h, frame_w))
    _detectors[key] = det
    _detectors.move_to_end(key)
    while len(_detectors) > _MAX_DETECTORS:
        _detectors.popitem(last=False)
    return det


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        _landmarker = Landmarker()
    return _landmarker


def analyze(job, on_progress=lambda stage, pct: None):
    video = job["video"]
    clip_start = float(job.get("clip_start", 0.0))
    clip_end = float(job.get("clip_end", 0.0))
    sample_fps = float(job.get("sample_fps", 12) or 12)
    config = job.get("config", {}) or {}

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if clip_end <= clip_start:
        clip_end = clip_start + (cap.get(cv2.CAP_PROP_FRAME_COUNT) /
                                 max(1e-6, cap.get(cv2.CAP_PROP_FPS)))
    dur = max(0.0, clip_end - clip_start)

    # Heartbeats: every phase below is silent but can take tens of seconds
    # under load; the app kills a sidecar that produces no output for too
    # long, so each phase boundary emits a progress line.
    on_progress("prep", 0.1)
    detector = _get_detector(frame_w, frame_h, config.get("det_long_side"))
    mode = _asd_mode(config)
    motion_mode = mode == "motion_vad"
    # In motion_vad mode the MediaPipe landmark pass — the dominant cost — is
    # skipped entirely; per-track mouth-ROI motion supplies the visual signal.
    landmarker = None
    mouth = None
    asd = None
    asd_fallback_reason = None
    if motion_mode:
        try:
            from mouth_motion import MouthMotion
            from vad import extract_speech_prob
            speech, hop = extract_speech_prob(video, clip_start, clip_end, ffmpeg=job.get("ffmpeg"))
            asd = MotionVadASD(speech, hop)
            mouth = MouthMotion((frame_h, frame_w))
        except Exception as exc:
            # [SILENT-FALLBACK-FIX] This used to swallow the exception with no
            # trace anywhere — motion_vad is an opt-in "experimental, faster"
            # toggle (renderer.js f_asdModeQ), so a user testing it would get
            # normal-looking results while silently running the OLD slow
            # blendshape/MediaPipe path the whole time, with zero indication
            # the fast path never engaged. Recorded here and ridden along on
            # `diag` below (same "app ignores unknown fields" extension point
            # detector diag already uses) so it reaches the render log instead
            # of vanishing into an unread stderr buffer (TrackerWorker only
            # reads _errBuf on a crash, never on a job that still succeeds).
            motion_mode = False
            asd_fallback_reason = f"{type(exc).__name__}: {exc}"

    if not motion_mode:
        landmarker = _get_landmarker()
        landmarker.reset()   # singleton: its adaptive state is per-CLIP, not per-process
        mouth = None
        rms, hop = extract_audio_rms(video, clip_start, clip_end, ffmpeg=job.get("ffmpeg"))
        asd = ASD(rms, hop)
    tracker = Tracker()
    on_progress("prep", 0.7)

    cuts = set(scene_cuts(video, clip_start, clip_end))
    on_progress("prep", 1.0)

    n_samples = max(1, int(dur * sample_fps))
    sample_times = [i / sample_fps for i in range(n_samples) if i / sample_fps <= dur]
    samples = []
    next_cut_reset = sorted(cuts)

    # [SPEED — DECODE/DETECT OVERLAP] Profiled live (real 4K source, real
    # installed bundle): cap.read() costs ~33ms/frame at 4K while
    # detector.detect() (SCRFD) costs ~150-280ms/frame regardless of
    # execution provider. Those used to run strictly back-to-back in one
    # thread — decode, then detect, then decode the next frame — so every
    # frame paid the FULL 33ms decode on top of its detect cost. Decoding is
    # I/O/CPU work; detecting runs on a separate accelerator (DirectML/
    # CoreML/CPU EP) — there's no reason they can't overlap. A background
    # thread now does ONLY cv2 grab/retrieve (this pipeline's only
    # VideoCapture user, so no thread-safety concern) and pushes decoded
    # frames through a small bounded queue; the main thread pulls a
    # ready-decoded frame and runs detect+landmark+track on it while the
    # producer is already decoding the NEXT one. Bounded to 4 in-flight
    # frames (~4×18MB for a 4K frame — trivial memory) so a stalled decode
    # can't buffer unbounded frames in memory.
    def _put(q_, item, stop_):
        # A plain blocking put() would never notice stop_event while the
        # queue is full — if the consumer thread exits early (an exception
        # mid-frame), the producer could block on put() forever. Poll with a
        # short timeout instead so stop_event is checked even while full.
        while not stop_.is_set():
            try:
                q_.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    def _produce(cap_, times_, start_, q_, stop_):
        idx = 0
        while idx < len(times_) and not stop_.is_set():
            ok = cap_.grab()
            if not ok:
                break
            cur_t = (cap_.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) - start_
            if cur_t + 1e-3 < times_[idx]:
                continue
            ok2, frame = cap_.retrieve()
            if not ok2 or frame is None:
                idx += 1
                continue
            while idx < len(times_) and cur_t + 1e-3 >= times_[idx] and not stop_.is_set():
                _put(q_, (times_[idx], frame), stop_)
                idx += 1
        _put(q_, None, stop_)  # sentinel: no more frames

    # Seek ONCE to the clip start, then decode sequentially and pick off
    # frames as playback time crosses each target sample time. Re-seeking
    # per sample (previous approach) forces the decoder to re-sync from the
    # nearest keyframe on every single sample — 10-20x slower than a single
    # seek + forward decode for typical GOP sizes.
    cap.set(cv2.CAP_PROP_POS_MSEC, clip_start * 1000.0)

    frame_q = queue.Queue(maxsize=4)
    stop_event = threading.Event()
    producer = threading.Thread(target=_produce, args=(cap, sample_times, clip_start, frame_q, stop_event),
                                 daemon=True)
    producer.start()
    try:
        processed = 0
        import os as _os
        _min_score = float(_os.environ.get("TRK_MIN_SCORE", 0.55))
        _min_h_frac = float(_os.environ.get("TRK_MIN_FACE_H", 0.05))
        while True:
            item = frame_q.get()
            if item is None:
                break
            t, frame = item
            processed += 1

            # Reset tracker/framing continuity at scene cuts. Also flag this
            # sample as a cut boundary so downstream segment-merger emits a
            # fresh segment.
            cut_here = False
            while next_cut_reset and t >= next_cut_reset[0]:
                tracker.reset()
                if mouth is not None:
                    mouth.reset()   # never diff a mouth ROI across a scene cut
                next_cut_reset.pop(0)
                cut_here = True

            dets = detector.detect(frame)
            # Reject spurious faces (wall posters, faces on the on-set TV, tiny
            # background heads): require a confident score and a minimum face
            # height relative to the frame. Real foreground speakers clear
            # both easily.
            dets = [d for d in dets
                    if d.score >= _min_score
                    and (d.y2 - d.y) >= _min_h_frac * frame_h]
            # blendshape mode: one MediaPipe pass over all dets (the costly path).
            # motion_vad mode: no landmark pass — motion is computed per tracked
            # face below, keyed by track id (needs the SAME face's previous ROI).
            bs = (landmarker.blendshapes(frame, detections=dets)
                  if (dets and not motion_mode) else [])
            tracked = tracker.update(dets, (frame_h, frame_w))

            faces = []
            for tid, d in tracked:
                if motion_mode:
                    score = asd.score(mouth.motion(frame, d.kps, tid), t)
                else:
                    # align blendshapes to this detection by index in dets
                    shape = None
                    try:
                        di = dets.index(d)
                        if di < len(bs):
                            shape = bs[di]
                    except ValueError:
                        shape = None
                    score = asd.score(shape, t)
                faces.append({
                    "id": tid,
                    "bbox": (d.x, d.y, d.x2, d.y2),
                    "yaw": d.yaw,
                    "asd": score,
                })
            samples.append({"t": round(t, 3), "faces": faces, "cut": cut_here})

            # [SPEED-UX] Was // 10 (10% steps). Per-frame detection cost is
            # ~150-280ms on integrated-GPU/no-dGPU Windows machines (profiled
            # live), so a 10% chunk of a long clip can go many seconds between
            # heartbeats — reads as "stuck" even though it's progressing.
            # Finer 2% steps cost nothing (heartbeat, not extra work) and keep
            # the UI progress bar visibly moving.
            if processed % max(1, len(sample_times) // 50) == 0:
                on_progress("detect", processed / len(sample_times))
    finally:
        stop_event.set()
        producer.join(timeout=2.0)
        cap.release()
    on_progress("detect", 1.0)

    # Which detector config actually survived the probe is the first thing
    # anyone debugging a slow clip needs, and it is invisible from outside the
    # process. Ride it along on the result; the app ignores unknown fields.
    diag = {"det_size": list(detector.det_size), "providers": detector.providers}
    # [SILENT-FALLBACK-FIX] Surfaces the motion_vad->blendshape downgrade (set
    # above) through the one channel that actually reaches a visible log line
    # on a SUCCESSFUL job — pipeline-inverse-v1.js logs "Clip N tracked (...)"
    # off this result object, unlike stderr which is only read on a crash.
    if asd_fallback_reason:
        diag["asd_mode_requested"] = "motion_vad"
        diag["asd_mode_used"] = "blendshape"
        diag["asd_fallback_reason"] = asd_fallback_reason

    if not samples:
        # No decodable frames: emit a safe centered single keyframe.
        return {"type": "result", "detector": diag, "keyframes": [
            {"t": 0.0, "mode": "single",
             "target": {"x": 0.5, "y": 0.5, "scale": 0.85}}]}

    output_aspect = job.get("output_aspect", "9:16")
    keyframes = build_grid_keyframes(samples, config, frame_w, frame_h,
                                     output_aspect=output_aspect)
    # [SMOOTH-V1] grid.py picks one median position per ~1.2s sub-segment
    # independently — no continuity constraint across sub-segments of the
    # SAME continuous shot, so consecutive positions could snap hard even
    # with no real cut. Blend across those (never across a real `cut`).
    keyframes = smooth_keyframes(keyframes)
    on_progress("frame", 1.0)
    if not keyframes:
        keyframes = [{"t": 0.0, "mode": "single",
                      "target": {"x": 0.5, "y": 0.5, "scale": 0.85}}]
    return {"type": "result", "detector": diag, "keyframes": keyframes}
