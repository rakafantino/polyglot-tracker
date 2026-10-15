# sidecar/tracker/landmarks.py
"""MediaPipe Face Landmarker v2 → blendshapes (jawOpen etc.) used for the
active-speaker heuristic.

Runs the full-frame pass in VIDEO mode (monotonic synthetic timestamps):
MediaPipe only enables its internal landmark tracking + temporal filtering in
VIDEO/LIVE_STREAM modes, so IMAGE mode re-detects from scratch every frame and
the blendshape scores jitter frame-to-frame. VIDEO mode gives materially
steadier mouthActivity, which is what the ASD threshold gates on. Set
TRK_LM_VIDEO=0 to fall back to per-frame IMAGE mode.

MediaPipe's built-in face detector downscales the frame to ~192px, so small
faces in a wide (e.g. 1920x1080 two-shot) frame vanish. To stay robust we try
the full frame first (fast path for close-ups), and when that finds nothing we
fall back to running an IMAGE-mode landmarker on upscaled crops around SCRFD
detections. (Crops must NOT go through the VIDEO-mode instance — each crop is
a different "camera view" and would corrupt its tracking state.)

`blendshapes(frame, detections)` returns a list ALIGNED WITH `detections`
(entry i belongs to detections[i]; None where no landmark result matched).
MediaPipe's face ordering has no relation to SCRFD's, so the full-frame path
matches results to detections spatially by landmark-bbox centre distance —
index-based zipping silently attached mouths to the wrong faces in
multi-person frames.
"""
import collections
import os

import cv2
import mediapipe as mp
from paths import model_path

# Padding factor around a detection box before cropping (2.0 => 2x the box).
_CROP_PAD = 2.0
# NOT a speed knob. Measured on the 5 training fixtures: dropping this to 192
# or 160 saves NOTHING (13.3ms vs 13.5ms per crop -- the cost is MediaPipe's
# fixed graph overhead, not the pixels) while costing 5-6% of ASD gate
# decisions. Leave it at 256.
_CROP_SIZE = 256
# Synthetic per-call timestamp step for VIDEO mode. MediaPipe only needs the
# timestamps to be monotonically increasing; the sidecar samples ~12fps but a
# constant ~30fps step keeps the internal filter responsive.
_TS_STEP_MS = 33

# [SPEED — ADAPTIVE FULL-FRAME PASS] The full-frame pass costs ~10ms/frame
# and, on 1080p multi-face clips, resolves almost nothing: measured across the
# training fixtures it accounted for only 18.6% of resolved detections, while
# the crop fallback did the other 81% (src2: 450 crop calls for 120 frames,
# 46% of total analyze() wall-clock). It still earns its place on close-up
# single-face clips, where one full-frame call resolves the only face more
# cheaply than a crop would -- so this is measured per clip rather than
# switched off outright. Turning it off where it does not pay measured 1.17x
# on the landmark stage with ZERO changes to ASD gate decisions.
_ADAPTIVE = os.environ.get("TRK_LM_ADAPTIVE", "1") not in ("0", "false")
_FF_WINDOW = int(os.environ.get("TRK_LM_FF_WINDOW", 16))     # frames of history
_FF_MIN_HIT = float(os.environ.get("TRK_LM_FF_MIN_HIT", 0.34))
_FF_REPROBE = int(os.environ.get("TRK_LM_FF_REPROBE", 24))   # frames between re-tests


class Landmarker:
    def __init__(self, num_faces=5, min_confidence=0.3):
        self._num_faces = num_faces
        self._min_conf = min_confidence
        self._video = os.environ.get("TRK_LM_VIDEO", "1") not in ("0", "false")
        self.lm = self._build(video=self._video)
        self._ts_ms = 0
        self._crop_lm = None   # lazily created IMAGE-mode instance for crops
        self._detector = None  # lazily created SCRFD for the crop fallback
        # Rolling record of how much the full-frame pass actually resolves:
        # one (resolved, total) pair per frame it ran on.
        self._ff_hist = collections.deque(maxlen=_FF_WINDOW)
        self._ff_skipped = 0   # frames since the last full-frame call

    def _build(self, video):
        base = mp.tasks.BaseOptions(model_asset_path=model_path("face_landmarker.task"))
        mode = (mp.tasks.vision.RunningMode.VIDEO if video
                else mp.tasks.vision.RunningMode.IMAGE)
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base, num_faces=self._num_faces,
            output_face_blendshapes=True,
            min_face_detection_confidence=self._min_conf,
            min_face_presence_confidence=self._min_conf,
            running_mode=mode)
        return mp.tasks.vision.FaceLandmarker.create_from_options(opts)

    @staticmethod
    def _pack(bs):
        # Return several talking-related blendshapes. `mouthActivity` is a
        # rough all-in-one signal for whether the mouth is doing something
        # (jaw drop, pucker, funnel, side stretch). Some speakers barely drop
        # their jaw (relaxed / smile-talking) but still register on pucker /
        # funnel / stretch — jawOpen alone missed the mind+ chef host.
        d = {c.category_name: c.score for c in bs}
        def g(k): return float(d.get(k, 0.0))
        activity = max(
            g("jawOpen"),
            g("mouthPucker"),
            g("mouthFunnel"),
            0.6 * (g("mouthStretchLeft") + g("mouthStretchRight")),
            0.5 * (g("mouthPressLeft")   + g("mouthPressRight")),
            0.5 * (g("mouthUpperUpLeft") + g("mouthUpperUpRight")),
        )
        return {"jawOpen": g("jawOpen"),
                "mouthClose": g("mouthClose"),
                "mouthActivity": activity}

    def _detect_full(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._video:
            self._ts_ms += _TS_STEP_MS
            return self.lm.detect_for_video(mp_img, self._ts_ms)
        return self.lm.detect(mp_img)

    def _detect_crop(self, bgr):
        if self._crop_lm is None:
            self._crop_lm = self._build(video=False)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._crop_lm.detect(mp_img)

    @staticmethod
    def _lm_center(landmarks, W, H):
        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]
        return (min(xs) + max(xs)) / 2.0 * W, (min(ys) + max(ys)) / 2.0 * H

    def _align(self, res, detections, W, H):
        """Assign each full-frame MediaPipe face to its nearest detection by
        landmark-bbox centre. Greedy nearest-first; a face must land within
        ~60% of the detection's size to count."""
        shapes = [None] * len(detections)
        if not res.face_blendshapes:
            return shapes
        pairs = []
        for fi, lms in enumerate(res.face_landmarks):
            cx, cy = self._lm_center(lms, W, H)
            for di, d in enumerate(detections):
                dcx, dcy = (d.x + d.x2) / 2.0, (d.y + d.y2) / 2.0
                size = max(d.x2 - d.x, d.y2 - d.y, 1.0)
                dist = ((cx - dcx) ** 2 + (cy - dcy) ** 2) ** 0.5
                if dist <= 0.6 * size:
                    pairs.append((dist, fi, di))
        pairs.sort()
        used_f, used_d = set(), set()
        for dist, fi, di in pairs:
            if fi in used_f or di in used_d:
                continue
            used_f.add(fi); used_d.add(di)
            shapes[di] = self._pack(res.face_blendshapes[fi])
        return shapes

    def reset(self):
        """Forget the per-clip adaptive state.

        The Landmarker is a module-level singleton reused for every job in a
        persistent sidecar process, so without this the gate learned on one
        clip carries into the next: a close-up clip following a crowded one
        would start with the full-frame pass switched off and stay that way
        until the re-probe fired, framing the first second of the clip off
        evidence gathered from different footage entirely."""
        self._ff_hist.clear()
        self._ff_skipped = 0

    def _full_frame_worth_it(self):
        """Should the full-frame pass run on this frame?

        Skipped only once there is enough evidence it is not paying for
        itself on THIS clip, and re-tested every _FF_REPROBE frames so a cut
        to a close-up (where it is the cheaper path) switches it back on
        without the pipeline having to tell us about the cut."""
        if not _ADAPTIVE:
            return True
        if len(self._ff_hist) < _FF_WINDOW:
            return True                      # still gathering evidence
        if self._ff_skipped >= _FF_REPROBE:
            return True                      # periodic re-test
        resolved = sum(r for r, _ in self._ff_hist)
        total = sum(t for _, t in self._ff_hist)
        return total == 0 or (resolved / total) >= _FF_MIN_HIT

    def blendshapes(self, bgr, detections=None):
        """Blendshape dicts aligned with `detections` (None where unmatched).
        With detections=None, returns the full-frame results in MediaPipe
        order (legacy behavior for callers without a detector)."""
        H, W = bgr.shape[:2]

        # The legacy no-detections path has no boxes to crop around, so it
        # always needs the full-frame result.
        if detections is None:
            res = self._detect_full(bgr)
            if res.face_blendshapes:
                return [self._pack(bs) for bs in res.face_blendshapes]
            # Find faces ourselves so the crop fallback can still resolve
            # small faces.
            if self._detector is None:
                from detect import Detector
                self._detector = Detector()
            detections = self._detector.detect(bgr)
            return [s for s in self._align_and_fallback(res, detections, bgr)
                    if s is not None]

        if self._full_frame_worth_it():
            self._ff_skipped = 0
            shapes = self._align(self._detect_full(bgr), detections, W, H)
            self._ff_hist.append((sum(1 for s in shapes if s is not None),
                                  len(detections)))
        else:
            self._ff_skipped += 1
            shapes = [None] * len(detections)
        return self._crop_fallback(shapes, detections, bgr)

    def _align_and_fallback(self, res, detections, bgr):
        H, W = bgr.shape[:2]
        return self._crop_fallback(self._align(res, detections, W, H), detections, bgr)

    def _crop_fallback(self, shapes, detections, bgr):
        H, W = bgr.shape[:2]
        if all(s is not None for s in shapes):
            return shapes
        # Fallback for detections the full-frame pass missed: crop around each
        # and upscale so MediaPipe's internal detector can resolve small faces
        # in wide frames. Keeps alignment — entry i stays detections[i].
        for i, d in enumerate(detections):
            if shapes[i] is not None:
                continue
            cx, cy = (d.x + d.x2) / 2, (d.y + d.y2) / 2
            s = max(d.x2 - d.x, d.y2 - d.y) * _CROP_PAD
            x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
            x1, y1 = int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
            if x1 <= x0 or y1 <= y0:
                continue
            crop = cv2.resize(bgr[y0:y1, x0:x1], (_CROP_SIZE, _CROP_SIZE))
            cres = self._detect_crop(crop)
            if cres.face_blendshapes:
                shapes[i] = self._pack(cres.face_blendshapes[0])
        return shapes
