# sidecar/tracker/asd.py
"""Heuristic active-speaker detection (no torch). A face's speaking score is
normalized mouth movement (jawOpen from MediaPipe blendshapes) gated by the
short-window audio loudness (RMS) at that instant. The face whose mouth moves
during loud audio scores high; a moving mouth in silence, or loud audio with a
closed mouth, scores low.

Audio is extracted once via the app's ffmpeg to 16kHz mono wav; per-frame RMS is
computed with numpy over a short window centered on the sample time.
"""
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np

_SR = 16000
_WIN = 0.12  # seconds; short window for RMS around each sample time


def _resolve_ffmpeg(explicit=None):
    """Find an ffmpeg binary to shell out to for audio extraction.

    [PORTABILITY] This used to be a single hardcoded absolute path
    (`/Users/zaree/socl-visuals-v05/app/node_modules/ffmpeg-static/ffmpeg`)
    left over from the original developer's Mac. On any other machine —
    confirmed live on a real Windows install — that path doesn't exist,
    subprocess.run() raises, the broad `except Exception` below swallows it,
    and extract_audio_rms silently returns an all-zero RMS series. Every
    ASD score is `mouth * loud` with `loud` permanently 0, so no face EVER
    registers as an active speaker — which silently disabled split-screen
    detection (grid.py requires >=2 *active speakers*) on every non-Mac
    machine, not just this one. Preferred order: caller-supplied path (the
    Electron app already resolves its bundled ffmpeg and can pass it via
    job["ffmpeg"]) -> TRK_FFMPEG/FFMPEG_BIN env vars -> whatever `ffmpeg` is
    on PATH. No hardcoded fallback path — better to fail loudly (caught by
    the same except below, ASD score 0 as before) than to silently point at
    a path that only ever existed on one person's machine.
    """
    for candidate in (explicit,
                      os.environ.get("TRK_FFMPEG"),
                      os.environ.get("FFMPEG_BIN")):
        if candidate and os.path.exists(candidate):
            return candidate
    found = shutil.which("ffmpeg")
    return found or "ffmpeg"


def extract_audio_rms(video_path, start, end, hop=1.0 / 24, ffmpeg=None):
    """Extract [start,end] of the video's audio to 16kHz mono and return
    (rms_series, hop) where rms_series[i] is the RMS over a short window ending
    at time i*hop (clip-local seconds). Normalized to 0..1 by its own p95.

    Returns (np.zeros(1), hop) if the clip has no audio track."""
    dur = max(0.0, end - start)
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        cmd = [_resolve_ffmpeg(ffmpeg), "-v", "error", "-y", "-ss", str(start), "-t", str(dur),
               "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(_SR),
               "-f", "wav", wav]
        subprocess.run(cmd, check=True, capture_output=True)
        with wave.open(wav, "rb") as w:
            n = w.getnframes()
            raw = w.readframes(n)
        if n == 0:
            return np.zeros(1), hop
        sig = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return np.zeros(1), hop
    finally:
        try:
            os.remove(wav)
        except OSError:
            pass

    win = max(1, int(_WIN * _SR))
    n_hops = max(1, int(dur / hop) + 1)
    rms = np.zeros(n_hops, dtype=np.float32)
    for i in range(n_hops):
        c = int(i * hop * _SR)
        a = max(0, c - win // 2)
        b = min(len(sig), c + win // 2)
        if b > a:
            seg = sig[a:b]
            rms[i] = float(np.sqrt(np.mean(seg * seg)))
    p95 = np.percentile(rms, 95) if rms.size else 0.0
    if p95 > 1e-6:
        rms = np.clip(rms / p95, 0.0, 1.0)
    return rms, hop


class ASD:
    def __init__(self, audio_rms_series, hop):
        self.rms = np.asarray(audio_rms_series, dtype=np.float32)
        self.hop = hop

    def _rms_at(self, t):
        if self.rms.size == 0:
            return 0.0
        idx = int(round(t / self.hop))
        idx = min(max(idx, 0), self.rms.size - 1)
        return float(self.rms[idx])

    def score(self, face_blendshapes, t):
        """0..1 active-speaker score for a face at clip-local time t.
        face_blendshapes: {"jawOpen":.., "mouthActivity":..} or None.

        We use `mouthActivity` — max of jawOpen / pucker / funnel / stretch /
        press / upper-lip lift — because some speakers barely drop the jaw
        (relaxed / smile-talking podcast hosts) and jawOpen alone missed them.
        Score gates mouth activity by audio loudness so a static face doesn't
        score high just because someone else is talking loudly."""
        if not face_blendshapes:
            return 0.0
        mouth_raw = float(face_blendshapes.get("mouthActivity",
                            face_blendshapes.get("jawOpen", 0.0)))
        floor = float(os.environ.get("TRK_MOUTH_FLOOR", 0.02))
        full = float(os.environ.get("TRK_MOUTH_FULL", 0.20))
        mouth = np.clip((mouth_raw - floor) / max(1e-3, full - floor), 0.0, 1.0)
        loud = self._rms_at(t)
        return float(mouth * loud)


class MotionVadASD:
    """Active-speaker scorer for the motion+VAD mode
    ([[lokaclip-asd-algorithm-reverse-engineered]]): the visual mouth signal is
    per-track mouth-ROI pixel motion (mouth_motion.MouthMotion) instead of
    MediaPipe blendshapes, and the audio gate is Silero VAD speech probability
    (vad.extract_speech_prob) instead of raw RMS loudness.

    Same 0..1 contract and floor/full normalisation philosophy as ASD.score, so
    downstream (grid.py / track selection) is unchanged. Kept a separate class
    rather than a mode flag on ASD so neither path can accidentally read the
    other's audio series.

    Normalisation is FIXED (env floor/full), not per-clip percentile: the
    pipeline scores each face inline while streaming frames, before the whole
    clip's motion distribution is known, and a fixed scale keeps face-to-face
    ordering within a frame stable — which is what the split/single decision
    actually consumes."""

    def __init__(self, speech_prob_series, hop):
        self.vad = np.asarray(speech_prob_series, dtype=np.float32)
        self.hop = hop

    def _vad_at(self, t):
        if self.vad.size == 0:
            return 0.0
        idx = int(round(t / self.hop))
        idx = min(max(idx, 0), self.vad.size - 1)
        return float(self.vad[idx])

    def score(self, motion, t):
        """0..1 active-speaker score from a face's mouth-ROI `motion` at time t,
        gated by Silero VAD speech probability. `motion` is the value from
        MouthMotion.motion(); None/0 yields 0.

        FLOOR/FULL are calibrated so the ACTIVE speaker saturates near 1 while
        PASSIVE faces stay below grid.py's TRK_ASD_THRESHOLD (0.2), giving the
        max-ASD speaker pick real discrimination. Measured mouth-ROI motion on
        the multi-face fixtures (two_speakers, panel_12s/20s): active speaker
        p25-p75 = 0.061-0.099, passive face p50-p90 = 0.033-0.063. FLOOR at the
        passive band's top and FULL in the active band's middle maps passive
        motion to <0.2 and active motion to >0.6. The earlier 0.006/0.045
        defaults sat BELOW the passive band, so every moving face saturated to
        1.0 and no face could be told from another — the split/speaker-pick
        precision bug. See [[lokaclip-asd-algorithm-reverse-engineered]]."""
        if not motion:
            return 0.0
        floor = float(os.environ.get("TRK_MOTION_FLOOR", 0.05))
        full = float(os.environ.get("TRK_MOTION_FULL", 0.09))
        m = np.clip((float(motion) - floor) / max(1e-6, full - floor), 0.0, 1.0)
        return float(m * self._vad_at(t))
