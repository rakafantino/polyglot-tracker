# sidecar/tracker/vad.py
"""Silero VAD (voice-activity detection) speech-probability track, torch-free.

Returns a per-time speech-probability series over a clip's audio, the audio
counterpart to `asd.extract_audio_rms`. Used by the motion+VAD active-speaker
mode ([[lokaclip-asd-algorithm-reverse-engineered]]): raw RMS loudness fires on
music/noise too, while Silero VAD fires only on human speech, so gating mouth
motion by VAD instead of RMS rejects "loud room, nobody talking" false actives.

Runs the bundled `models/silero_vad.onnx` (Silero VAD, MIT-licensed) through
onnxruntime with the SAME provider preference as the detector. Deliberately no
torch dependency: the sidecar's whole point is a light runtime.

CRITICAL detail (cost a debugging session): Silero's 16kHz graph consumes
512-sample chunks but each call must be fed the previous chunk's last 64
samples PREPENDED (576 total). Omitting that context starves the model and its
output silently collapses to ~0.002 for everything — looks like "no speech ever"
rather than an error. See _CONTEXT below.
"""
import os
import subprocess
import tempfile
import wave

import numpy as np

from paths import model_path

_SR = 16000
_CHUNK = 512          # Silero's required chunk size at 16kHz
_CONTEXT = 64         # samples of previous-chunk lookback prepended per call
_STATE_SHAPE = (2, 1, 128)


def _providers():
    """Match detect.py's provider preference so VAD rides the same accelerator."""
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider"):
            if p in avail:
                return [p, "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


class _VadSession:
    """Lazily-built, process-wide Silero session (mirrors the Landmarker/Detector
    singleton pattern — model load is not free, and one clip reuses it)."""
    _sess = None

    @classmethod
    def get(cls):
        if cls._sess is None:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            path = model_path("silero_vad.onnx")
            try:
                cls._sess = ort.InferenceSession(path, sess_options=opts, providers=_providers())
            except Exception:
                cls._sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        return cls._sess


def _resolve_ffmpeg(explicit=None):
    # Same resolution order as asd._resolve_ffmpeg — kept local so vad.py has no
    # import cycle with asd.py.
    import shutil
    for candidate in (explicit, os.environ.get("TRK_FFMPEG"), os.environ.get("FFMPEG_BIN")):
        if candidate and os.path.exists(candidate):
            return candidate
    return shutil.which("ffmpeg") or "ffmpeg"


def extract_speech_prob(video_path, start, end, hop=1.0 / 24, ffmpeg=None):
    """Return (speech_prob_series, hop): speech_prob[i] in 0..1 is the Silero VAD
    probability at time i*hop (clip-local seconds), resampled from Silero's
    32ms native step to the requested hop.

    Returns (np.zeros(1), hop) if the clip has no audio track — same contract as
    extract_audio_rms, so a silent clip degrades to "never a VAD-gated speaker"
    exactly like it degrades to "never loud", never crashes."""
    dur = max(0.0, end - start)
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        cmd = [_resolve_ffmpeg(ffmpeg), "-v", "error", "-y", "-ss", str(start), "-t", str(dur),
               "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(_SR), "-f", "wav", wav]
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

    try:
        sess = _VadSession.get()
        state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        context = np.zeros((1, _CONTEXT), dtype=np.float32)
        sr_in = np.array(_SR, dtype=np.int64)

        probs = []
        i = 0
        while i + _CHUNK <= len(sig):
            chunk = sig[i:i + _CHUNK][None, :].astype(np.float32)
            x = np.concatenate([context, chunk], axis=1)   # (1, 576) — context is mandatory
            out, state = sess.run(["output", "stateN"], {"input": x, "state": state, "sr": sr_in})
            probs.append(float(out[0, 0]))
            context = x[:, -_CONTEXT:]
            i += _CHUNK

        if not probs:
            return np.zeros(1), hop
        probs = np.array(probs, dtype=np.float32)
        native_hop = _CHUNK / _SR

        # Resample Silero's native-step series onto the requested hop grid.
        n_hops = max(1, int(dur / hop) + 1)
        idx = np.clip((np.arange(n_hops) * hop / native_hop).round().astype(int), 0, len(probs) - 1)
        return probs[idx], hop
    except Exception:
        return np.zeros(1), hop
