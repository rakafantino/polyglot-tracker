# sidecar/tracker/detect.py
"""Face detection via InsightFace (SCRFD). Returns Detection objects with
bbox, score, 5-point kps, and yaw (from pose) for look-space framing."""
import os
from dataclasses import dataclass

import numpy as np
from insightface.app import FaceAnalysis
from paths import insightface_root

# SCRFD's native training resolution. The frame is letterboxed into a
# det_size box before inference, and the LONG side of that box alone sets how
# many pixels a face gets: a 1920-wide frame in a 640-wide box is scaled
# 0.333x whatever the box's height is.
DEFAULT_LONG_SIDE = int(os.environ.get("TRK_DET_LONG_SIDE", 640))

# SCRFD's coarsest stride. insightface computes its anchor grid as
# `input_dim // stride` while the ONNX graph convolves with SAME padding
# (ceil), so a det_size that is not a multiple of 32 desynchronises the two
# and the model's scores stop lining up with the anchor centres. Every
# det_size this module produces is rounded UP to a multiple of 32.
_STRIDE = 32

CPU_ONLY = ["CPUExecutionProvider"]


@dataclass
class Detection:
    x: float; y: float; x2: float; y2: float
    score: float
    kps: object          # 5x2 array or None
    yaw: float           # degrees, + = turned to image-right


def _providers():
    import onnxruntime as ort
    avail = ort.get_available_providers()
    for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider"):
        if p in avail:
            return [p, "CPUExecutionProvider"]
    return CPU_ONLY


def _ceil_stride(v):
    return max(_STRIDE, int(np.ceil(v / _STRIDE)) * _STRIDE)


def det_size_for(frame_w, frame_h, long_side=None):
    """The smallest det_size that detects faces at exactly the same scale as
    a square `long_side` box, without the wasted padding.

    insightface letterboxes the frame into det_size preserving aspect, so a
    16:9 frame in a 640x640 box occupies 640x360 and **44% of the inference
    input is black padding** -- pixels the network convolves over for nothing.
    Matching the box to the frame's aspect keeps `det_scale` identical (same
    face pixels, same detections: measured IoU 1.000 and recall 334/334 across
    the 5 training fixtures) while cutting the input area by that padding.
    Measured 1.66x faster on the CPU execution provider at 1080p.

    Returns a (w, h) tuple, both multiples of 32 (see _STRIDE).
    """
    long_side = int(long_side or DEFAULT_LONG_SIDE)
    frame_w, frame_h = max(1, int(frame_w)), max(1, int(frame_h))
    if frame_w >= frame_h:
        return (_ceil_stride(long_side), _ceil_stride(long_side * frame_h / frame_w))
    return (_ceil_stride(long_side * frame_w / frame_h), _ceil_stride(long_side))


def detector_candidates(frame_w, frame_h, long_side=None):
    """Ordered (det_size, providers) preferences, fastest-first, for a frame
    of this shape. `Detector` walks the list and keeps the first that a real
    inference actually survives.

    The order encodes measured numbers, not guesses:

      1. accelerator @ aspect-matched -- the ideal. Free 1.66x when it works.
      2. accelerator @ square         -- today's shipping config. Both CoreML
         (ORT 1.19.2) and DirectML compile a static-shape graph for this model
         and die at inference on ANY det_size but 640x640, so on those
         machines step 1 always fails and this is what runs -- i.e. no change
         from today. Only offered when the caller did not ask to shrink
         long_side, because falling back to a FULL-SIZE box would throw away
         the speed the caller explicitly asked for.
      3. CPU @ aspect-matched        -- for a caller who did shrink long_side.
         On a no-dGPU Windows box CPU at a reduced size measured FASTER than
         DirectML at 640 (77.8ms vs 141.7ms), so this is the useful path there.
    """
    accel = _providers()
    square = (_ceil_stride(long_side or DEFAULT_LONG_SIDE),) * 2
    # Escape hatch: TRK_DET_ASPECT_FIT=0 pins the old square box. Kept because
    # the aspect-matched shape is the one thing here that changes what the
    # ONNX graph is asked to do, so a runtime that mis-handles it needs an off
    # switch that does not require a rebuild -- and because it is how
    # bench/plan_diff.py isolates this change from the landmark one.
    fit = (det_size_for(frame_w, frame_h, long_side)
           if os.environ.get("TRK_DET_ASPECT_FIT", "1") not in ("0", "false")
           else square)
    out = [(fit, accel)]
    if int(long_side or DEFAULT_LONG_SIDE) == 640:
        out.append((square, accel))
    out.append((fit, CPU_ONLY))
    # Drop duplicates (a CPU-only machine's accel list IS the CPU list) while
    # keeping order.
    seen, uniq = set(), []
    for size, provs in out:
        key = (tuple(size), tuple(provs))
        if key not in seen:
            seen.add(key)
            uniq.append((tuple(size), list(provs)))
    return uniq


class Detector:
    def __init__(self, det_size=(640, 640), providers=None, probe_shape=None,
                 candidates=None):
        """Build the SCRFD session.

        `candidates` is an ordered [(det_size, providers), ...] preference
        list from `detector_candidates`; each is tried until one survives a
        real inference on a `probe_shape`-shaped frame. The probe is the
        point: an accelerated provider that cannot handle a det_size fails at
        the FIRST INFERENCE, not at session creation, so without it a bad
        choice surfaces as a clip that dies mid-analysis instead of one that
        quietly falls back.

        Called with no `candidates` it keeps the original single-config
        behaviour (and skips probing when no probe_shape is given), which is
        what the tests and the legacy Landmarker fallback path use.
        """
        cands = candidates or [(tuple(det_size), list(providers or _providers()))]
        errors = []
        for i, (size, provs) in enumerate(cands):
            try:
                self._build(size, provs)
                if probe_shape is not None:
                    self._probe(probe_shape)
                self.det_size, self.providers = tuple(size), list(provs)
                return
            except Exception as e:
                errors.append(f"{size}/{provs[0]}: {e.__class__.__name__}: {e}")
                if i == len(cands) - 1:
                    raise RuntimeError(
                        "no usable SCRFD configuration; tried " + " | ".join(errors))

    def _build(self, det_size, providers):
        # buffalo_l ships 5 ONNX models. We load ONLY `detection` (det_10g,
        # 17MB) now: the other four are unused, and landmark_3d_68 (143MB!)
        # existed solely to compute `yaw` — which NOTHING in the active layout
        # path reads (grid.py ignores it; the only consumer, frame.py, is dead
        # code). Dropping it removes a 143MB model load + its per-frame pose
        # inference, and — together with shipping det_10g in the bundle
        # (insightface_root) — eliminates InsightFace's ~325MB runtime download
        # of the full pack. `root=None` falls back to ~/.insightface for dev.
        self.app = FaceAnalysis(name="buffalo_l",
                                 root=insightface_root() or "~/.insightface",
                                 providers=providers,
                                 allowed_modules=["detection"])
        self.app.prepare(ctx_id=0, det_size=tuple(det_size))

    def _probe(self, shape):
        h, w = int(shape[0]), int(shape[1])
        self.app.get(np.zeros((max(1, h), max(1, w), 3), dtype=np.uint8))

    def detect(self, bgr):
        out = []
        for f in self.app.get(bgr):
            b = f.bbox
            # yaw is unused downstream (grid.py) — kept at 0.0 for the dataclass.
            out.append(Detection(float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                                 float(f.det_score), f.kps, 0.0))
        return out
