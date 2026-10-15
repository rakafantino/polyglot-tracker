# sidecar/tracker/mouth_motion.py
"""Per-track mouth-ROI pixel motion — the visual half of the motion+VAD
active-speaker signal ([[lokaclip-asd-algorithm-reverse-engineered]]).

Instead of MediaPipe's 468-landmark + blendshape pass (the 46-50% wall-time
bottleneck — see [[tracker-landmark-crop-fallback-bottleneck]]), this measures
how much the pixels inside a small mouth region-of-interest CHANGE from frame to
frame for the same tracked face. A talking mouth's ROI changes a lot; a silent,
still mouth barely changes — regardless of the head moving, because the ROI is
re-cropped around the mouth every frame from that frame's own keypoints.

Signal source is SCRFD's EXISTING 5-point keypoints (kps[3]=right mouth corner,
kps[4]=left mouth corner) that detect.py already returns — no extra model, no
extra inference. Reverse-engineered from LokaClip's `track.rs` (mouth_roi_w/h)
and validated on SOCL's own fixtures: mouth-ROI motion separates speech from
silence with AUC up to 0.91, and unlike a single-landmark distance it stays
positively correlated across every fixture.

State is per-track-id (needs the previous frame's ROI for the same face) so this
must be fed tracked detections and `reset()` at every scene cut / new clip, like
the Tracker and Landmarker.
"""
import os

import cv2
import numpy as np

_ROI_SIZE = int(os.environ.get("TRK_MOUTH_ROI_SIZE", 64))   # ROI is resized to this square
_ROI_PAD = float(os.environ.get("TRK_MOUTH_ROI_PAD", 1.6))  # mouth-corner span -> box width factor


class MouthMotion:
    def __init__(self, frame_shape=None):
        # Optional (H, W) to clamp ROI crops; can also be passed per call.
        self._hw = frame_shape
        self._prev = {}   # track_id -> previous ROI (float32 HxW) or None

    def reset(self):
        """Forget per-track ROI history. Call at each scene cut and per clip so
        motion never diffs across a cut (which would spike a false 'active')."""
        self._prev.clear()

    def _roi(self, frame, kps):
        if kps is None:
            return None
        H, W = frame.shape[:2]
        rm = np.asarray(kps[3], dtype=np.float32)   # right mouth corner
        lm = np.asarray(kps[4], dtype=np.float32)   # left mouth corner
        cx, cy = (rm + lm) / 2.0
        span = float(np.linalg.norm(rm - lm)) * _ROI_PAD
        if span < 2.0:
            return None
        half_w = span / 2.0
        half_h = span * 0.42                 # mouth box is a bit shorter than wide
        cy = cy + span * 0.10                # bias down: the mouth opens toward the chin
        x0, y0 = int(max(0, cx - half_w)), int(max(0, cy - half_h))
        x1, y1 = int(min(W, cx + half_w)), int(min(H, cy + half_h))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (_ROI_SIZE, _ROI_SIZE)).astype(np.float32)

    def motion(self, frame, kps, track_id):
        """0..~0.1 mouth-ROI motion for one tracked face this frame.

        Returns 0.0 on the first frame a track is seen (no previous ROI to diff),
        or whenever the mouth ROI can't be cropped (missing kps / off-frame).
        The value is the mean absolute per-pixel difference (normalised to 0..1)
        between this frame's mouth ROI and the same track's previous one."""
        roi = self._roi(frame, kps)
        if roi is None:
            # Keep whatever previous ROI we had; a momentary miss shouldn't spike
            # motion on the next hit (that would diff across a gap).
            return 0.0
        prev = self._prev.get(track_id)
        self._prev[track_id] = roi
        if prev is None or prev.shape != roi.shape:
            return 0.0
        return float(np.mean(np.abs(roi - prev))) / 255.0
