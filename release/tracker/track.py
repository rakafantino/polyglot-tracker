# sidecar/tracker/track.py
"""Thin wrapper over supervision.ByteTrack that assigns a stable tracker_id to
each face across frames. Feeds per-frame Detection lists in, gets
(tracker_id, Detection) pairs out."""
import numpy as np
import supervision as sv


class Tracker:
    def __init__(self):
        self.bt = sv.ByteTrack()

    def reset(self):
        """Drop all track state (call at a scene cut)."""
        self.bt.reset()

    def update(self, detections, frame_shape):
        """detections: list[Detection]. frame_shape: (h, w). Returns a list of
        (tracker_id, Detection). Detections that ByteTrack drops are omitted."""
        if not detections:
            # ByteTrack still needs a tick to age out lost tracks.
            self.bt.update_with_detections(sv.Detections.empty())
            return []
        xyxy = np.array([[d.x, d.y, d.x2, d.y2] for d in detections], dtype=float)
        conf = np.array([d.score for d in detections], dtype=float)
        sv_det = sv.Detections(xyxy=xyxy, confidence=conf,
                               class_id=np.zeros(len(detections), dtype=int))
        tracked = self.bt.update_with_detections(sv_det)
        out = []
        for i in range(len(tracked)):
            tid = tracked.tracker_id[i]
            if tid is None:
                continue
            # Map the tracked box back to the nearest input Detection so we keep
            # kps/yaw (ByteTrack only carries xyxy + confidence).
            tb = tracked.xyxy[i]
            best, best_d = None, 1e18
            for d in detections:
                dist = ((d.x - tb[0]) ** 2 + (d.y - tb[1]) ** 2 +
                        (d.x2 - tb[2]) ** 2 + (d.y2 - tb[3]) ** 2)
                if dist < best_d:
                    best_d, best = dist, d
            out.append((int(tid), best))
        return out
