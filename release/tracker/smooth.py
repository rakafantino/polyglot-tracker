# sidecar/tracker/smooth.py
"""Smooth keyframe positions within each continuous run so grid.py's periodic
sub-segment re-framing (TRK_MAX_SUBSEG, ~every 1.2s) glides like a stabilized
camera instead of snapping. A run boundary is grid.py's `cut` flag — True on
the first keyframe of every merged run (a real scene cut, speaker change, or a
fresh run after the min_seg merge); False on later sub-segment slices of that
SAME run (person drifting / camera zooming, no real cut). Smoothing never
blends across a `cut=True` boundary — that jump is real and intentional.

Analysis runs offline (the whole clip's keyframes are known before any frame
is rendered), so unlike a live filter we are NOT limited to causal blending.
A symmetric (non-causal) Gaussian-weighted average across the whole run is
what actual camera stabilizers use for exactly this reason: it has no
directional lag and no "settle after the fact" feel — every keyframe is
pulled toward its neighbors on BOTH sides, so the whole run reads as one
continuous glide instead of a chain of catch-up steps.

NOTE: an earlier version used a one-pass (causal-only) EMA blended toward the
previous OUTPUT keyframe. That still snapped visibly on short runs (2-3
keyframes) because the first keyframe of a run got no smoothing at all and
every later step only leaned toward one side. This version fixes that: it's
zero-phase and considers the entire run's trajectory at once.

Generic over panel keys (target/top/bottom/left/right) and whatever numeric
fields each panel dict carries (x/y/scale for `single`, x/y/w/h rects for
`split`) — no hardcoded schema, so it doesn't silently no-op (or crash) again
if grid.py's panel shape changes."""
import os
import numpy as np


def _panel_dicts(kf):
    return [k for k in ("target", "top", "bottom", "left", "right") if k in kf]


def _split_runs(keyframes):
    """Group keyframes into continuous runs, splitting at cut=True (or the
    very first keyframe, which starts the first run).

    Also force a split whenever the panel-key layout changes (e.g. a 'split'
    sub-segment collapses to 'single' mid-run via grid.py's rect-IoU dedup
    guard, line ~405). That collapse does NOT set cut=True — it's not a real
    scene cut, just this one sub-segment losing its second panel — so without
    this check a run could mix keyframes with different panel dicts
    ('top'/'bottom' vs 'target'). smooth_keyframes derives its panel set from
    run[0] only and indexes every other keyframe in the run with those same
    keys, so a mixed run means a later keyframe missing a key crashes with
    KeyError. Confirmed via a real production clip (source window
    686.24-746.24s) that reliably crashes analyze() without this guard."""
    runs = []
    cur = []
    cur_panels = None
    for i, kf in enumerate(keyframes):
        is_cut = bool(kf.get("cut", True))
        panels = tuple(_panel_dicts(kf))
        layout_changed = cur_panels is not None and panels != cur_panels
        if i == 0 or is_cut or layout_changed:
            if cur:
                runs.append(cur)
            cur = [kf]
        else:
            cur.append(kf)
        cur_panels = panels
    if cur:
        runs.append(cur)
    return runs


def _gaussian_kernel(radius, sigma):
    xs = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    return k / k.sum()


def smooth_keyframes(keyframes):
    if not keyframes or len(keyframes) < 2:
        return keyframes

    # radius=2, sigma=1.4 → weights ~[0.06,0.24,0.40,0.24,0.06]: each keyframe
    # blends mostly with its immediate neighbors on both sides but still pulls
    # a little from two steps away, which is what removes the "snap then
    # glide" feel of only-adjacent blending.
    radius = max(1, int(float(os.environ.get("TRK_SMOOTH_RADIUS", 2))))
    sigma = max(0.1, float(os.environ.get("TRK_SMOOTH_SIGMA", 1.4)))
    # Deadband: if a field's whole-run excursion is below this (normalized
    # 0..1 units), freeze it to the run median instead of letting the camera
    # micro-drift ("breathe") after tiny detection noise. Real auto-framers
    # hold perfectly still until movement is worth following.
    deadband = float(os.environ.get("TRK_SMOOTH_DEADBAND", 0.008))
    kernel = _gaussian_kernel(radius, sigma)

    out = []
    for run in _split_runs(keyframes):
        m = len(run)
        if m < 2:
            out.extend(run)
            continue

        panels = _panel_dicts(run[0])
        smoothed = {}
        for p in panels:
            fields = [f for f, v in run[0][p].items() if isinstance(v, (int, float))]
            for f in fields:
                # Missing panel/field on a later keyframe in the same run
                # (shouldn't normally happen, but stay defensive) falls back
                # to the first keyframe's value rather than crashing.
                arr = np.array([kf.get(p, run[0][p]).get(f, run[0][p].get(f, 0.0))
                                 for kf in run], dtype=float)
                # Edge-pad (repeat the boundary value) so the run's own start
                # and end don't get pulled toward zero — they're still
                # blended with their real inward neighbors, just not with
                # values outside the run.
                if float(arr.max() - arr.min()) < deadband:
                    smoothed[(p, f)] = np.full(m, float(np.median(arr)))
                    continue
                padded = np.pad(arr, radius, mode="edge")
                smoothed[(p, f)] = np.convolve(padded, kernel, mode="valid")

        for i, kf in enumerate(run):
            kf = dict(kf)
            for p in panels:
                raw = kf[p]
                blended = dict(raw)
                for f in raw:
                    if (p, f) in smoothed:
                        blended[f] = round(float(smoothed[(p, f)][i]), 4)
                kf[p] = blended
            out.append(kf)
    return out
