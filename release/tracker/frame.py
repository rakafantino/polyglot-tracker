# sidecar/tracker/frame.py
"""Per-sample framing decisions (mode + normalized crop coords), pre-smoothing.

Coordinate contract (see app/src/track-plan.js):
- single: the input frame is scaled to COVER a 9:16 window (tallW x tallH), then
  a crop of cw=tallW*scale, ch=tallH*scale is placed with top-left at
  x=(scaled_w-cw)*xn, y=(scaled_h-ch)*yn. So xn/yn are fractions of the
  available slack in the scaled image, and scale is the fraction of the
  full-height 9:16 window used.
- split: each panel crops cw=frame_w*w, ch=frame_h*w from the ORIGINAL frame at
  x=(frame_w-cw)*xn, y=(frame_h-ch)*yn; w~0.55.
"""

# 9:16 reference window (normalized fractions are resolution-independent).
import os
_TALL_W, _TALL_H = 1080.0, 1920.0
_FACE_FRAC = float(os.environ.get("TRK_FACE_FRAC", 0.55))  # face height as frac of 9:16 window
_SCALE_FLOOR = float(os.environ.get("TRK_SCALE_FLOOR", 0.34))  # tightest zoom (smaller=tighter)
_EYES_DOWN = 0.42      # face center sits ~42% down the crop (rule-of-thirds headroom)
_SPLIT_W = 0.55        # split panel crop fraction of frame width/height
_DEADZONE = 0.08       # min center move (fraction of frame width) to re-aim


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _single_target(face, frame_w, frame_h, prev=None):
    """Compute a single-mode target dict for one face bbox (pixels)."""
    x, y, x2, y2 = face["bbox"]
    fcx, fcy = (x + x2) / 2.0, (y + y2) / 2.0
    face_h = max(1.0, y2 - y)
    yaw = face.get("yaw", 0.0)

    sf = max(_TALL_W / frame_w, _TALL_H / frame_h)  # cover scale factor
    scaled_w, scaled_h = frame_w * sf, frame_h * sf

    # Pick scale so face height ≈ _FACE_FRAC of the crop height.
    face_h_scaled = face_h * sf
    scale = face_h_scaled / (_FACE_FRAC * _TALL_H)
    scale = _clamp(scale, _SCALE_FLOOR, 0.95)
    cw, ch = _TALL_W * scale, _TALL_H * scale

    fx_s, fy_s = fcx * sf, fcy * sf
    # Look-space: nudge the crop a little in the direction the subject faces so
    # they have room to look into (yaw + = turned image-right).
    look = (yaw / 45.0) * cw * 0.10
    crop_left = fx_s - cw / 2.0 + look
    crop_top = fy_s - ch * _EYES_DOWN

    slack_x = scaled_w - cw
    slack_y = scaled_h - ch
    xn = crop_left / slack_x if slack_x > 1e-6 else 0.5
    yn = crop_top / slack_y if slack_y > 1e-6 else 0.5
    xn, yn = _clamp(xn, 0.0, 1.0), _clamp(yn, 0.0, 1.0)

    # Hysteresis dead-zone: hold previous target unless the face moved enough.
    if prev is not None:
        move = abs((fcx / frame_w) - prev.get("_cx", fcx / frame_w))
        if move < _DEADZONE:
            held = dict(prev)
            held.pop("_cx", None)
            return {**held, "_cx": prev.get("_cx", fcx / frame_w)}

    return {"x": round(xn, 4), "y": round(yn, 4), "scale": round(scale, 4),
            "_cx": fcx / frame_w}


def _panel_target(face, frame_w, frame_h):
    x, y, x2, y2 = face["bbox"]
    fcx, fcy = (x + x2) / 2.0, (y + y2) / 2.0
    cw, ch = frame_w * _SPLIT_W, frame_h * _SPLIT_W
    xn = (fcx - cw / 2.0) / (frame_w - cw) if frame_w > cw else 0.5
    yn = (fcy - ch / 2.0) / (frame_h - ch) if frame_h > ch else 0.5
    return {"x": round(_clamp(xn, 0.0, 1.0), 4),
            "y": round(_clamp(yn, 0.0, 1.0), 4),
            "scale": _SPLIT_W}


def _commit_speaker_ids(samples, thr, dwell_s, margin):
    """Stateful anti-'bolak-balik' pass: pick ONE committed speaker id per sample.
    Hold the current speaker; only switch to a challenger once that challenger has
    been the top active face continuously for >= dwell_s AND beats the incumbent's
    score by `margin` (or the incumbent has gone silent). Returns list of ids
    (or None) aligned to samples."""
    def top_active(s):
        act = [f for f in s["faces"] if f.get("asd", 0.0) > thr]
        return max(act, key=lambda f: f.get("asd", 0.0)) if act else None

    committed = [None] * len(samples)
    cur_id = None
    challenger_id = None
    challenger_since = None
    for i, s in enumerate(samples):
        t = s["t"]
        by_id = {f.get("id"): f for f in s["faces"]}
        top = top_active(s)
        cur = by_id.get(cur_id)
        cur_active = cur is not None and cur.get("asd", 0.0) > thr

        if cur_id is None and top is not None:
            cur_id = top.get("id")                       # first lock-on
        elif top is not None and top.get("id") != cur_id:
            incumbent_asd = cur.get("asd", 0.0) if cur is not None else 0.0
            beats = top.get("asd", 0.0) >= incumbent_asd + margin
            if not cur_active or beats:
                # candidate challenger — require sustained dominance before switching
                if challenger_id == top.get("id"):
                    if (t - challenger_since) >= dwell_s:
                        cur_id = top.get("id")
                        challenger_id = None
                else:
                    challenger_id, challenger_since = top.get("id"), t
            else:
                challenger_id = None
        else:
            challenger_id = None
        committed[i] = cur_id
    return committed


def _split_windows(committed_ids, samples, win_s, min_switches, min_frac):
    """Find stretches of RAPID 2-person exchange (turn-taking), which read best
    as a split-screen rather than cutting back and forth. For each sample, look
    at the committed-speaker sequence within +/- win_s. If exactly two ids
    dominate (each present for >= min_frac of the window) and they alternate
    >= min_switches times, mark this sample as split for that ordered pair.
    Returns a list aligned to samples: None or (id_a, id_b)."""
    n = len(samples)
    out = [None] * n
    times = [s["t"] for s in samples]
    for i in range(n):
        t = times[i]
        idxs = [j for j in range(n) if abs(times[j] - t) <= win_s]
        seq = [committed_ids[j] for j in idxs if committed_ids[j] is not None]
        if len(seq) < 3:
            continue
        counts = {}
        for cid in seq:
            counts[cid] = counts.get(cid, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:2]
        if len(top) < 2:
            continue
        (a, ca), (b, cb) = top[0], top[1]
        if ca < min_frac * len(seq) or cb < min_frac * len(seq):
            continue
        # count alternations between a and b in order
        sw = 0
        last = None
        for cid in seq:
            if cid in (a, b):
                if last is not None and cid != last:
                    sw += 1
                last = cid
        if sw >= min_switches:
            out[i] = (a, b)
    return out


def decide_frames(samples, config, frame_w, frame_h):
    """samples: list of {t, faces:[{id, bbox(px), yaw, asd}]}. Returns a raw
    per-sample list of framing dicts (mode + target/top/bottom), pre-smoothing.
    """
    import os as _os
    # The jawOpen×RMS heuristic rarely sustains >0.4, so default the speaking
    # threshold lower; tune per footage via config or TRK_ASD_THRESHOLD.
    thr = config.get("asd_threshold") or float(_os.environ.get("TRK_ASD_THRESHOLD", 0.2))
    split_enabled = config.get("split_screen", False)
    dwell_s = float(_os.environ.get("TRK_DWELL", 0.8))     # min hold before switching speaker
    margin = float(_os.environ.get("TRK_SWITCH_MARGIN", 0.15))
    split_win = float(_os.environ.get("TRK_SPLIT_WINDOW", 3.0))    # +/- seconds
    split_switches = int(_os.environ.get("TRK_SPLIT_SWITCHES", 2)) # alternations to trigger
    split_frac = float(_os.environ.get("TRK_SPLIT_FRAC", 0.3))     # each speaker's min share

    def active_faces(s):
        return [f for f in s["faces"] if f.get("asd", 0.0) > thr]

    committed_ids = _commit_speaker_ids(samples, thr, dwell_s, margin)

    # Raw per-frame top speaker (who is ACTUALLY speaking this frame) — used to
    # detect rapid exchange. Must be the raw signal, NOT the sticky committed
    # sequence, or the stickiness would hide the alternation from itself.
    raw_top = []
    for s in samples:
        act = [f for f in s["faces"] if f.get("asd", 0.0) > thr]
        raw_top.append(max(act, key=lambda f: f.get("asd", 0.0))["id"] if act else None)

    # Rapid 2-person exchange → split-screen (Opus-style), instead of cutting.
    split_pairs = (_split_windows(raw_top, samples, split_win,
                                  split_switches, split_frac)
                   if split_enabled else [None] * len(samples))

    # Forward/backward-filled last-known bbox per id so a split panel can still
    # frame a speaker in frames where their detection momentarily dropped.
    last_bbox = {}
    seen_bbox = [dict() for _ in samples]
    for i, s in enumerate(samples):
        for f in s["faces"]:
            last_bbox[f["id"]] = f["bbox"]
        seen_bbox[i] = dict(last_bbox)
    # backfill for early frames
    fut = {}
    for i in range(len(samples) - 1, -1, -1):
        for f in samples[i]["faces"]:
            fut.setdefault(f["id"], f["bbox"])
        for k, v in fut.items():
            seen_bbox[i].setdefault(k, v)

    def bbox_for(i, cid):
        for f in samples[i]["faces"]:
            if f["id"] == cid:
                return f["bbox"]
        return seen_bbox[i].get(cid)

    # Decide stable top/bottom identity per contiguous split run so panels don't
    # swap. Keyed by frozenset(pair); the higher-or-left face is 'top'.
    pair_order = {}

    out = []
    prev_single = None
    for i, s in enumerate(samples):
        t = s["t"]
        faces = s["faces"]
        pair = split_pairs[i]

        if pair is not None:
            a, b = pair
            ba, bb = bbox_for(i, a), bbox_for(i, b)
            if ba is not None and bb is not None:
                key = frozenset((a, b))
                if key not in pair_order:
                    ay, by = _center(ba)[1], _center(bb)[1]
                    ax, bx = _center(ba)[0], _center(bb)[0]
                    pair_order[key] = (a, b) if (ay, ax) <= (by, bx) else (b, a)
                top_id, bot_id = pair_order[key]
                top_bb = bbox_for(i, top_id) or ba
                bot_bb = bbox_for(i, bot_id) or bb
                out.append({"t": t, "mode": "split",
                            "top": _panel_target({"bbox": top_bb}, frame_w, frame_h),
                            "bottom": _panel_target({"bbox": bot_bb}, frame_w, frame_h)})
                prev_single = None
                continue

        # single: the COMMITTED speaker (sticky), else largest face, else hold.
        act = active_faces(s)
        chosen = None
        cid = committed_ids[i]
        if cid is not None:
            chosen = next((f for f in faces if f.get("id") == cid), None)
        if chosen is None and act:
            chosen = max(act, key=lambda f: f.get("asd", 0.0))
        if chosen is None and faces:
            chosen = max(faces, key=lambda f: _area(f["bbox"]))

        if chosen is None:
            if prev_single is not None:
                tgt = dict(prev_single)
            else:
                tgt = {"x": 0.5, "y": 0.5, "scale": 0.85, "_cx": 0.5}
        else:
            tgt = _single_target(chosen, frame_w, frame_h, prev_single)
        prev_single = tgt
        clean = {k: v for k, v in tgt.items() if not k.startswith("_")}
        out.append({"t": t, "mode": "single", "target": clean})
    return out


def _center(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
