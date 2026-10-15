# sidecar/tracker/grid.py
"""Simple layout logic for smart face tracking. Only two modes:

  * single — one speaker, centred, medium framing.
  * split  — two speakers concurrently active. Orientation follows the output
             aspect: landscape (16:9) → left/right; portrait or square → top/
             bottom. The JS-side vfSplitTrack picks the ffmpeg stack direction
             from the format.

Speaker = jawOpen×audio-RMS (ASD) crossing the threshold within a short window.
No grid, no wide-blur, no zoom past source. Panels stay medium (much body/
context) so we never upscale.

Output keyframes:
  {t, mode:'single', target:{x,y,scale}}
  {t, mode:'split',  top:{x,y,scale},   bottom:{x,y,scale}}      # 9:16 / 1:1
  {t, mode:'split',  left:{x,y,scale},  right:{x,y,scale}}       # 16:9

The JS side accepts either {top,bottom} or {left,right} depending on orientation.
"""
import os
import numpy as np

# Reference (W, H) per output format. The user's format choice:
#   Face 9:16 / Blur BG 9:16 → portrait 1080×1920
#   Square 1:1              → 1080×1080
#   Landscape 16:9          → 1920×1080
FORMAT_REF = {
    "9:16": (1080.0, 1920.0),
    "1:1":  (1080.0, 1080.0),
    "16:9": (1920.0, 1080.0),
}

_LANDSCAPE_ASPECTS = {"16:9"}


def _cfg_f(name, d):
    return float(os.environ.get(name, d))


def _center(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (ua + ub - inter + 1e-6)


def _single_target(bbox, frame_w, frame_h, face_in_panel, ref_w, ref_h, others=None):
    """Medium single-speaker framing → {x,y,scale} for the JS single builder.

    Coordinate contract (matches track-plan.js vfSingleTrack): the source is
    cover-scaled to fit an output ref_w×ref_h canvas, then a scale·ref_h tall
    window is cropped at fractional slack (x, y). We never upscale past the
    scaled source (cap ch by scaled_h), so output stays HD."""
    sf = max(ref_w / frame_w, ref_h / frame_h)
    scaled_w, scaled_h = frame_w * sf, frame_h * sf
    x1, y1, x2, y2 = bbox
    fcx, fcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    fh = max(1.0, y2 - y1)
    # crop height derived from desired face-fraction of the panel
    ch = min(ref_h, scaled_h, (fh * sf) / max(0.05, face_in_panel))
    scale = float(np.clip(ch / ref_h, 0.3, 1.0))
    cw, ch = ref_w * scale, ref_h * scale
    left = fcx * sf - cw / 2.0
    top = fcy * sf - 0.38 * ch   # rule-of-thirds headroom
    sx, sy = scaled_w - cw, scaled_h - ch
    # [NO-HALF-FACE-2026-10-15] People sitting shoulder to shoulder: a window
    # centred on one face cut the neighbour in half at the crop edge (measured
    # 1.3-1.5% of frames on two real podcast clips). Slide the window the
    # least distance so every other face is either fully in or fully out,
    # never past the main face's own edges. Heads are widened 25% a side for
    # hair/ears. No valid slide -> keep the centred window.
    if others and sx > 1e-6:
        def _span(b, grow):
            w = (b[2] - b[0]) * grow
            return ((b[0] - w) * sf, (b[2] + w) * sf)
        mx1, mx2 = _span(bbox, 0.25)
        lo = max(0.0, mx2 - cw)
        hi = min(sx, mx1)
        if lo <= hi:
            spans = [_span(o, 0.25) for o in others]

            def _score(l):
                cut = full = 0
                for a, b in spans:
                    ov = min(l + cw, b) - max(l, a)
                    if 0.10 * (b - a) < ov < 0.90 * (b - a):
                        cut += 1
                    elif ov >= 0.90 * (b - a):
                        full += 1
                return cut, -full
            # Fewest half-faces first, then the MOST neighbours fully in:
            # whichever of two people sitting close is "main" this slice, the
            # window that holds both is the same window — no swing (and no
            # half-face mid-swing) when the speaker changes.
            base = float(np.clip(left, lo, hi))
            cands = {base, lo, hi}
            for a, b in spans:
                cands.update((a, b - cw, b, a - cw))
            best_l, best_k = base, _score(base) + (0.0,)
            for c in cands:
                c = float(np.clip(c, lo, hi))
                k = _score(c) + (abs(c - left),)
                if k < best_k:
                    best_l, best_k = c, k
            left = best_l
    xn = float(np.clip(left / sx, 0.0, 1.0)) if sx > 1e-6 else 0.5
    yn = float(np.clip(top / sy, 0.0, 1.0)) if sy > 1e-6 else 0.5
    return {"x": round(xn, 4), "y": round(yn, 4), "scale": round(scale, 4)}


def _panel_target(bbox, frame_w, frame_h, panel_w, panel_h, face_in_panel,
                   other_bbox=None):
    """Source-normalized crop rect {x,y,w,h} for one split panel.

    The rect has the panel's aspect, is sized so the face is ~face_in_panel of
    the crop height, and is positioned with rule-of-thirds headroom.

    `other_bbox` is the OTHER split panel's target face — OR a list of every
    other visible face (the split partner plus any bystanders). Accepting a
    list matters for dense multi-person shots (5+ people seated close
    together): clamping against only the immediate split partner still let a
    THIRD person bleed into the edge of the crop (confirmed 2026-08-18 on a
    5-person panel-show fixture — panel A's crop, clamped only against
    partner B, still included a bystander seated between/beside them). Each
    bystander's clamp only ever SHRINKS cw/ch further (never grows it back),
    so looping the same per-pair clamp over every bystander converges to the
    tightest constraint from all of them — same math, just applied per-face
    instead of once. When given, the panel is clamped so it never crosses the
    midpoint between the target face and EACH other face along the axis that
    separates them. This is what stops split panels from ending up showing
    the same slice of a two-shot podcast, or a bystander at the edge, when
    the source is much wider than a single-panel frame."""
    x1, y1, x2, y2 = bbox
    fcx, fcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    fh = max(1.0, y2 - y1)
    panel_asp = panel_w / panel_h
    ch = fh / max(0.05, face_in_panel)          # crop height in source px
    cw = ch * panel_asp

    # Global cap: never grow past a fraction of the source. This is the safety
    # net for the case where there's only one speaker (other_bbox=None).
    cw_cap_frac = _cfg_f("TRK_PANEL_MAX_W_FRAC", 0.55)
    cw = min(cw, frame_w * cw_cap_frac)
    ch = cw / panel_asp

    if cw > frame_w:                            # wider than source (tiny src)
        cw = float(frame_w)
        ch = cw / panel_asp
    if ch > frame_h:                            # taller than source
        ch = float(frame_h)
        cw = min(float(frame_w), ch * panel_asp)

    # Pair cap: the panel may extend inward toward the other speaker up to that
    # speaker's inner face edge (not just the midpoint) — this lets each panel
    # keep a comfortable medium framing on its own face while still not showing
    # the other person as a visible face in the crop.
    if other_bbox is not None:
        # Normalize to a list of bystander bboxes — accept either a single
        # bbox (4 numbers) or an iterable of bboxes, so existing callers
        # (single split-partner) and the new multi-person case both work.
        if len(other_bbox) == 4 and all(isinstance(v, (int, float)) for v in other_bbox):
            others = [other_bbox]
        else:
            others = list(other_bbox)

        # Small guard band so the OTHER face doesn't touch the panel edge.
        guard = _cfg_f("TRK_PAIR_GUARD", 0.02)
        for ob in others:
            ox1, oy1, ox2, oy2 = ob
            ocx = (ox1 + ox2) / 2.0
            ocy = (oy1 + oy2) / 2.0
            dx = abs(ocx - fcx); dy = abs(ocy - fcy)
            if dx >= dy:                         # separated horizontally
                mid_x = (fcx + ocx) / 2.0
                if fcx < ocx:                    # this panel is on the LEFT
                    right_limit = mid_x - guard * frame_w
                    max_half_w = max(1.0, right_limit - fcx)
                    max_w = max(1.0, right_limit)
                else:                            # this panel is on the RIGHT
                    left_limit = mid_x + guard * frame_w
                    max_half_w = max(1.0, fcx - left_limit)
                    max_w = max(1.0, frame_w - left_limit)
                if cw / 2.0 > max_half_w:
                    cw = 2 * max_half_w
                    ch = cw / panel_asp
                if cw > max_w:
                    cw = max_w
                    ch = cw / panel_asp
            else:                                # separated vertically (rare)
                mid_y = (fcy + ocy) / 2.0
                if fcy < ocy:                    # this panel is on TOP
                    bot_limit = mid_y - guard * frame_h
                    max_half_h = max(1.0, bot_limit - fcy)
                    max_h = max(1.0, bot_limit)
                else:                            # this panel is on BOTTOM
                    top_limit = mid_y + guard * frame_h
                    max_half_h = max(1.0, fcy - top_limit)
                    max_h = max(1.0, frame_h - top_limit)
                if ch / 2.0 > max_half_h:
                    ch = 2 * max_half_h
                    cw = ch * panel_asp
                if ch > max_h:
                    ch = max_h
                    cw = ch * panel_asp

    # [EDGE-CENTER] Frame-edge cap on cw: without this, a face near the source's
    # left/right edge (routine for side-by-side two-shots, where each speaker
    # sits toward their own outer third) forces `left = fcx - cw/2` negative or
    # past `frame_w - cw`, and the clamp below shoves the crop against the
    # frame edge — the face is no longer centered in it, and the OTHER side of
    # the crop (toward frame center) fills with dead space: couch, shelf,
    # candy jars. Capping cw to twice the smaller face-to-edge distance means
    # the clamp is never needed, so the face stays horizontally centered; the
    # side effect (crop zooms in tighter when the face truly is near an edge)
    # is strictly better than showing empty background instead of the speaker.
    edge_cap_w = 2.0 * max(1.0, min(fcx, frame_w - fcx))
    if cw > edge_cap_w:
        cw = edge_cap_w
        ch = cw / panel_asp

    left = fcx - cw / 2.0
    top = fcy - 0.38 * ch                       # rule-of-thirds headroom
    left = min(max(0.0, left), frame_w - cw)
    top = min(max(0.0, top), frame_h - ch)
    return {"x": round(left / frame_w, 5), "y": round(top / frame_h, 5),
            "w": round(cw / frame_w, 5), "h": round(ch / frame_h, 5)}


def build_grid_keyframes(samples, config, frame_w, frame_h, output_aspect="9:16"):
    thr = config.get("asd_threshold") or _cfg_f("TRK_ASD_THRESHOLD", 0.2)
    win = _cfg_f("TRK_CAST_WINDOW", 2.0)
    min_seg = _cfg_f("TRK_MIN_SEG", 1.2)
    face_in_panel = _cfg_f("TRK_FACE_IN_PANEL", 0.15)   # more body, less zoom
    max_people = 2                                       # HARD CAP — max 2 panels
    ref_w, ref_h = FORMAT_REF.get(output_aspect, FORMAT_REF["9:16"])
    landscape = output_aspect in _LANDSCAPE_ASPECTS

    n = len(samples)
    if n == 0:
        return [{"t": 0.0, "mode": "single",
                 "target": {"x": 0.5, "y": 0.5, "scale": 0.9}}]
    times = [s["t"] for s in samples]

    # [FACE-CUT-2026-10-13] PySceneDetect misses camera switches between
    # angles of the SAME dark studio (measured: 2-shot -> close-up -> 2-shot
    # inside 2.3 s with zero cut flags), so a split kept its panels across a
    # close-up and cut faces in half. A sample whose visible face set shares
    # no track id with the previous sample's (both non-empty) is a camera
    # switch: mark it as a cut so runs, medians and the min-seg merge break
    # there. ByteTrack keeps ids through motion and short occlusion, so a
    # fully disjoint set means the shot itself changed.
    # Judged on GEOMETRY, not ids: ByteTrack can hand an old id to a face in
    # the new shot (seen live: the close-up reused the 2-shot's id), so "no
    # face continues near where a face just was, at a similar size" is the
    # switch signal. 1/12 s apart, a real face moves far less than 8% of W.
    prev_faces = None
    for s in samples:
        cur = s["faces"]
        if cur and prev_faces:
            cont = False
            for f in cur:
                (fx, fy), fh_ = _center(f["bbox"]), max(1.0, f["bbox"][3] - f["bbox"][1])
                for g in prev_faces:
                    (gx, gy), gh = _center(g["bbox"]), max(1.0, g["bbox"][3] - g["bbox"][1])
                    if (abs(fx - gx) < 0.08 * frame_w and abs(fy - gy) < 0.08 * frame_h
                            and max(fh_, gh) / min(fh_, gh) < 1.3):
                        cont = True; break
                if cont:
                    break
            if not cont:
                s["cut"] = True
        if cur:
            prev_faces = cur

    # [STATIC-FACE-2026-10-15] Posters, paintings and cartoon faces on the set
    # wall are detected as faces (SCRFD scores them like people). Measured on
    # two real multi-cam podcast clips: within one shot a painted face's box
    # centre spreads < 0.0093 of its own height (detector noise only), a real
    # person's >= 0.0112 even sitting still in a wide shot. A face that is
    # that still for a whole shot AND (almost) never active (<=10% of its
    # samples: motion+VAD ASD flickers on posters too) is dropped from
    # that shot, so it can never become the single target, a split panel or a
    # bystander that squeezes the real panels.
    static_std = _cfg_f("TRK_STATIC_FACE_STD", 0.010)
    if static_std > 0:
        shot, shots = [], []
        for s in samples:
            if s.get("cut") and shot:
                shots.append(shot); shot = []
            shot.append(s)
        if shot:
            shots.append(shot)
        # One entry per (shot, track id): where it sits, how much it moves.
        entries = []
        for si_, sh in enumerate(shots):
            per = {}
            for s in sh:
                for f in s["faces"]:
                    per.setdefault(f["id"], []).append(f)
            for cid, fs in per.items():
                cx = np.array([_center(f["bbox"])[0] / frame_w for f in fs])
                cy = np.array([_center(f["bbox"])[1] / frame_h for f in fs])
                h = float(np.mean([(f["bbox"][3] - f["bbox"][1]) / frame_h for f in fs])) or 1e-6
                entries.append({
                    "shot": si_, "id": cid, "n": len(fs), "x": float(cx.mean()), "y": float(cy.mean()), "h": h,
                    "spread": float(np.hypot(cx.std(), cy.std())) / h,
                    "active": sum(1 for f in fs if f.get("asd", 0.0) > thr),
                })
        # A still silent listener in one short shot looks like a poster, but
        # the camera comes back to them and they move then. A poster is still
        # in EVERY shot that shows that spot (measured: a real host sitting
        # still for 1.5 s was 0.008, the same host 0.015-0.05 in other shots).
        def _same_spot(a, b):
            return (abs(a["x"] - b["x"]) < 0.03 and abs(a["y"] - b["y"]) < 0.03
                    and max(a["h"], b["h"]) / min(a["h"], b["h"]) < 1.2)
        drop = set()
        for e in entries:
            if e["n"] < 8 or e["spread"] >= static_std or e["active"] > max(2, 0.1 * e["n"]):
                continue
            # "Moves elsewhere" needs a clear margin: on tiny faces (h ~0.05)
            # detector jitter alone reaches 0.011 (measured, a wall poster).
            if any(o is not e and _same_spot(e, o) and o["n"] >= 8
                   and (o["spread"] >= 1.5 * static_std or o["active"] > max(2, 0.1 * o["n"])) for o in entries):
                continue
            drop.add((e["shot"], e["id"]))
        if drop:
            for si_, sh in enumerate(shots):
                for s in sh:
                    s["faces"] = [f for f in s["faces"] if (si_, f["id"]) not in drop]

    # Only faces that speak at some point qualify. Static posters and TV heads
    # rarely trip the ASD bar consistently.
    active_count = {}
    for s in samples:
        for f in s["faces"]:
            if f.get("asd", 0.0) > thr:
                active_count[f["id"]] = active_count.get(f["id"], 0) + 1
    speakers = {cid for cid, c in active_count.items()
                if c >= int(_cfg_f("TRK_MIN_ACTIVE_SAMPLES", 3))}

    # Ids seen at least a handful of times overall — used as the presence pool
    # (independent of ASD, so quiet podcast guests still qualify).
    seen_count = {}
    for s in samples:
        for f in s["faces"]:
            seen_count[f["id"]] = seen_count.get(f["id"], 0) + 1
    min_seen = int(_cfg_f("TRK_MIN_SEEN", max(3, n // 8)))
    presence_pool = {cid for cid, c in seen_count.items() if c >= min_seen}

    # Per-sample class: SPLIT trigger requires ≥2 DIFFERENT SPEAKERS whose mouths
    # opened somewhere in the wider ±speaker_win window (default 3s). Turn-taking
    # podcasts still qualify — reflandscape's two hosts alternate, and both
    # trigger jawOpen in any 3s window. A single-speaker wide shot (src2, one
    # host talking while others just sit) stays SINGLE. Quiet-listener panels
    # ("orang yang cuma gerak doang") are gone.
    speaker_win = _cfg_f("TRK_SPEAKER_WIN", 8.0)

    # [SPLIT-VISIBLE-2026-10-13] The ±speaker_win window says WHO talks, not
    # who is ON SCREEN now. Multi-cam podcasts cut to a one-person close-up
    # every few seconds while both hosts keep talking, so the window stayed
    # "2 speakers" and the second panel pointed at the empty spot the other
    # person used to occupy (measured on a real 6-clip session: 19.8% of
    # split time had a panel with no face in it). Split now also needs two
    # spatially distinct faces actually visible in most samples of a short
    # local window around t.
    pool_ids = speakers | presence_pool
    min_sep = _cfg_f("TRK_SPLIT_MIN_SEP", 0.18)

    def _n_visible(sample):
        cs = []
        for f in sample["faces"]:
            if f["id"] not in pool_ids:
                continue
            cx, cy = _center(f["bbox"])
            cx /= frame_w; cy /= frame_h
            if all(((cx - a) ** 2 + (cy - b) ** 2) ** 0.5 >= min_sep for a, b in cs):
                cs.append((cx, cy))
        return len(cs)

    vis = [_n_visible(s) for s in samples]
    vis_win = _cfg_f("TRK_SPLIT_VIS_WIN", 0.5)
    vis_frac = _cfg_f("TRK_SPLIT_VIS_FRAC", 0.6)

    def _two_visible(i):
        t = times[i]
        tot = two = 0
        for j in range(n):
            if abs(times[j] - t) <= vis_win:
                tot += 1
                two += vis[j] >= 2
        return tot > 0 and two >= vis_frac * tot

    cls = []
    for i, s in enumerate(samples):
        t = s["t"]
        active_speakers = set()
        pres_count = {}
        pres_window = 0
        for j in range(n):
            if abs(times[j] - t) <= win:
                pres_window += 1
                seen_this = set()
                for f in samples[j]["faces"]:
                    if f["id"] in speakers and f.get("asd", 0.0) > thr \
                            and abs(times[j] - t) <= speaker_win:
                        active_speakers.add(f["id"])
                    if f["id"] in presence_pool:
                        seen_this.add(f["id"])
                for cid in seen_this:
                    pres_count[cid] = pres_count.get(cid, 0) + 1
        # Only ≥2 different speakers (open-mouth in wide window) triggers split.
        # Presence-based split was too eager on wide-shot panels — a 5-person
        # podcast where only 1 host talks would split onto host + random listener.
        if len(active_speakers) >= 2 and _two_visible(i):
            cls.append(("split",))
        else:
            cls.append(("single",))

    # Merge short flickers into previous run for stability. Also force a new
    # run at scene-cut boundaries (samples[i]['cut']=True) — otherwise the
    # median bbox over the whole run averages positions across camera cuts and
    # the static crop lands between them, missing every speaker.
    runs = []
    for i in range(n):
        is_cut = samples[i].get("cut", False) if i > 0 else False
        if runs and runs[-1][2] == cls[i] and not is_cut:
            runs[-1][1] = i
        else:
            runs.append([i, i, cls[i]])
    merged = []
    for r in runs:
        dur = times[r[1]] - times[r[0]]
        starts_at_cut = samples[r[0]].get("cut", False) if r[0] > 0 else False
        if merged and dur < min_seg and not starts_at_cut:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    # Sub-segment long runs: within a merged run, the framing takes ONE median
    # bbox. If speakers drift smoothly (no scene cut but the person walks or
    # the camera zooms) the median lands between positions and the crop misses
    # them at both ends. Chop long runs into <= max_sub second slices so panels
    # get re-framed as positions evolve.
    # [SMOOTH-V1] Track which subdivided slice is the FIRST of its parent
    # merged run (a real discontinuity — scene cut, speaker change, or a fresh
    # run after the min_seg merge) vs. a later periodic re-frame slice within
    # the SAME continuous run (person drifting/camera zooming — no real cut).
    # Downstream smoothing must never blend across the former but SHOULD
    # blend across the latter, and the keyframe list had no way to tell them
    # apart before this — every keyframe looked like an independent frame.
    # [SPLIT-SUBSEG-2026-08-06] split runs NEVER merge sub-segments back
    # together downstream — _assign_seg() below gives every split sub-slice
    # its own _seg because panelChain (track-plan.js) has no time-based
    # interpolation for the {x,y,w,h} rect format yet, only a static per-_seg
    # crop from its first keyframe (documented at _assign_seg's definition).
    # A single dense max_sub applied to BOTH modes means every split slice
    # becomes its OWN ffmpeg render call — measured on a real multi-speaker
    # window: TRK_MAX_SUBSEG=0.1 produced 250 segments for a 60s clip (most
    # of it split-mode), and that many tiny re-encodes compounds enough
    # per-segment rounding to inflate the final duration by several seconds.
    # Single mode has no such ceiling (piecewiseLerp interpolates smoothly
    # across many keyframes within ONE _seg, capped safely by
    # track-plan.js's simplifyToBudget), so it can stay dense. Split mode
    # gets its own, coarser knob — defaults to the ORIGINAL 1.2s so existing
    # split-mode behaviour is unchanged unless explicitly tuned.
    #
    # [DENSE-DEFAULT-2026-08-06] 1.2s was only ever validated as "safe to
    # raise" (25 keyframes/30s -> 240, render still succeeds via the budget
    # cap) — it was never actually shipped as the default, so every real
    # render kept using the coarse ~1 keyframe/1.2s spacing this whole
    # investigation was about fixing. A real user comparison against the
    # dense render confirmed it reads as genuinely smoother ("cocok dengan
    # yang 11.2mb"), so 0.1s (near the sample_fps=12 floor — the densest
    # value that adds real information) is now the actual default.
    # [SINGLE-SUBSEG-2026-09-03] Tried raising this to 1.2s (matching split)
    # to fight a "hunting" crop in single mode — see SINGLE-LOCK below for the
    # actual root cause (candidate identity flip-flopping between two
    # comparably-active people at every resegmentation). The lock fixes that
    # independent of window size, so the DENSE-DEFAULT-2026-08-06 tuning
    # (validated against a real user comparison) stays put — reslicing every
    # 0.1s is fine once `best` can't swap identity mid-run.
    max_sub_single = _cfg_f("TRK_MAX_SUBSEG", 0.1)
    max_sub_split = _cfg_f("TRK_MAX_SUBSEG_SPLIT", 1.2)
    subdivided = []
    for (si, ei, rcls) in merged:
        max_sub = max_sub_single if rcls == ("single",) else max_sub_split
        dur = times[ei] - times[si]
        if dur <= max_sub:
            subdivided.append((si, ei, rcls, True))
            continue
        # split into N slices of <= max_sub each
        n_slice = max(1, int((dur + max_sub - 1e-6) // max_sub))
        # index step per slice
        step = (ei - si + 1) / n_slice
        cur = si
        for k in range(n_slice):
            end_i = si + int(round((k + 1) * step)) - 1
            if k == n_slice - 1:
                end_i = ei
            end_i = max(cur, min(ei, end_i))
            subdivided.append((cur, end_i, rcls, k == 0))
            cur = end_i + 1
            if cur > ei:
                break
    merged = subdivided

    def _others_in(si, ei, main_bbox):
        """Median box of every OTHER face seen in samples[si..ei], straight
        from detection — _seg_bboxes' dedup merges neighbours sitting closer
        than TRK_DEDUP_CDIST, which are exactly the ones a crop cuts in half.
        A box overlapping the main face is the same person, not a neighbour."""
        per = {}
        for i in range(si, ei + 1):
            for f in samples[i]["faces"]:
                per.setdefault(f["id"], []).append(f["bbox"])
        mw = max(1.0, main_bbox[2] - main_bbox[0])
        mcx = (main_bbox[0] + main_bbox[2]) / 2.0
        out = []
        for bxs in per.values():
            if len(bxs) < max(1, 0.3 * (ei - si + 1)):
                continue
            b = tuple(np.median(np.asarray(bxs, dtype=float), axis=0))
            if abs((b[0] + b[2]) / 2.0 - mcx) < 0.6 * mw:
                continue
            out.append(b)
        return out

    def _seg_bboxes(si, ei, floor=None):
        """{id: (median_bbox, activity, seen)} for ids that survive the
        detection-in-segment gate and IoU-dedup (kills ByteTrack drift).
        Includes quiet listeners from `presence_pool` so 2-shot podcasts
        still land two panels even when only one person is currently talking."""
        seg_len = ei - si + 1
        per = {}
        allowed = speakers | presence_pool
        for i in range(si, ei + 1):
            for f in samples[i]["faces"]:
                if f["id"] not in allowed:
                    continue
                d = per.setdefault(f["id"], {"bxs": [], "asd": 0.0, "act": 0})
                d["bxs"].append(f["bbox"])
                d["asd"] += f.get("asd", 0.0)
                if f.get("asd", 0.0) > thr:
                    d["act"] += 1
        # Detection floor keeps a panel tied to someone actually seen in this
        # run. 15% is loose enough for a second host whose MediaPipe/SCRFD
        # coverage is intermittent (src4 secondary speaker showed at 22% of
        # frames — 25% floor cut them out, dedup then collapsed to single).
        if floor is None:
            floor = _cfg_f("TRK_SEG_DET_FLOOR", 0.15)
        raw = {}
        for cid, d in per.items():
            if len(d["bxs"]) < max(1, floor * seg_len):
                continue
            med = tuple(np.median(np.asarray(d["bxs"], dtype=float), axis=0))
            raw[cid] = (med, d["act"], len(d["bxs"]))
        # Dedup — same person, multiple IDs → keep the more active. Two signals:
        #   (a) IoU > 0.15 (tightened from 0.30 — ByteTrack drift often has 0.15
        #       to 0.25 overlap because the box drifts a little each frame)
        #   (b) center-distance < 0.13 of the frame diagonal (catches drift that
        #       shrinks/grows the box while keeping the centre near the same
        #       person — IoU alone missed this in src4: id1 and id2 had IoU 0.19
        #       and centres only 0.09 apart, i.e. the same woman).
        def _too_close(a, b):
            # Signal 1: center-distance (tightened default 0.13 → 0.18 to cover
            # the grey-zone where ByteTrack re-entry IDs land: cdist 0.13–0.18)
            cax = (a[0] + a[2]) / 2.0; cay = (a[1] + a[3]) / 2.0
            cbx = (b[0] + b[2]) / 2.0; cby = (b[1] + b[3]) / 2.0
            dx = (cax - cbx) / frame_w
            dy = (cay - cby) / frame_h
            cdist = (dx * dx + dy * dy) ** 0.5
            if cdist < _cfg_f("TRK_DEDUP_CDIST", 0.18):
                return True
            # Signal 2: face-size ratio — same person (re-entry) → nearly identical
            # bbox height. Different people at different depths differ by > 1.4×.
            # Only active when cdist < TRK_DEDUP_CDIST_SIZE (0.25) — not applied to
            # pairs that are clearly far apart on screen.
            ha = max(1.0, a[3] - a[1])
            hb = max(1.0, b[3] - b[1])
            size_ratio = max(ha, hb) / min(ha, hb)
            if cdist < _cfg_f("TRK_DEDUP_CDIST_SIZE", 0.25) and size_ratio < _cfg_f("TRK_DEDUP_SIZE_RATIO", 1.35):
                return True
            return False
        ordered = sorted(raw, key=lambda k: (-raw[k][1], -raw[k][2]))
        kept = []
        for cid in ordered:
            dup = False
            for kid in kept:
                if _iou(raw[cid][0], raw[kid][0]) > _cfg_f("TRK_DEDUP_IOU", 0.15) \
                        or _too_close(raw[cid][0], raw[kid][0]):
                    dup = True; break
            if not dup:
                kept.append(cid)
        # Return with seen as third element so callers can rank presence too.
        return [(cid, raw[cid][0], raw[cid][1], raw[cid][2]) for cid in kept]

    keyframes = []
    seg_ix = -1
    # [SEG-MERGE 2026-08-06] seg_ix used to increment on EVERY sub-segment
    # unconditionally, so `_seg` (and therefore the JS-side render segment —
    # see track-plan.js's `_segKey`/`planToSegments`) was unique per ~1.2s
    # sub-segment even within one unbroken continuous run. `_segKey` prefers
    # `_seg` when present, so `planToSegments` never merged same-run
    # sub-segments into one job — every keyframe became its own isolated,
    # STATIC-position ffmpeg segment, hard-concatenated to the next. The
    # piecewiseLerp smoothstep interpolation in vfSingleTrack/panelChain (for
    # exactly this purpose — see track-plan.js's own comment: "the crop
    # glides into and out of each move instead of snapping") never fired in
    # practice: verified `planToSegments` on a real 55-keyframe trackPlan
    # produced 52 segments, ALL with exactly 1 keyframe. Reported as "motion
    # kurang smooth sepanjang klip" even after fixing the false-positive
    # scene-cut bug (2026-08-06) — that fix reduced how far apart the
    # snapshots landed, but every ~1.2s boundary was ALREADY a hard cut
    # regardless of whether `cut` was true or false.
    #
    # Fix: `_seg` only advances when this keyframe is a REAL cut (`cut=True`
    # for this specific keyframe, not the raw `run_start` loop variable —
    # several branches below force `cut=False` on an otherwise-run_start
    # boundary specifically to keep continuity, e.g. the no-detection and
    # degenerate-panel fallbacks). Non-cut single-mode continuations share
    # the previous keyframe's `_seg`, so planToSegments merges them into one
    # job and the existing piecewiseLerp path actually renders a continuous
    # glide within a run instead of a staircase of static holds.
    #
    # Deliberately SINGLE-MODE ONLY. `panelChain` (track-plan.js) branches on
    # whether a panel keyframe carries `{w,h}` (the format every split
    # keyframe here uses) vs legacy `{scale}` — the `{w,h}` branch has NO
    # time-based expression at all, it always renders `panelKfs[0]` as a
    # static crop for the whole segment. Merging split sub-segments the same
    # way single ones are merged would silently FREEZE panel position at the
    # first sub-segment's rect for the whole merged run instead of updating
    # as the tracker re-measures — a real regression for src2/src3/src4-style
    # multi-speaker clips. Fixing that needs panelChain to grow its own
    # rect-interpolation path first; out of scope here, so split segments
    # keep exactly their current one-`_seg`-per-sub-segment behavior.
    # [PAN-JUMP-CAP-2026-08-18] Max center-position delta (fraction of frame)
    # a merged split run is allowed to glide smoothly between two adjacent
    # sub-segments before it's forced to cut instead. Without this, a locked
    # identity's face position drifting a long way over several sub-segments
    # (head turns, imprecise landmarks, or a genuinely different framing) gets
    # faithfully animated by piecewiseLerp as one long slow pan — reported as
    # "kegeser-geser kasar" on a real clip: panel A's x went 0.57→0.16 (41% of
    # frame width) over ~6s within a single locked run. Before [SPLIT-SMOOTH]
    # this drift was invisible (hidden inside per-~1.2s hard-cut statics);
    # smoothing it made the drift itself visible as an ugly pan. A real
    # continuous shot rarely needs a >15% reframe in one ~1.2s sub-segment
    # step, so treat a bigger jump as a scene/framing change: cut, don't pan.
    PAN_JUMP_MAX = _cfg_f("TRK_SPLIT_PAN_MAX_JUMP", 0.15)

    def _rect_center(r):
        return (r["x"] + r["w"] / 2.0, r["y"] + r["h"] / 2.0)

    def _panel_jumped(anchor, cur):
        if anchor is None or cur is None:
            return False
        ax, ay = _rect_center(anchor)
        cx, cy = _rect_center(cur)
        # bool(...) — some upstream rect values trace back through numpy ops
        # (np.clip/np.median), so this comparison can yield numpy.bool_,
        # which json.dumps() rejects ("Object of type bool_ is not JSON
        # serializable"). This keyframe field must be a plain Python bool.
        return bool(abs(cx - ax) > PAN_JUMP_MAX or abs(cy - ay) > PAN_JUMP_MAX)

    # [PAN-JUMP-CAP] Compared against the RUN'S ANCHOR (its first keyframe),
    # not merely the previous one — a slow steady drift (each ~1.2s step
    # under the per-step threshold) still accumulates past PAN_JUMP_MAX over
    # several steps, e.g. 0.049+0.106+0.133+0.093=0.38 total with no single
    # step over 0.133. Comparing step-to-step never caught this; comparing to
    # the anchor does, and the anchor resets to whichever keyframe starts the
    # next mini-run once a cut fires — so drift is measured fresh each time.
    split_anchor_a = None
    split_anchor_b = None

    def _assign_seg(mode, cut_val, orientation=None, panel_a=None, panel_b=None):
        nonlocal seg_ix, split_anchor_a, split_anchor_b
        # [SPLIT-SMOOTH-2026-08-18] Split sub-segments in the SAME continuous
        # run now merge into one _seg, just like single mode already did — so
        # planToSegments hands panelChain (track-plan.js) MULTIPLE keyframes and
        # its piecewiseLerp x/y path glides the panel between them instead of
        # hard-cutting at every ~1.2s sub-segment boundary (the "patah-patah"
        # complaint). This is only SAFE now because [PANEL-IDENTITY-LOCK] above
        # guarantees "top"/"bottom" refers to the SAME face across every
        # sub-segment of a run — before that, merging would have interpolated
        # between two DIFFERENT people. A real cut (cut_val), an orientation
        # change, or too large a position jump ([PAN-JUMP-CAP] above) still
        # forces a fresh _seg. panelChain takes the MEDIAN crop size across
        # the merged keyframes, so size stays stable while position glides —
        # no size pumping.
        if last_kf is None:
            can_merge = False
        elif cut_val:
            can_merge = False
        elif mode != last_kf.get("mode"):
            can_merge = False
        elif mode == "single":
            can_merge = True
        elif mode == "split":
            # [PAN-JUMP-CAP] The jump check itself now runs at the CALL SITE
            # (folded into `cut_val` as `run_start or jumped`) — track-plan.js
            # `canContinueSegment` merges split continuations by
            # mode+orientation alone and does NOT gate on `_seg` (by design,
            # see its own comment + the legacy tests that pin that behavior).
            # So the only lever that actually stops JS from smoothly panning
            # across a detected jump is the keyframe's `cut` field, which is
            # exactly what `cut_val` here already is.
            can_merge = (orientation is not None
                         and last_kf.get("orientation") == orientation)
        else:
            can_merge = False
        if not can_merge:
            seg_ix += 1
            if mode == "split":
                # This keyframe starts a fresh mini-run — it becomes the new
                # anchor every later keyframe in this _seg gets compared to.
                split_anchor_a, split_anchor_b = panel_a, panel_b
        return seg_ix
    # [NO-FACE FALLBACK] A no-detection sub-segment (nobody clears the
    # TRK_SEG_DET_FLOOR presence bar — e.g. a hand gesture briefly covers the
    # face, motion blur, or someone glances away for a beat) used to snap to
    # a hardcoded wide/full-frame view {x:0.5,y:0.5,scale:0.9}. Visually that
    # reads as a sudden zoom-OUT to the whole set between two tight face
    # crops — confirmed live via a contact sheet of a real rendered clip
    # (frames looked correct, then one frame every ~15-20s was the full
    # scene, tiny people, then correct again). A 1.2s sub-segment with no
    # detections is very likely a momentary tracking blip, not a real framing
    # change, so holding the LAST good target is far less jarring than
    # cutting to a totally different (and usually worse) composition. Only
    # the very first segment — no prior target exists yet — uses the neutral
    # centred fallback.
    last_kf = None
    # [PANEL-IDENTITY-LOCK] Persists {face_id: 'top'/'bottom' (or 'left'/
    # 'right')} for the SPLIT run currently being emitted; reset whenever
    # `run_start` begins a new run. Split runs are chopped into ~1.2s
    # sub-segments (TRK_MAX_SUBSEG_SPLIT) that each render as an independent
    # STATIC crop — no interpolation between them — so if which-face-goes-
    # top gets re-decided every sub-segment, a run reads as a sequence of
    # hard jump-cuts instead of two people holding their sides.
    #
    # The tie-break below (near-equal y → order by x; otherwise order by y)
    # picks a stable axis in isolation, but the two rules DISAGREE when a
    # pair sits near the 0.05*frame_h boundary on ONE axis while separated
    # on the other (real production case: two hosts side-by-side, y-gap
    # oscillating 65-86px on either side of the 72px/1440 threshold from
    # frame-to-frame landmark noise). Whichever rule fires flips who is
    # "top" between adjacent sub-segments even though neither person moved,
    # and the interpolation/hard-cut between the resulting rects sweeps
    # across the frame's dead centre — showing curtain/furniture instead of
    # either face. Confirmed 2026-08-18 on a real two-shot source: emitted
    # "top" alternated between the LEFT face (x-tiebreak fired) and the
    # RIGHT face (y-tiebreak fired) every ~1.2s.
    #
    # Fix: decide the mapping ONCE per run (first sub-segment) and hold it
    # for every later sub-segment in the same run, regardless of which axis
    # the geometry would pick that frame. A genuinely new pair (candidate
    # ids not in the locked map — e.g. someone new enters mid-clip) falls
    # through to the same tie-break and relocks, same as the original logic.
    split_panel_id = {}
    single_locked_id = None
    for (si, ei, rcls, run_start) in merged:
        if run_start:
            split_panel_id = {}   # new run — forget any previous panel lock
            split_anchor_a = split_anchor_b = None  # and any pan-drift anchor
            single_locked_id = None  # and any single-mode candidate lock
        t0 = round(times[si], 3)
        # [SPLIT-VISIBLE] A split panel must hold someone seen through most of
        # its slice, not a face glimpsed in 15% of it before a camera cut.
        cand = _seg_bboxes(si, ei, _cfg_f("TRK_SPLIT_DET_FLOOR", 0.5)
                           if rcls[0] == "split" else None)
        if not cand and run_start and last_kf is not None:
            # [NEW-SHOT-NO-CAND-2026-10-15] A new shot whose faces have not
            # yet been seen enough to join the presence pool (a fresh camera
            # angle) used to HOLD the previous shot's crop across the cut —
            # measured 0.8-1.0 s of an empty frame on two sources. Frame the
            # biggest face actually visible in this slice instead.
            vis = {}
            for i in range(si, ei + 1):
                for f in samples[i]["faces"]:
                    vis.setdefault(f["id"], []).append(f["bbox"])
            if vis:
                bb = max((tuple(np.median(np.asarray(v, dtype=float), axis=0)) for v in vis.values()),
                         key=lambda b: b[3] - b[1])
                keyframes.append({"t": t0, "mode": "single", "_seg": _assign_seg("single", True), "cut": True,
                                  "target": _single_target(bb, frame_w, frame_h, face_in_panel, ref_w, ref_h,
                                                           others=_others_in(si, ei, bb))})
                last_kf = keyframes[-1]
                continue
        if not cand:
            if last_kf is not None:
                held = dict(last_kf)
                held["t"] = t0
                held["cut"] = False   # continuation, not a real cut
                # [PAN-JUMP-CAP-FIX-2026-08-18] Must pass orientation/panel_a/
                # panel_b too — omitting them (as this call used to) makes
                # _assign_seg's split branch see orientation=None, which
                # ALWAYS fails `orientation is not None` and forces
                # can_merge=False regardless of cut_val. That incorrectly
                # bumps `_seg` on a frame that intentionally holds the exact
                # same position as last_kf, and WORSE: `_assign_seg` then
                # resets split_anchor_a/b to (None, None) (panel_a/panel_b
                # default), since it defaults to updating them from this
                # call's own args — an anchor of None makes every later
                # `_panel_jumped()` check in this run return False
                # unconditionally (its own None-guard), silently disabling
                # pan-jump detection for the REST of the run. Passing the
                # held frame's own orientation/panels keeps `_assign_seg`'s
                # bookkeeping consistent AND preserves the anchor — correct,
                # since a held frame repeats the last real position, so it
                # can never register as a jump anyway.
                held["_seg"] = _assign_seg(held["mode"], False,
                                            held.get("orientation"),
                                            held.get("top") or held.get("left"),
                                            held.get("bottom") or held.get("right"))
                keyframes.append(held)
            else:
                keyframes.append({"t": t0, "mode": "single", "_seg": _assign_seg("single", run_start),
                                  "cut": run_start,
                                  "target": {"x": 0.5, "y": 0.5, "scale": 0.9}})
            last_kf = keyframes[-1]
            continue

        if rcls[0] == "single" or len(cand) < 2:
            # Single: prefer the ACTIVE speaker; fall back to the most-seen face
            # so we never end up centred on the empty middle of a 2-shot.
            #
            # [SINGLE-LOCK] `cand` is re-derived per SUB-SEGMENT (TRK_MAX_SUBSEG,
            # ~1.2s slices), and with 2 comparably-active people still short of
            # the split threshold, `best`'s id can flip between them at every
            # slice boundary — same class of bug as split's `split_panel_id`
            # lock above, just one level up (which PERSON, not which panel).
            # Visually: the single-mode crop swings across most of the frame
            # every ~1.2s ("cari-cari"), confirmed on a real clip where a
            # 2-shot ran single for several seconds before enough ASD evidence
            # accumulated to call it split. Fix: pick `best` ONCE per run (its
            # first sub-segment) and keep tracking that SAME id — still using
            # each sub-segment's own median bbox for that id, so genuine drift
            # (the person moving) is followed — for every later sub-segment in
            # the run. A run boundary (real cut) clears the lock via
            # single_locked_id above, same as split_panel_id.
            by_id = {c[0]: c for c in cand}
            if single_locked_id is not None and single_locked_id in by_id:
                best = by_id[single_locked_id]
            else:
                best = max(cand, key=lambda x: (x[2], x[3]))
                single_locked_id = best[0]
            keyframes.append({"t": t0, "mode": "single", "_seg": _assign_seg("single", run_start), "cut": run_start,
                              "target": _single_target(best[1], frame_w,
                                                        frame_h, face_in_panel,
                                                        ref_w, ref_h,
                                                        others=_others_in(si, ei, best[1]))})
            last_kf = keyframes[-1]
            continue

        # SPLIT — pick the two best candidates by (activity, then presence).
        cand.sort(key=lambda x: (-x[2], -x[3]))
        id_a, a = cand[0][0], cand[0][1]
        id_b, b = cand[1][0], cand[1][1]

        # [POST-DEDUP-GUARD] Safety net: if both top candidates survived
        # _seg_bboxes dedup but are still very likely the same physical
        # person (size ratio ≈ 1.0 AND center distance tight), collapse
        # to single before doing any pair-cap work. This catches grey-zone
        # re-entry pairs that slipped the per-ID _too_close check.
        def _is_same_person_post_dedup(ba, bb):
            ha = max(1.0, ba[3] - ba[1])
            hb = max(1.0, bb[3] - bb[1])
            size_ratio = max(ha, hb) / min(ha, hb)
            cax2, cay2 = (ba[0] + ba[2]) / 2.0, (ba[1] + ba[3]) / 2.0
            cbx2, cby2 = (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
            dx2 = (cax2 - cbx2) / frame_w
            dy2 = (cay2 - cby2) / frame_h
            cdist2 = (dx2 * dx2 + dy2 * dy2) ** 0.5
            return (size_ratio < _cfg_f("TRK_POSTDEDUP_SIZE_RATIO", 1.25)
                    and cdist2 < _cfg_f("TRK_POSTDEDUP_CDIST", 0.22))

        if _is_same_person_post_dedup(a, b):
            best = max(cand[:2], key=lambda x: (x[2], x[3]))
            keyframes.append({
                "t": t0, "mode": "single",
                "_seg": _assign_seg("single", run_start),
                "cut": run_start,
                "target": _single_target(best[1], frame_w, frame_h,
                                          face_in_panel, ref_w, ref_h,
                                          others=_others_in(si, ei, best[1])),
            })
            last_kf = keyframes[-1]
            continue

        if landscape:
            panel_w, panel_h = ref_w / 2.0, ref_h
        else:
            panel_w, panel_h = ref_w, ref_h / 2.0
        # Pass every OTHER visible face (split partner + any bystanders), so
        # each panel stays clear of everyone else on screen — not just its
        # partner. `cand` holds every candidate that cleared the presence/
        # speaker floor, and `a`/`b` are only the top-2 by activity; in a
        # dense multi-person shot (5+ people seated together) `cand[2:]` is
        # exactly the bystanders whose faces could otherwise bleed into the
        # edge of a panel clamped against only its partner (2026-08-18 fix).
        bystanders = [c[1] for c in cand[2:]]
        rect_a = _panel_target(a, frame_w, frame_h, panel_w, panel_h,
                               face_in_panel, other_bbox=[b] + bystanders)
        rect_b = _panel_target(b, frame_w, frame_h, panel_w, panel_h,
                               face_in_panel, other_bbox=[a] + bystanders)

        # Panel-rect dedup: if the two candidate panels end up looking at almost
        # the same slice of source (rect IoU > 0.35), we'd draw the same face
        # twice on screen. Collapse to single on the more active candidate.
        def _rect_iou(r1, r2):
            ax1, ay1 = r1["x"], r1["y"]
            ax2, ay2 = ax1 + r1["w"], ay1 + r1["h"]
            bx1, by1 = r2["x"], r2["y"]
            bx2, by2 = bx1 + r2["w"], by1 + r2["h"]
            iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            ih = max(0.0, min(ay2, by2) - max(ay1, by1))
            inter = iw * ih
            ua = r1["w"] * r1["h"] + r2["w"] * r2["h"] - inter
            return inter / (ua + 1e-6)

        # Degenerate-panel guard: when the two candidate faces are momentarily
        # very close together (near-duplicate ByteTrack ids that dedup didn't
        # catch, or a transient close-talk moment), _panel_target's other_bbox
        # pair-cap (line ~130/140 above) can clamp a panel's crop to a near-zero
        # size instead of a sane framing — the two RECTS themselves don't
        # overlap (so the IoU check below doesn't fire) but each is too tiny to
        # be a usable crop. Confirmed on a real production clip (source
        # ~149.478s in a 2-speaker clip): one 1.2s sub-segment emitted panels
        # w=0.124 and w=0.042 (vs the steady-state w=0.55 on every neighboring
        # sub-segment) positioned off in a corner. Because that sub-segment
        # wasn't a scene cut, smooth.py's Gaussian kernel then blended this
        # single garbage keyframe into its clean neighbors, visibly shrinking
        # and overlapping panels across a ~4-second span — reading as a
        # spurious 4-quadrant grid even though grid.py never emits mode='grid'.
        # Falling back to the held last-good keyframe (same pattern as the
        # no-candidate branch above) keeps the run visually continuous instead
        # of injecting a single-frame outlier for smoothing to spread.
        min_panel_frac = _cfg_f("TRK_MIN_PANEL_FRAC", 0.20)
        degenerate = (rect_a["w"] < min_panel_frac or rect_a["h"] < min_panel_frac
                      or rect_b["w"] < min_panel_frac or rect_b["h"] < min_panel_frac)
        # [DEGENERATE-ACROSS-CUT-2026-10-15] Holding last_kf is only right
        # WITHIN a shot. At a camera cut last_kf is the previous shot's crop:
        # a multi-cam podcast going close-up → wide (small faces + poster
        # "faces" as bystanders squeeze both panels to ~0.17 W) held the
        # close-up's x≈0.46 for the whole wide shot = an empty desk. A new
        # run that cannot split frames its most active face instead.
        if degenerate and run_start and last_kf is not None:
            best = cand[0]
            keyframes.append({"t": t0, "mode": "single", "_seg": _assign_seg("single", run_start),
                              "cut": run_start,
                              "target": _single_target(best[1], frame_w, frame_h,
                                                        face_in_panel, ref_w, ref_h,
                                                        others=_others_in(si, ei, best[1]))})
            single_locked_id = best[0]
            last_kf = keyframes[-1]
            continue
        if degenerate and last_kf is not None:
            held = dict(last_kf)
            held["t"] = t0
            held["cut"] = False   # continuation, not a real cut
            # [PAN-JUMP-CAP-FIX-2026-08-18] Same fix as the no-candidates
            # fallback above — must pass orientation/panel_a/panel_b so
            # _assign_seg doesn't see orientation=None (which forces a
            # spurious _seg bump AND resets the pan-jump anchor to None,
            # silently disabling jump detection for the rest of this run).
            held["_seg"] = _assign_seg(held["mode"], False,
                                        held.get("orientation"),
                                        held.get("top") or held.get("left"),
                                        held.get("bottom") or held.get("right"))
            keyframes.append(held)
            last_kf = keyframes[-1]
            continue

        # [POST-CAP-COLLAPSE] pair-cap deliberately pushes rect_a and
        # rect_b toward opposite sides — so _rect_iou is near zero even
        # when both rects frame the SAME physical face. Check the face
        # centres in source pixels instead: if they are still close the
        # two candidates are the same person and we must render single.
        def _face_center_cdist(ba, bb):
            cax3, cay3 = (ba[0] + ba[2]) / 2.0, (ba[1] + ba[3]) / 2.0
            cbx3, cby3 = (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
            dx3 = (cax3 - cbx3) / frame_w
            dy3 = (cay3 - cby3) / frame_h
            return (dx3 * dx3 + dy3 * dy3) ** 0.5

        post_cap_cdist = _face_center_cdist(a, b)
        # Add size-ratio guard: post-cap collapse only fires when the two faces
        # are also similar in size (same person at same depth). Two people at
        # different depths (size_ratio > TRK_POSTDEDUP_SIZE_RATIO=1.25) should
        # NOT be collapsed here even if their face centres are within 0.20 —
        # e.g. a close-up subject and a background person in the same quarter.
        _post_cap_ha = max(1.0, a[3] - a[1])
        _post_cap_hb = max(1.0, b[3] - b[1])
        _post_cap_size_ratio = max(_post_cap_ha, _post_cap_hb) / min(_post_cap_ha, _post_cap_hb)
        if (post_cap_cdist < _cfg_f("TRK_POSTCAP_COLLAPSE_CDIST", 0.20)
                and _post_cap_size_ratio < _cfg_f("TRK_POSTDEDUP_SIZE_RATIO", 1.25)):
            best_bbox = a if cand[0][2] >= cand[1][2] else b
            keyframes.append({
                "t": t0, "mode": "single",
                "_seg": _assign_seg("single", run_start),
                "cut": run_start,
                "target": _single_target(best_bbox, frame_w, frame_h,
                                          face_in_panel, ref_w, ref_h,
                                          others=_others_in(si, ei, best_bbox)),
            })
            last_kf = keyframes[-1]
            continue

        # Panels are allowed to overlap on source (each still frames a different
        # face at its own centre after cropping). Only near-identical rects mean
        # true duplicate — bbox-level dedup upstream is the primary guard.
        if _rect_iou(rect_a, rect_b) > _cfg_f("TRK_PANEL_IOU", 0.75):
            keyframes.append({"t": t0, "mode": "single", "_seg": _assign_seg("single", run_start), "cut": run_start,
                              "target": _single_target(a, frame_w, frame_h,
                                                        face_in_panel, ref_w, ref_h,
                                                        others=_others_in(si, ei, a))})
            last_kf = keyframes[-1]
            continue

        if landscape:
            locked = id_a in split_panel_id and id_b in split_panel_id
            if locked:
                # Honour the run's existing lock instead of re-deciding —
                # keeps "left"/"right" identity stable across sub-segments.
                if split_panel_id[id_a] == "right":
                    id_a, id_b, rect_a, rect_b = id_b, id_a, rect_b, rect_a
            else:
                # order left→right
                if _center(a)[0] > _center(b)[0]:
                    id_a, id_b, rect_a, rect_b = id_b, id_a, rect_b, rect_a
                split_panel_id = {id_a: "left", id_b: "right"}
            # [PAN-JUMP-CAP] `cut` (not `_seg`) is what track-plan.js actually
            # honours as a hard boundary for split — see [PAN-JUMP-CAP] note
            # in _assign_seg above.
            jumped = _panel_jumped(split_anchor_a, rect_a) or _panel_jumped(split_anchor_b, rect_b)
            cut_now = run_start or jumped
            keyframes.append({
                "t": t0, "mode": "split", "_seg": _assign_seg("split", cut_now, "hstack", rect_a, rect_b), "cut": cut_now,
                "orientation": "hstack",
                "left":  rect_a,
                "right": rect_b,
            })
            last_kf = keyframes[-1]
        else:
            locked = id_a in split_panel_id and id_b in split_panel_id
            if locked:
                # Honour the run's existing lock — see [PANEL-IDENTITY-LOCK]
                # above for why re-deciding every sub-segment flip-flops.
                if split_panel_id[id_a] == "bottom":
                    id_a, id_b, rect_a, rect_b = id_b, id_a, rect_b, rect_a
            else:
                # order top→bottom by face-centre y; for hosts sitting side-by-side
                # (near-identical y) put left face on top so it reads left→right.
                ca, cb = _center(a), _center(b)
                if abs(ca[1] - cb[1]) > 0.05 * frame_h:
                    if ca[1] > cb[1]:
                        id_a, id_b, rect_a, rect_b = id_b, id_a, rect_b, rect_a
                else:
                    if ca[0] > cb[0]:
                        id_a, id_b, rect_a, rect_b = id_b, id_a, rect_b, rect_a
                split_panel_id = {id_a: "top", id_b: "bottom"}
            # [PAN-JUMP-CAP] `cut` (not `_seg`) is what track-plan.js actually
            # honours as a hard boundary for split — see [PAN-JUMP-CAP] note
            # in _assign_seg above.
            jumped = _panel_jumped(split_anchor_a, rect_a) or _panel_jumped(split_anchor_b, rect_b)
            cut_now = run_start or jumped
            keyframes.append({
                "t": t0, "mode": "split", "_seg": _assign_seg("split", cut_now, "vstack", rect_a, rect_b), "cut": cut_now,
                "orientation": "vstack",
                "top":    rect_a,
                "bottom": rect_b,
            })
            last_kf = keyframes[-1]

    if not keyframes:
        keyframes = [{"t": 0.0, "mode": "single",
                      "target": {"x": 0.5, "y": 0.5, "scale": 0.9}}]
    keyframes[0]["t"] = 0.0
    for i in range(1, len(keyframes)):
        if keyframes[i]["t"] <= keyframes[i - 1]["t"]:
            keyframes[i]["t"] = round(keyframes[i - 1]["t"] + 0.001, 3)
    return keyframes
