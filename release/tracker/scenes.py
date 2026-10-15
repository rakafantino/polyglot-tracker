# sidecar/tracker/scenes.py
"""Scene-cut detection over a clip window, used to reset framing/tracker
continuity so the crop doesn't smoothly drift across a hard cut.

[FALSE-POSITIVE FIX 2026-08-06] threshold=22 was documented as "aggressive so
multi-cam podcasts (src2) segment on every camera switch" — but re-measured,
src2 (and all 5 training fixtures) detect ZERO cuts at any threshold from 3 to
30, so that rationale no longer holds against the current fixture set (it may
have been true for a since-replaced fixture). Meanwhile on a real 60s single
continuous talking-head clip (one person, hand gestures near the face), 22
fired 6 FALSE-POSITIVE cut pairs (~every 5-7s, confirmed via direct
`scenedetect.detect()` calls matching the trackPlan's spurious position jumps
exactly). The downstream smoother intentionally never blends across a `cut`
boundary — correct for real cuts, but a false one produces a visible snap/
recover every few seconds, i.e. exactly a "not smooth throughout" complaint.
30 eliminates all 6 false positives on that clip and changes NOTHING on the 5
training fixtures (still zero cuts there either way — this codebase currently
has no fixture that exercises a genuine detected cut, so this change could
not be positively verified against a true-positive case; a synthetic
concatenated-clip test hit an unrelated windowing quirk in this wrapper's
start_time/end_time usage that suppressed a real cut regardless of threshold —
separate, not yet understood, flagged for its own investigation).

[FALSE-POSITIVE FIX #2 2026-08-06] 30 was NOT sufficient — a real user report
("pergantian ke speaker lainnya masih kasar") led to inspecting an actual
generated clip's trackPlan: 42 of 64 keyframes were `cut`-marked (65%!),
several with >20% frame-width position jumps and zero interpolation, on a 60s
window that is provably ONE continuous static studio shot (same background,
same chair, same laptop/logo on both sides of every "cut", confirmed by
extracting frames at the jump timestamps). This is a DIFFERENT source video
than the one that validated 30 (different lighting/camera setup), and
threshold clearly does not generalize across sources: re-measured directly
against this video's exact clip window, 30->26 false cuts, 40->20, 50->0.
Broad-scanned 7 more 60s windows spread across this same 44-minute source at
threshold=50: zero cuts everywhere. Raised to 60 for headroom above that
measured cliff edge rather than riding exactly on it. Same caveat as above
still applies — no fixture in this codebase exercises a genuine positive cut,
so this is tuned entirely against false-positive elimination; if a real,
intentional multi-cam switch ever needs detecting again, it needs its own
fixture before touching this number, not a guess."""
import os
from scenedetect import detect, ContentDetector


def _sec_to_timecode(sec):
    """Format seconds as HH:MM:SS.mmm — the one start_time/end_time type that
    reliably works with this pinned scenedetect==0.6.4 (see [WINDOW-BUG] below)."""
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def scene_cuts(video_path, start, end):
    """Return a list of cut times (seconds, clip-local, 0-based) inside the
    [start, end] window. Empty list when there are no cuts (or on failure)."""
    # [WINDOW-BUG-FIX 2026-08-18] 60 was tuned SOLELY against false positives
    # (see docstring above) with no way to verify true-positive recall — the
    # windowing bug above was silently suppressing every cut in every test
    # clip available at the time, including the "genuine multi-cam switch"
    # fixture the prior investigation explicitly said it was missing. With
    # windowing fixed, a real production clip (2-shot podcast, hard cut
    # closeup->wide at t=7.44s, confirmed by frame extraction) now gives a
    # real calibration point: cut detected at every threshold <=50, lost at
    # 60. 50 is exactly the value the prior false-positive investigation
    # recorded as clean on ITS clip too ("50->0" false cuts) — so 50 is the
    # one value confirmed against BOTH known constraints; 60 satisfies only
    # the (possibly cut-blind) false-positive side. Not re-verified against
    # the original false-positive clip directly (not available in this
    # session) — if a false-positive regression shows up, that clip is the
    # one to re-test against, not a blind re-raise of this number.
    thr = float(os.environ.get("TRK_SCENE_THRESHOLD", 50.0))
    min_len = int(os.environ.get("TRK_SCENE_MIN_LEN", 6))  # frames
    try:
        # [PERF] Bound PySceneDetect to the clip window. Measured on a real
        # 26m40s source analyzing a 30s clip window: unbounded detect()
        # decodes the ENTIRE source video every call — 59.3s — to answer a
        # question about 30s of it (48.6x waste, confirmed byte-identical cut
        # times before/after this change). For a 1-hour source cut into
        # 30-50 clips this re-decodes the full hour 30-50 times; it was the
        # single largest cost in the whole analysis pipeline, well above
        # model load or detection/landmark inference.
        #
        # [WINDOW-BUG-FIX 2026-08-18] This docstring above previously flagged
        # "an unrelated windowing quirk ... suppressed a real cut regardless
        # of threshold — separate, not yet understood." Root-caused: passing
        # `start_time`/`end_time` as a bare Python `float` (seconds) to this
        # pinned scenedetect==0.6.4's `detect()` silently returns ZERO scenes
        # — confirmed directly (`end_time=20.0` -> 0 scenes; `end_time=600`
        # int frames, `end_time="00:00:20.000"` string, and a FrameTimecode
        # object all correctly -> 8 scenes on the same clip). This is why NO
        # threshold tuning session ever found a real cut: the float type
        # alone suppressed detection before threshold was even evaluated —
        # every one of the 5 training fixtures reporting "zero cuts at any
        # threshold" may have been hitting this, not genuinely cut-free.
        # Fix: pass HH:MM:SS.mmm strings instead of bare floats.
        scenes = detect(str(video_path),
                        ContentDetector(threshold=thr, min_scene_len=min_len),
                        start_time=_sec_to_timecode(start),
                        end_time=_sec_to_timecode(end))
    except Exception:
        return []
    cuts = []
    for scene_start, _scene_end in scenes:
        # get_seconds() (not the newer `.seconds` property) — requirements.txt
        # pins scenedetect==0.6.4, which does NOT have `.seconds` at all
        # (AttributeError). get_seconds() exists and is numerically identical
        # on both 0.6.4 and later versions (just deprecated post-0.6.4), so
        # it's the correct call for the dependency that's actually shipped.
        t = scene_start.get_seconds()
        if start < t < end:  # a cut strictly inside the window
            cuts.append(round(t - start, 3))
    return sorted(cuts)
