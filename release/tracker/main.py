# sidecar/tracker/main.py
"""Smart face-tracking sidecar. PERSISTENT process: reads one JSON job per
newline-delimited line on stdin, and for EACH job streams {"type":"progress"}
lines followed by one {"type":"result"} or {"type":"error"} line, then waits
for the next line. Keeps running until stdin closes (EOF) — this lets the
caller reuse one already-warm process (loaded models) across many clips
instead of paying model-load cost per clip. A single-job call (write one
line, close stdin) still works exactly as before: one result/error line, and
the process exits nonzero if that job errored.

The heavy model pipeline lives in a sibling `pipeline` module; this file only
owns the protocol. Detector/Landmarker are cached as module-level singletons
inside pipeline.py, so the SECOND and later jobs in a session skip model
load/compile entirely — only the first job in a process pays that cost."""
import sys, json


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def run(job):
    # `_stub` lets the protocol be tested without models or a real video.
    if job.get("config", {}).get("_stub"):
        return {"type": "result", "keyframes": [
            {"t": 0.0, "mode": "single",
             "target": {"x": 0.5, "y": 0.3, "scale": 0.85}}]}
    # Heartbeat BEFORE the deferred heavy imports (~7s of silence otherwise).
    # The app's idle watchdog treats total silence as a wedged process; every
    # long silent phase needs at least one line.
    emit({"type": "progress", "stage": "init", "pct": 0.0})
    from pipeline import analyze  # heavy imports deferred until first real job
    return analyze(job, on_progress=lambda stage, pct: emit(
        {"type": "progress", "stage": stage, "pct": pct}))


def main():
    # Protocol handshake: lets the app distinguish this persistent protocol-2
    # sidecar from the legacy single-job bundle (which reads stdin to EOF and
    # emits nothing until then). Must be sent BEFORE blocking on stdin.
    emit({"type": "ready", "protocol": 2})
    had_error = False
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except Exception as e:
            emit({"type": "error", "message": f"bad job json: {e}"})
            had_error = True
            continue
        try:
            emit(run(job))
        except Exception as e:  # one job's failure must not kill the process —
            emit({"type": "error", "message": str(e)})  # more jobs may follow
            had_error = True
    sys.exit(1 if had_error else 0)


if __name__ == "__main__":
    main()
