# Polyglot Tracker

Smart face-tracking sidecar for PolyglotClipper. Face detection, landmarking,
active-speaker scoring, and crop-framing keyframes — delivered over a
persistent JSON-lines protocol (stdin/stdout).

## Stack (all OSS)

- Python 3.11 · [insightface](https://github.com/deepinsight/insightface) (SCRFD det_10g) · onnxruntime (CoreML/CPU)
- MediaPipe face landmarker · Silero VAD (MIT) · OpenCV · scikit-learn/scipy
- License: see source headers; insightface MIT, onnxruntime MIT, silero MIT, mediapipe Apache-2.0

## Dev

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
python sidecar/tracker/main.py < job.jsonl
```

Model files live in `sidecar/tracker/models/` (silero_vad.onnx,
face_landmarker.task, insightface/models/buffalo_l/det_10g.onnx).

## Protocol (v2)

One JSON job per line on stdin, JSON progress/result/error lines on stdout.
`{"type":"ready","protocol":2}` on startup. Process stays alive for many jobs
so model load is paid once.
