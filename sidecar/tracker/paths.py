# sidecar/tracker/paths.py
import os, pathlib
def model_dir():
    return pathlib.Path(os.environ.get("TRACKER_MODEL_DIR",
                        pathlib.Path(__file__).parent / "models"))
def model_path(name):
    p = model_dir() / name
    if not p.exists():
        raise FileNotFoundError(f"model missing: {p}")
    return str(p)

def insightface_root():
    """Root the InsightFace loader reads models from: `<root>/models/<pack>/`.
    We ship ONLY det_10g.onnx (detection) under models/insightface/, so when
    that exists we point InsightFace there — avoiding its ~325MB runtime
    download of the full buffalo_l pack (5 models, of which we use 1). Returns
    None to fall back to the default (~/.insightface, dev machines) so the dev
    venv still works without the bundled model."""
    root = model_dir() / "insightface"
    det = root / "models" / "buffalo_l" / "det_10g.onnx"
    return str(root) if det.exists() else None
