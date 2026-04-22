import os
from pathlib import Path

# web/backend/app/ -> web/backend/ -> web/ -> project root
BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent.parent

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / "model" / "yolopv2.pt")))
SOURCE_PATH = PROJECT_ROOT / "source"

UPLOAD_DIR = BACKEND_ROOT / "uploads"
RESULTS_DIR = BACKEND_ROOT / "results"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Model constants — must match the traced model's fixed input shape
IMG_SIZE = 640           # letterbox target (width)
WORK_SIZE = (1280, 720)  # (w, h) work canvas; masks are post-processed to 720×1280
STRIDE = 32

DEVICE = os.getenv("DEVICE", "cpu")
