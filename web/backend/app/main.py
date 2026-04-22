"""
YOLOPv2 Lane Detection — FastAPI backend

Endpoints
─────────
POST /api/detect            Upload an image, run inference, get JSON result
GET  /api/results           List all past inference runs
GET  /api/results/{uid}     Get metadata for a single run
GET  /files/results/...     Serve result + viz images (static)
GET  /files/uploads/...     Serve uploaded images (static)
GET  /health                Health check
"""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import RESULTS_DIR, UPLOAD_DIR, MODEL_PATH
from app.inference import InferenceResult, YOLOPv2

# ---------------------------------------------------------------------------
# App lifecycle — load model once at startup
# ---------------------------------------------------------------------------

_model: YOLOPv2 | None = None


def _write_image(path: Path, image: np.ndarray) -> None:
    """Save an image file using Python I/O so Unicode paths work on Windows."""
    suffix = path.suffix.lower()
    encode_ext = suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".png"
    success, buffer = cv2.imencode(encode_ext, image)
    if not success:
        raise RuntimeError(f"Failed to encode image for {path}")
    path.write_bytes(buffer.tobytes())


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = YOLOPv2()
    yield


app = FastAPI(
    title="YOLOPv2 Lane Detection API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve saved files directly
app.mount("/files/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
app.mount("/files/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
    }


@app.post("/api/detect")
async def detect(
    request: Request,
    filename: str = Query("upload.jpg", description="Original filename (used for extension)"),
    lanes_only: bool = Query(True, description="Show only lane lines, hide drivable area & vehicle boxes"),
    visualize: bool = Query(False, description="Save every intermediate pipeline step to viz/"),
    conf_thres: float = Query(0.3, ge=0.0, le=1.0, description="Object confidence threshold"),
    iou_thres: float = Query(0.45, ge=0.0, le=1.0, description="NMS IoU threshold"),
):
    """
    Send raw image binary as the request body (Content-Type: application/octet-stream).

    Example:
        curl -X POST "http://localhost:8000/api/detect?filename=test.jpg" \\
             -H "Content-Type: application/octet-stream" \\
             --data-binary @test.jpg
    """
    # ── Read raw body ─────────────────────────────────────────────────────
    content = await request.body()
    if not content:
        raise HTTPException(status_code=400, detail="Request body is empty.")

    # ── Save upload to disk ───────────────────────────────────────────────
    uid = uuid.uuid4().hex[:10]
    suffix = Path(filename).suffix or ".jpg"
    upload_path = UPLOAD_DIR / f"{uid}{suffix}"
    upload_path.write_bytes(content)

    # ── Decode ────────────────────────────────────────────────────────────
    arr = np.frombuffer(content, np.uint8)
    orig = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if orig is None:
        raise HTTPException(status_code=400, detail="Cannot decode image data.")

    # ── Inference ─────────────────────────────────────────────────────────
    result: InferenceResult = _model.run(
        orig,
        lanes_only=lanes_only,
        visualize=visualize,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
    )

    # ── Save result image ─────────────────────────────────────────────────
    run_dir = RESULTS_DIR / uid
    run_dir.mkdir(parents=True, exist_ok=True)
    result_filename = f"result{suffix}"
    _write_image(run_dir / result_filename, result.result_image)

    # Store metadata for /api/results/{uid}
    _save_meta(run_dir, uid, filename, suffix, result, lanes_only, visualize,
               conf_thres, iou_thres)

    # ── Save viz images ───────────────────────────────────────────────────
    viz_urls: dict[str, str] | None = None
    if visualize and result.viz_images:
        viz_dir = run_dir / "viz"
        viz_dir.mkdir(exist_ok=True)
        viz_urls = {}
        for step_name, img in result.viz_images.items():
            ext = ".png" if "mask" in step_name else ".jpg"
            fname = f"{step_name}{ext}"
            _write_image(viz_dir / fname, img)
            viz_urls[step_name] = f"/files/results/{uid}/viz/{fname}"

    return {
        "uid": uid,
        "upload_url": f"/files/uploads/{uid}{suffix}",
        "result_url": f"/files/results/{uid}/{result_filename}",
        "original_size": {"w": result.original_size[0], "h": result.original_size[1]},
        "inf_time_ms": round(result.inf_time_ms, 2),
        "nms_time_ms": round(result.nms_time_ms, 2),
        "detections": [
            {"bbox": d.bbox, "conf": round(d.conf, 4), "cls": d.cls}
            for d in result.detections
        ],
        "viz_urls": viz_urls,
    }


@app.get("/api/results")
def list_results():
    """List all inference runs (newest first)."""
    runs = []
    for d in sorted(RESULTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir():
            meta_path = d / "meta.json"
            if meta_path.exists():
                import json
                runs.append(json.loads(meta_path.read_text()))
            else:
                runs.append({"uid": d.name})
    return runs


@app.get("/api/results/{uid}")
def get_result(uid: str):
    """Get metadata for a single run."""
    import json
    meta_path = RESULTS_DIR / uid / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Run not found.")
    return json.loads(meta_path.read_text())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_meta(run_dir: Path, uid: str, filename: str | None,
               suffix: str, result: InferenceResult,
               lanes_only: bool, visualize: bool,
               conf_thres: float, iou_thres: float) -> None:
    import json
    meta = {
        "uid": uid,
        "filename": filename,
        "upload_url": f"/files/uploads/{uid}{suffix}",
        "result_url": f"/files/results/{uid}/result{suffix}",
        "original_size": {"w": result.original_size[0], "h": result.original_size[1]},
        "inf_time_ms": round(result.inf_time_ms, 2),
        "nms_time_ms": round(result.nms_time_ms, 2),
        "options": {
            "lanes_only": lanes_only,
            "visualize": visualize,
            "conf_thres": conf_thres,
            "iou_thres": iou_thres,
        },
        "detections": [
            {"bbox": d.bbox, "conf": round(d.conf, 4), "cls": d.cls}
            for d in result.detections
        ],
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
