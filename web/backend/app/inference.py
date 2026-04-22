"""
YOLOPv2 inference engine.

Extracted from root main.py and wrapped as a reusable class so the model
is loaded once at server startup and reused across requests.
"""

import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from app.config import MODEL_PATH, SOURCE_PATH, IMG_SIZE, WORK_SIZE, STRIDE, DEVICE

# Make source/utils importable
sys.path.insert(0, str(SOURCE_PATH))

from utils.utils import (           # noqa: E402
    select_device,
    time_synchronized,
    scale_coords,
    xyxy2xywh,
    non_max_suppression,
    split_for_trace_model,
    driving_area_mask,
    lane_line_mask,
    plot_one_box,
    show_seg_result,
    letterbox,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    bbox: list[float]   # [x1, y1, x2, y2] in original image coordinates
    conf: float
    cls: int


@dataclass
class InferenceResult:
    original_size: tuple[int, int]           # (w, h) of input image
    detections: list[Detection]
    inf_time_ms: float                       # pure model forward pass
    nms_time_ms: float
    result_image: np.ndarray                 # BGR, original resolution
    viz_images: dict[str, np.ndarray] = field(default_factory=dict)  # step -> BGR


# ---------------------------------------------------------------------------
# Helpers (same as root main.py)
# ---------------------------------------------------------------------------

def _logits_to_heatmap(tensor, size_wh: tuple[int, int]) -> np.ndarray:
    """Render a 2-D tensor as a JET-colormap heatmap."""
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    mn, mx = float(arr.min()), float(arr.max())
    arr = (arr - mn) / (mx - mn + 1e-9)
    arr = (arr * 255).astype(np.uint8)
    arr = cv2.resize(arr, size_wh, interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(arr, cv2.COLORMAP_JET)


def _mask_to_image(mask: np.ndarray) -> np.ndarray:
    """Binary mask -> white-on-black BGR image."""
    m = (mask > 0).astype(np.uint8) * 255
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class YOLOPv2:
    def __init__(self, device: str = DEVICE):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Download from https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
            )
        self.device = select_device(device)
        self.half = self.device.type != "cpu"
        print(f"[YOLOPv2] Loading model from {MODEL_PATH} on {self.device} …")
        self.model = torch.jit.load(str(MODEL_PATH), map_location=self.device)
        self.model = self.model.to(self.device)
        if self.half:
            self.model.half()
        self.model.eval()
        print("[YOLOPv2] Model ready.")

    # ------------------------------------------------------------------

    def run(
        self,
        orig: np.ndarray,
        lanes_only: bool = False,
        visualize: bool = False,
        conf_thres: float = 0.3,
        iou_thres: float = 0.45,
        classes: list[int] | None = None,
    ) -> InferenceResult:
        with torch.no_grad():
            return self._run(orig, lanes_only, visualize, conf_thres, iou_thres, classes)

    # ------------------------------------------------------------------

    def _run(self, orig, lanes_only, visualize, conf_thres, iou_thres, classes):
        viz: dict[str, np.ndarray] = {}

        # ── Step 00 ─────────────────────────────────────────────────────
        # Original image as-is from disk
        if visualize:
            viz["00_original"] = orig.copy()

        # ── Step 01 ─────────────────────────────────────────────────────
        # Resize to work canvas 1280×720 so that driving_area_mask /
        # lane_line_mask internal crop+upsample yields a 720×1280 mask.
        im0w = cv2.resize(orig, WORK_SIZE, interpolation=cv2.INTER_LINEAR)
        if visualize:
            viz["01_work_canvas"] = im0w.copy()

        # ── Step 02 ─────────────────────────────────────────────────────
        # Letterbox 1280×720 → 640×384 (stride-multiple, model input shape).
        lb_rgb, _, _ = letterbox(im0w, IMG_SIZE, stride=STRIDE)
        if visualize:
            viz["02_letterboxed"] = lb_rgb[:, :, ::-1].copy()   # store as BGR

        # ── Step 03 ─────────────────────────────────────────────────────
        # BGR→RGB, HWC→CHW, /255 → float tensor (1, 3, 384, 640)
        img_np = lb_rgb[:, :, ::-1].transpose(2, 0, 1)
        img_np = np.ascontiguousarray(img_np)
        img = torch.from_numpy(img_np).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        if visualize:
            t_view = (img[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
            viz["03_tensor_ch0"] = cv2.cvtColor(t_view, cv2.COLOR_GRAY2BGR)

        # ── Step 04-05 ──────────────────────────────────────────────────
        # Model forward: detection head + drivable-area seg + lane-line seg
        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = self.model(img)
        t2 = time_synchronized()
        inf_ms = (t2 - t1) * 1000

        if visualize:
            ww, wh = WORK_SIZE
            viz["04_seg_raw"] = _logits_to_heatmap(seg[0, 1], (ww, wh))
            ll_ch = ll[0, 0] if ll.shape[1] == 1 else ll[0, 1]
            viz["05_ll_raw"] = _logits_to_heatmap(ll_ch, (ww, wh))

        # ── Step NMS ────────────────────────────────────────────────────
        pred = split_for_trace_model(pred, anchor_grid)
        t3 = time_synchronized()
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes)
        t4 = time_synchronized()
        nms_ms = (t4 - t3) * 1000

        # ── Step 06-07 ──────────────────────────────────────────────────
        # Post-process segmentation: crop head/foot padding, upsample ×2,
        # argmax/round → binary masks at 720×1280 (work-canvas resolution)
        da_mask_raw = driving_area_mask(seg)
        ll_mask_raw = lane_line_mask(ll)
        if visualize:
            viz["06_da_mask"] = _mask_to_image(da_mask_raw)
            viz["07_ll_mask"] = _mask_to_image(ll_mask_raw)

        # ── Step 08-09 ──────────────────────────────────────────────────
        # Upscale masks back to original image resolution (INTER_NEAREST
        # prevents blurring the sharp mask edges)
        h0, w0 = orig.shape[:2]
        da_mask = cv2.resize(da_mask_raw.astype(np.uint8), (w0, h0),
                             interpolation=cv2.INTER_NEAREST)
        ll_mask = cv2.resize(ll_mask_raw.astype(np.uint8), (w0, h0),
                             interpolation=cv2.INTER_NEAREST)
        if visualize:
            viz["08_da_mask_orig"] = _mask_to_image(da_mask)
            viz["09_ll_mask_orig"] = _mask_to_image(ll_mask)

        # ── Step 10 ─────────────────────────────────────────────────────
        # Scale detection bboxes from model-input space (640×384) →
        # original image space, then decode per detection
        det = pred[0]
        detections: list[Detection] = []
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orig.shape).round()
            for *xyxy, conf, cls in det.tolist():
                detections.append(Detection(
                    bbox=[float(v) for v in xyxy],
                    conf=float(conf),
                    cls=int(cls),
                ))
        if visualize:
            det_canvas = orig.copy()
            for d in detections:
                plot_one_box(d.bbox, det_canvas, line_thickness=3)
            viz["10_detections_raw"] = det_canvas

        # ── Steps 11-14 ─────────────────────────────────────────────────
        # Compose overlays onto the original-resolution canvas
        if visualize:
            da_only = orig.copy()
            show_seg_result(da_only, (da_mask, np.zeros_like(ll_mask)), is_demo=True)
            viz["11_overlay_da"] = da_only

            ll_only = orig.copy()
            show_seg_result(ll_only, (np.zeros_like(da_mask), ll_mask), is_demo=True)
            viz["12_overlay_ll"] = ll_only

            box_only = orig.copy()
            for d in detections:
                plot_one_box(d.bbox, box_only, line_thickness=3)
            viz["13_overlay_boxes"] = box_only

        da_draw = np.zeros_like(da_mask) if lanes_only else da_mask
        result = orig.copy()
        if not lanes_only:
            for d in detections:
                plot_one_box(d.bbox, result, line_thickness=3)
        show_seg_result(result, (da_draw, ll_mask), is_demo=True)

        if visualize:
            viz["14_final"] = result.copy()

        return InferenceResult(
            original_size=(w0, h0),
            detections=detections,
            inf_time_ms=inf_ms,
            nms_time_ms=nms_ms,
            result_image=result,
            viz_images=viz,
        )
