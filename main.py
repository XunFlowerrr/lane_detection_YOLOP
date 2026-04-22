import sys
import time
import glob
import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "source"))

from utils.utils import (
    time_synchronized, select_device, increment_path,
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model,
    driving_area_mask, lane_line_mask, plot_one_box, show_seg_result,
    AverageMeter, letterbox,
)

ROOT = Path(__file__).parent

IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
VID_FORMATS = ["mov", "avi", "mp4", "mpg", "mpeg", "m4v", "wmv", "mkv"]


class LoadImagesOrig:
    """Loader that preserves the original image/frame resolution.

    Yields per frame:
        path : str
        img  : np.ndarray  (3, H, W) letterboxed model input (RGB, stride-aligned)
        im0w : np.ndarray  work-canvas frame at (work_h, work_w) for mask-space
        orig : np.ndarray  original image at its native resolution (BGR)
        cap  : cv2.VideoCapture or None
    """

    def __init__(self, path, img_size=640, stride=32, work_size=(1280, 720)):
        p = str(Path(path).absolute())
        if "*" in p:
            files = sorted(glob.glob(p, recursive=True))
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, "*.*")))
        elif os.path.isfile(p):
            files = [p]
        else:
            raise Exception(f"ERROR: {p} does not exist")

        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split(".")[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.work_size = work_size  # (w, h) — must match mask-producing behaviour
        self.files = images + videos
        self.nf = ni + nv
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        if any(videos):
            self.new_video(videos[0])
        else:
            self.cap = None
        assert self.nf > 0, f"No images or videos found in {p}."

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            self.mode = "video"
            ret_val, orig = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:
                    raise StopIteration
                path = self.files[self.count]
                self.new_video(path)
                ret_val, orig = self.cap.read()
            self.frame += 1
        else:
            self.count += 1
            orig = cv2.imread(path)
            assert orig is not None, "Image Not Found " + path

        # Working canvas at the size the mask post-processing expects (1280x720).
        # This is required because driving_area_mask / lane_line_mask contain
        # hardcoded crop+upsample that produce a 720x1280 mask.
        im0w = cv2.resize(orig, self.work_size, interpolation=cv2.INTER_LINEAR)

        # Letterbox to stride-aligned model input
        img = letterbox(im0w, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
        img = np.ascontiguousarray(img)

        return path, img, im0w, orig, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(ROOT / "model" / "yolopv2.pt"))
    parser.add_argument("--source", type=str, default=str(ROOT / "test_images"))
    parser.add_argument("--img-size", type=int, default=640,
                        help="model input size. NOTE: the bundled TorchScript model is traced at 640 — changing this will fail.")
    parser.add_argument("--conf-thres", type=float, default=0.3)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nosave", action="store_true")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    parser.add_argument("--classes", nargs="+", type=int)
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--project", default=str(ROOT / "runs" / "detect"))
    parser.add_argument("--name", default="exp")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--lanes-only", action="store_true",
                        help="show only lane lines, hide drivable area overlay and vehicle boxes")
    parser.add_argument("--visualize", action="store_true",
                        help="save every intermediate step of the pipeline to a viz/ subfolder")
    return parser


ALGORITHM_STEPS = [
    ("00_original",          "ภาพต้นฉบับที่อ่านมาจาก disk (BGR)"),
    ("01_work_canvas",       "Resize เป็น work canvas 1280x720 (ให้ขนาด output mask ตรงกับ hardcode crop ใน post-process)"),
    ("02_letterboxed",       "Letterbox เป็น 640x384 (model input ที่ถูก trace ไว้) — รักษาสัดส่วนแล้ว pad สีเทา"),
    ("03_tensor_rgb",        "Convert BGR->RGB, HWC->CHW, normalize /255 -> tensor (1,3,384,640) สำหรับป้อน model"),
    ("04_seg_raw",           "Raw segmentation logits จาก head SEG ของ model (2-ch: background vs drivable) — ก่อน argmax"),
    ("05_ll_raw",            "Raw lane-line logits จาก head LL ของ model — ก่อน threshold"),
    ("06_da_mask",           "Drivable area mask: crop [12:372,:], upsample x2, argmax -> binary mask 720x1280"),
    ("07_ll_mask",           "Lane line mask: crop [12:372,:], upsample x2, round -> binary mask 720x1280"),
    ("08_da_mask_orig",      "Upscale DA mask กลับเป็นขนาดภาพต้นฉบับด้วย INTER_NEAREST"),
    ("09_ll_mask_orig",      "Upscale LL mask กลับเป็นขนาดภาพต้นฉบับด้วย INTER_NEAREST"),
    ("10_detections_raw",    "Detection head output หลังผ่าน split_for_trace_model + NMS (xyxy, conf, cls)"),
    ("11_overlay_da",        "วาดเฉพาะ drivable area (เขียว) ทับบนภาพต้นฉบับ"),
    ("12_overlay_ll",        "วาดเฉพาะ lane line (แดง) ทับบนภาพต้นฉบับ"),
    ("13_overlay_boxes",     "วาดเฉพาะ bounding box รถ (ฟ้า) ทับบนภาพต้นฉบับ"),
    ("14_final",             "รวมทุก overlay: drivable area + lane line + boxes บนภาพต้นฉบับ (ผลลัพธ์สุดท้าย)"),
]


def _logits_to_heatmap(tensor, size_hw):
    """Turn a 1-channel activation (H,W) torch tensor into a BGR heatmap at size_hw=(W,H)."""
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        arr = np.zeros_like(arr)
    else:
        arr = (arr - mn) / (mx - mn)
    arr = (arr * 255).astype(np.uint8)
    arr = cv2.resize(arr, size_hw, interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(arr, cv2.COLORMAP_JET)


def _mask_to_image(mask, size_hw=None):
    """Binary/int mask -> 3-channel white-on-black image."""
    m = mask
    if hasattr(m, "detach"):
        m = m.detach().cpu().numpy()
    m = (m > 0).astype(np.uint8) * 255
    m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    if size_hw is not None:
        m = cv2.resize(m, size_hw, interpolation=cv2.INTER_NEAREST)
    return m


def _print_algorithm_summary():
    print("\n" + "=" * 78)
    print("YOLOPv2 LANE-DETECTION PIPELINE — สรุปขั้นตอน")
    print("=" * 78)
    for name, desc in ALGORITHM_STEPS:
        print(f"  [{name}] {desc}")
    print("=" * 78 + "\n")


def detect(opt):
    source, weights, save_txt, imgsz = opt.source, opt.weights, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith(".txt")

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    stride = 32
    device = select_device(opt.device)
    half = device.type != "cpu"
    model = torch.jit.load(weights, map_location=device)
    model = model.to(device)
    if half:
        model.half()
    model.eval()

    dataset = LoadImagesOrig(source, img_size=imgsz, stride=stride)

    if device.type != "cpu":
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    if opt.visualize:
        _print_algorithm_summary()

    t0 = time.time()
    for path, img_np, im0w, orig, vid_cap in dataset:
        # Keep copies for visualisation BEFORE tensor conversion
        viz_dir = None
        if opt.visualize:
            stem = Path(path).stem + (f"_{getattr(dataset,'frame',0)}" if dataset.mode == "video" else "")
            viz_dir = save_dir / "viz" / stem
            viz_dir.mkdir(parents=True, exist_ok=True)
            # [00] original
            cv2.imwrite(str(viz_dir / "00_original.jpg"), orig)
            # [01] work canvas 1280x720
            cv2.imwrite(str(viz_dir / "01_work_canvas.jpg"), im0w)
            # [02] letterboxed RGB model input -> convert back to BGR for viewing
            lb_view = np.ascontiguousarray(img_np.transpose(1, 2, 0)[:, :, ::-1])
            cv2.imwrite(str(viz_dir / "02_letterboxed.jpg"), lb_view)

        img = torch.from_numpy(img_np).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if opt.visualize:
            # [03] normalized tensor — show first channel as grayscale for sanity
            t_view = (img[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(viz_dir / "03_tensor_ch0.jpg"),
                        cv2.cvtColor(t_view, cv2.COLOR_GRAY2BGR))

        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img)
        t2 = time_synchronized()

        if opt.visualize:
            # [04] raw drivable-area logits (class-1 channel, full 384x640)
            da_logit = seg[0, 1]  # probability-like score for "drivable"
            cv2.imwrite(str(viz_dir / "04_seg_raw.jpg"),
                        _logits_to_heatmap(da_logit, (im0w.shape[1], im0w.shape[0])))
            # [05] raw lane-line logits
            ll_logit = ll[0, 0] if ll.shape[1] == 1 else ll[0, 1]
            cv2.imwrite(str(viz_dir / "05_ll_raw.jpg"),
                        _logits_to_heatmap(ll_logit, (im0w.shape[1], im0w.shape[0])))

        tw1 = time_synchronized()
        pred = split_for_trace_model(pred, anchor_grid)
        tw2 = time_synchronized()

        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask_raw = driving_area_mask(seg)   # (work_h, work_w)
        ll_seg_mask_raw = lane_line_mask(ll)       # (work_h, work_w)

        if opt.visualize:
            # [06] DA binary mask at work-canvas res
            cv2.imwrite(str(viz_dir / "06_da_mask.png"),
                        _mask_to_image(da_seg_mask_raw))
            # [07] LL binary mask at work-canvas res
            cv2.imwrite(str(viz_dir / "07_ll_mask.png"),
                        _mask_to_image(ll_seg_mask_raw))

        da_seg_mask = da_seg_mask_raw.copy()
        ll_seg_mask = ll_seg_mask_raw.copy()
        if opt.lanes_only:
            da_seg_mask = np.zeros_like(da_seg_mask)

        # Upscale masks from work-canvas back to the ORIGINAL image size
        h0, w0 = orig.shape[:2]
        da_seg_mask = cv2.resize(da_seg_mask.astype(np.uint8), (w0, h0),
                                 interpolation=cv2.INTER_NEAREST)
        ll_seg_mask = cv2.resize(ll_seg_mask.astype(np.uint8), (w0, h0),
                                 interpolation=cv2.INTER_NEAREST)

        if opt.visualize:
            cv2.imwrite(str(viz_dir / "08_da_mask_orig.png"),
                        _mask_to_image(da_seg_mask))
            cv2.imwrite(str(viz_dir / "09_ll_mask_orig.png"),
                        _mask_to_image(ll_seg_mask))

        for i, det in enumerate(pred):
            p, s, frame = path, "", getattr(dataset, "frame", 0)
            im0 = orig.copy()  # draw final on a copy of the original-resolution canvas
            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")
            s += "%gx%g " % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path + ".txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")
                    if save_img and not opt.lanes_only:
                        plot_one_box(xyxy, im0, line_thickness=3)

            print(f"{s}-> {w0}x{h0} Done. ({t2 - t1:.3f}s)")

            if opt.visualize:
                # [10] detections visualised alone
                det_canvas = orig.copy()
                if len(det):
                    for *xyxy, conf, cls in reversed(det):
                        plot_one_box(xyxy, det_canvas, line_thickness=3)
                cv2.imwrite(str(viz_dir / "10_detections_raw.jpg"), det_canvas)

                # [11] drivable-area-only overlay on original
                da_only = orig.copy()
                show_seg_result(da_only, (da_seg_mask_raw_orig := cv2.resize(
                    da_seg_mask_raw.astype(np.uint8), (w0, h0),
                    interpolation=cv2.INTER_NEAREST),
                    np.zeros_like(ll_seg_mask)), is_demo=True)
                cv2.imwrite(str(viz_dir / "11_overlay_da.jpg"), da_only)

                # [12] lane-line-only overlay on original
                ll_only = orig.copy()
                show_seg_result(ll_only,
                                (np.zeros_like(da_seg_mask), ll_seg_mask),
                                is_demo=True)
                cv2.imwrite(str(viz_dir / "12_overlay_ll.jpg"), ll_only)

                # [13] boxes-only overlay on original
                box_only = orig.copy()
                if len(det):
                    for *xyxy, conf, cls in reversed(det):
                        plot_one_box(xyxy, box_only, line_thickness=3)
                cv2.imwrite(str(viz_dir / "13_overlay_boxes.jpg"), box_only)

            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)

            if opt.visualize:
                cv2.imwrite(str(viz_dir / "14_final.jpg"), im0)
                print(f"  viz saved: {viz_dir}")

            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                    print(f"Result saved: {save_path}")
                else:
                    if not hasattr(detect, "_vid_path") or detect._vid_path != save_path:
                        detect._vid_path = save_path
                        if hasattr(detect, "_vid_writer") and isinstance(detect._vid_writer, cv2.VideoWriter):
                            detect._vid_writer.release()
                        fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                        w, h = im0.shape[1], im0.shape[0]
                        if not vid_cap:
                            save_path += ".mp4"
                        detect._vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                    detect._vid_writer.write(im0)

        inf_time.update(t2 - t1, img.size(0))
        nms_time.update(t4 - t3, img.size(0))
        waste_time.update(tw2 - tw1, img.size(0))

    print("inf : (%.4fs/frame)   nms : (%.4fs/frame)" % (inf_time.avg, nms_time.avg))
    print(f"Done. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    opt = make_parser().parse_args()
    print(opt)
    with torch.no_grad():
        detect(opt)
