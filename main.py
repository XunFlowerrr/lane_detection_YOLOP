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
    return parser


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

    t0 = time.time()
    for path, img, im0w, orig, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img)
        t2 = time_synchronized()

        tw1 = time_synchronized()
        pred = split_for_trace_model(pred, anchor_grid)
        tw2 = time_synchronized()

        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)   # shape: (work_h, work_w) e.g. (720, 1280)
        ll_seg_mask = lane_line_mask(ll)       # shape: (work_h, work_w)
        if opt.lanes_only:
            da_seg_mask = np.zeros_like(da_seg_mask)

        # Upscale masks from work-canvas back to the ORIGINAL image size
        h0, w0 = orig.shape[:2]
        da_seg_mask = cv2.resize(da_seg_mask.astype(np.uint8), (w0, h0),
                                 interpolation=cv2.INTER_NEAREST)
        ll_seg_mask = cv2.resize(ll_seg_mask.astype(np.uint8), (w0, h0),
                                 interpolation=cv2.INTER_NEAREST)

        for i, det in enumerate(pred):
            p, s, frame = path, "", getattr(dataset, "frame", 0)
            im0 = orig  # draw everything on the original-resolution canvas
            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")
            s += "%gx%g " % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if len(det):
                # Scale detection boxes from model input space directly to the
                # original image size (skip the work-canvas intermediate).
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
            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)

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
