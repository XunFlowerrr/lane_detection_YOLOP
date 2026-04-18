# YOLOPv2 Lane Detection

Inference wrapper for [YOLOPv2](https://github.com/CAIC-AD/YOLOPv2) — joint vehicle detection, drivable area segmentation, and lane line detection.

## Setup

```bash
uv sync
```

Place the model weights at `model/yolopv2.pt`. Download from:
```
https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt
```

## Quick Usage

Run on the default test images:
```bash
uv run main.py
```

Run on a custom image or folder:
```bash
uv run main.py --source path/to/image.jpg
uv run main.py --source path/to/folder/
```

Show lane lines only (no drivable area overlay, no vehicle boxes):
```bash
uv run main.py --lanes-only
```

Results are saved to `runs/detect/exp/`.

## Options

| Option | Default | Description |
|---|---|---|
| `--weights` | `model/yolopv2.pt` | Path to model weights |
| `--source` | `test_images/` | Input image, folder, or video |
| `--img-size` | `640` | Inference image size (pixels) |
| `--conf-thres` | `0.3` | Object detection confidence threshold |
| `--iou-thres` | `0.45` | NMS IoU threshold |
| `--device` | `cpu` | Device to run on (`cpu`, `0`, `0,1,2,3`) |
| `--lanes-only` | `False` | Show only lane line overlay, hide drivable area and vehicle boxes |
| `--nosave` | `False` | Do not save output images/videos |
| `--save-txt` | `False` | Save detection results to `.txt` files |
| `--save-conf` | `False` | Include confidence scores in saved `.txt` files |
| `--classes` | all | Filter detections by class index (e.g. `--classes 0 2`) |
| `--agnostic-nms` | `False` | Class-agnostic NMS |
| `--project` | `runs/detect` | Directory to save results |
| `--name` | `exp` | Run name (results saved to `project/name/`) |
| `--exist-ok` | `False` | Overwrite existing run folder instead of incrementing |

## Output

Each run produces annotated images (or video) in `runs/detect/<name>/` with:
- **Green overlay** — drivable area
- **Red overlay** — lane lines
- **Cyan boxes** — detected vehicles

With `--lanes-only`, only the red lane line overlay is shown.
