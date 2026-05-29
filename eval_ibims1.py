"""Zero-shot iBims-1 eval — log-LS alignment, 10m cap (indoor), 100 frames."""
import os, sys, glob, argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/rick/ultralytics_depth_anything")
import os, sys
_hub = os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main')
if os.path.isdir(_hub) and _hub not in sys.path:
    sys.path.insert(0, _hub)

from ultralytics import YOLO
from eval_make3d import log_ls_align, compute_metrics, infer_ms_tta


# DINOv2 hub module is needed when loading DINOv2-based checkpoints
import os as _os
import sys as _sys
_hub = _os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main')
if _os.path.isdir(_hub) and _hub not in _sys.path:
    _sys.path.insert(0, _hub)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", default="/data/depth_anything/raw/ibims1/ibims1_core_raw")
    p.add_argument("--max_depth", type=float, default=10.0)
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--ms_sizes", type=int, nargs="+", default=[704, 768, 832])
    p.add_argument("--ms_weights", type=float, nargs="+", default=[1.0, 3.0, 1.0])
    p.add_argument("--hflip", action="store_true", default=True)
    p.add_argument("--device", default="0")
    p.add_argument("--label", default="eval")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    dev = "cuda:0"
    model = YOLO(args.model, task="depth")
    model.model.eval().to(dev)

    rgbs = sorted(glob.glob(f"{args.data_dir}/rgb/*.png"))
    print(f"iBims-1 frames: {len(rgbs)}")

    all_m = []
    for i, rgb_path in enumerate(rgbs):
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        d_path = f"{args.data_dir}/depth/{stem}.png"
        if not os.path.exists(d_path):
            continue
        img = Image.open(rgb_path).convert("RGB")
        # iBims-1 depth encoding (Koch et al. 2018, readme): depth_m = uint16 * 50 / 65535.
        gt = np.array(Image.open(d_path), dtype=np.float32) * 50.0 / 65535.0
        # Apply mask_invalid if present (zero-out invalid pixels — they have no GT).
        mask_path = f"{args.data_dir}/mask_invalid/{stem}.png"
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path)) > 0
            gt[~mask] = 0.0
        Hd, Wd = gt.shape
        pred = infer_ms_tta(model, img, args, dev)
        pred_at_gt = np.array(Image.fromarray(pred).resize((Wd, Hd), Image.BILINEAR))
        met = compute_metrics(pred_at_gt, gt, args.max_depth)
        if met is None:
            continue
        all_m.append(met)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rgbs)}  d1={met['d1']:.4f}")

    mean = {k: float(np.mean([m[k] for m in all_m])) for k in all_m[0].keys()}
    print(f"iBims-1 mean: {mean}")
    print(f"TSV\t{args.label}\tibims1\t{mean['d1']:.4f}\t{mean['d2']:.4f}\t{mean['d3']:.4f}\t"
          f"{mean['abs_rel']:.4f}\t{mean['sq_rel']:.4f}\t{mean['rmse']:.4f}\t{mean['rmse_log']:.4f}\t{mean['silog']:.2f}")


if __name__ == "__main__":
    main()
