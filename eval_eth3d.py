"""Zero-shot ETH3D eval — log-LS alignment, 60m cap, mixed indoor+outdoor (12 scenes)."""
import os, sys, glob, argparse
import numpy as np
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, "/home/rick/ultralytics_depth_anything")
from ultralytics import YOLO
from eval_make3d import log_ls_align, compute_metrics, infer_ms_tta

# ETH3D depth file shape — verified empirically across all 12 train scenes.
# Files are float32 little-endian raw binaries at the original DSLR sensor res.
ETH3D_DEPTH_SHAPE = (4032, 6048)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", default="/data/depth_anything/raw/eth3d")
    p.add_argument("--max_depth", type=float, default=60.0)
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--ms_sizes", type=int, nargs="+", default=[704, 768, 832])
    p.add_argument("--ms_weights", type=float, nargs="+", default=[1.0, 3.0, 1.0])
    p.add_argument("--hflip", action="store_true", default=True)
    p.add_argument("--device", default="0")
    p.add_argument("--label", default="eval")
    # Downsample pred and GT to this height before computing metrics; speeds up 4K eval ~16x.
    p.add_argument("--eval_h", type=int, default=1008)
    return p.parse_args()


def load_eth3d_depth(path):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size != ETH3D_DEPTH_SHAPE[0] * ETH3D_DEPTH_SHAPE[1]:
        raise ValueError(f"{path}: expected {ETH3D_DEPTH_SHAPE[0]*ETH3D_DEPTH_SHAPE[1]} floats, got {raw.size}")
    d = raw.reshape(ETH3D_DEPTH_SHAPE)
    d = np.where(np.isfinite(d), d, 0.0).astype(np.float32)
    return d


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    dev = "cuda:0"
    model = YOLO(args.model, task="depth")
    model.model.eval().to(dev)

    pairs = []
    for scene in sorted(os.listdir(args.data_dir)):
        scene_dir = os.path.join(args.data_dir, scene)
        if not os.path.isdir(scene_dir):
            continue
        d_dir = f"{scene_dir}/ground_truth_depth/dslr_images"
        i_dir = f"{scene_dir}/images/dslr_images_undistorted"
        if not os.path.isdir(d_dir):
            continue
        for d_path in sorted(glob.glob(f"{d_dir}/*.JPG")):
            stem = os.path.basename(d_path)
            i_path = f"{i_dir}/{stem}"
            if os.path.exists(i_path):
                pairs.append((i_path, d_path))
    print(f"ETH3D pairs: {len(pairs)}")

    # Downsample target for metrics (4K is overkill; 1008 height matches ETH3D-style 1/4)
    H_eval = args.eval_h
    W_eval = int(round(H_eval * ETH3D_DEPTH_SHAPE[1] / ETH3D_DEPTH_SHAPE[0]))

    all_m = []
    for i, (img_path, d_path) in enumerate(pairs):
        img = Image.open(img_path).convert("RGB")
        gt_full = load_eth3d_depth(d_path)
        gt = np.array(Image.fromarray(gt_full).resize((W_eval, H_eval), Image.NEAREST))
        pred_full = infer_ms_tta(model, img, args, dev)
        pred = np.array(Image.fromarray(pred_full).resize((W_eval, H_eval), Image.BILINEAR))
        met = compute_metrics(pred, gt, args.max_depth)
        if met is None:
            continue
        all_m.append(met)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pairs)}  d1={met['d1']:.4f}")

    if not all_m:
        print("ETH3D: NO valid frames"); return
    mean = {k: float(np.mean([m[k] for m in all_m])) for k in all_m[0].keys()}
    print(f"ETH3D mean: {mean}")
    print(f"TSV\t{args.label}\teth3d\t{mean['d1']:.4f}\t{mean['d2']:.4f}\t{mean['d3']:.4f}\t"
          f"{mean['abs_rel']:.4f}\t{mean['sq_rel']:.4f}\t{mean['rmse']:.4f}\t{mean['rmse_log']:.4f}\t{mean['silog']:.2f}")


if __name__ == "__main__":
    main()
