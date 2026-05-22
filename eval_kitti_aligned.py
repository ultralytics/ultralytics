"""Zero-shot KITTI eval for YOLO depth model — log-LS alignment, 80m cap, Garg crop."""
import os, sys, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", default="/data/depth_anything/kitti_yolo")
    p.add_argument("--max_depth", type=float, default=80.0)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--label", type=str, default="eval")
    return p.parse_args()

def log_ls_align(pred, gt_valid):
    log_pred = np.log(np.clip(pred, 1e-6, None))
    log_gt   = np.log(np.clip(gt_valid, 1e-6, None))
    A = np.stack([log_pred, np.ones_like(log_pred)], axis=1)
    s, t = np.linalg.lstsq(A, log_gt, rcond=None)[0]
    return np.exp(s * log_pred + t)

def compute_metrics(pred, gt, max_depth=80.0):
    gt_crop   = gt[153:371, :]
    pred_crop = pred[153:371, :]
    valid = (gt_crop > 0.1) & (gt_crop < max_depth)
    if valid.sum() < 10:
        return None
    gt_v, pred_v = gt_crop[valid], pred_crop[valid]
    pred_a = log_ls_align(pred_v, gt_v)
    thresh = np.maximum(gt_v / pred_a, pred_a / gt_v)
    log_d  = np.log(pred_a) - np.log(gt_v)
    return dict(
        d1       = (thresh < 1.25  ).mean(),
        d2       = (thresh < 1.25**2).mean(),
        d3       = (thresh < 1.25**3).mean(),
        abs_rel  = np.mean(np.abs(gt_v - pred_a) / gt_v),
        sq_rel   = np.mean((gt_v - pred_a)**2 / gt_v),
        rmse     = np.sqrt(np.mean((gt_v - pred_a)**2)),
        rmse_log = np.sqrt(np.mean((np.log(gt_v) - np.log(pred_a))**2)),
        silog    = np.sqrt(np.mean(log_d**2) - np.mean(log_d)**2) * 100,
    )

def main():
    args = parse_args()
    dev = f"cuda:{args.device}"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    sys.path.insert(0, "/home/rick/ultralytics_depth_anything")
    from ultralytics import YOLO
    model = YOLO(args.model, task="depth")
    model.model.eval().to(dev)

    img_files = sorted(glob.glob(f"{args.data_dir}/images/*.png"))
    print(f"Found {len(img_files)} KITTI images")

    all_metrics = []
    for i, img_path in enumerate(img_files):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        depth_path = f"{args.data_dir}/depth/{stem}.npy"
        if not os.path.exists(depth_path):
            continue

        gt = np.load(depth_path).astype(np.float32)
        H_orig, W_orig = gt.shape

        img = Image.open(img_path).convert("RGB")
        img_r = img.resize((args.imgsz, args.imgsz), Image.BILINEAR)
        x = torch.from_numpy(np.array(img_r)).permute(2,0,1).float().div(255).unsqueeze(0).to(dev)

        with torch.no_grad():
            out = model.model(x)
            if isinstance(out, dict):
                out = out["depth"]
            if out.ndim == 4:
                out = out[0, 0]
            elif out.ndim == 3:
                out = out[0]

        pred_full = F.interpolate(
            out.unsqueeze(0).unsqueeze(0).float(),
            (H_orig, W_orig), mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()

        m = compute_metrics(pred_full, gt, max_depth=args.max_depth)
        if m:
            all_metrics.append(m)

        if (i+1) % 500 == 0:
            a = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
            print(f"[{i+1}/{len(img_files)}] d1={a['d1']:.4f} abs_rel={a['abs_rel']:.4f} rmse={a['rmse']:.3f}")

    a = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    print(f"\n=== KITTI Zero-Shot ({len(all_metrics)} imgs, cap={args.max_depth}m, log-LS align) ===")
    print(f"  delta1:   {a['d1']:.4f}")
    print(f"  delta2:   {a['d2']:.4f}")
    print(f"  delta3:   {a['d3']:.4f}")
    print(f"  abs_rel:  {a['abs_rel']:.4f}")
    print(f"  sq_rel:   {a['sq_rel']:.4f}")
    print(f"  rmse:     {a['rmse']:.4f}")
    print(f"  rmse_log: {a['rmse_log']:.4f}")
    print(f"  silog:    {a['silog']:.4f}")
    print(f"TSV\t{args.label}\tkitti_eigen\t{a['d1']:.4f}\t{a['d2']:.4f}\t{a['d3']:.4f}\t"
          f"{a['abs_rel']:.4f}\t{a['sq_rel']:.4f}\t{a['rmse']:.4f}\t{a['rmse_log']:.4f}\t{a['silog']:.2f}")

if __name__ == "__main__":
    main()
