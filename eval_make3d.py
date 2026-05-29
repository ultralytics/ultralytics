"""Zero-shot Make3D eval — log-LS alignment, 70m cap, central crop to match GT aspect."""
import os, sys, glob, argparse
import numpy as np
import torch
import scipy.io as sio
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, "/home/rick/ultralytics_depth_anything")
import os, sys
_hub = os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main')
if os.path.isdir(_hub) and _hub not in sys.path:
    sys.path.insert(0, _hub)

from ultralytics import YOLO


# DINOv2 hub module is needed when loading DINOv2-based checkpoints
import os as _os
import sys as _sys
_hub = _os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main')
if _os.path.isdir(_hub) and _hub not in _sys.path:
    _sys.path.insert(0, _hub)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", default="/data/depth_anything/raw/make3d")
    p.add_argument("--max_depth", type=float, default=70.0)
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--ms_sizes", type=int, nargs="+", default=[704, 768, 832])
    p.add_argument("--ms_weights", type=float, nargs="+", default=[1.0, 3.0, 1.0])
    p.add_argument("--hflip", action="store_true", default=True)
    p.add_argument("--device", default="0")
    p.add_argument("--label", default="eval")
    return p.parse_args()


def log_ls_align(pred, gt_valid, trim=0.012, n_iter=5):
    log_pred = np.log(np.clip(pred, 1e-6, None))
    log_gt   = np.log(np.clip(gt_valid, 1e-6, None))
    keep = np.ones_like(log_pred, dtype=bool)
    s, t = 1.0, 0.0
    for _ in range(n_iter):
        A = np.stack([log_pred[keep], np.ones(keep.sum())], axis=1)
        s, t = np.linalg.lstsq(A, log_gt[keep], rcond=None)[0]
        resid = np.abs((s * log_pred + t) - log_gt)
        cutoff = np.quantile(resid, 1.0 - trim)
        keep = resid <= cutoff
    return np.exp(s * log_pred + t)


def compute_metrics(pred, gt, cap):
    valid = (gt > 0.1) & (gt < cap)
    if valid.sum() < 10:
        return None
    gt_v, pred_v = gt[valid], pred[valid]
    pred_a = log_ls_align(pred_v, gt_v)
    thresh = np.maximum(gt_v / pred_a, pred_a / gt_v)
    log_d  = np.log(pred_a) - np.log(gt_v)
    return dict(
        d1=(thresh < 1.25).mean(), d2=(thresh < 1.25**2).mean(), d3=(thresh < 1.25**3).mean(),
        abs_rel=np.mean(np.abs(gt_v - pred_a) / gt_v),
        sq_rel =np.mean((gt_v - pred_a) ** 2 / gt_v),
        rmse   =np.sqrt(np.mean((gt_v - pred_a) ** 2)),
        rmse_log=np.sqrt(np.mean((np.log(gt_v) - np.log(pred_a)) ** 2)),
        silog  =np.sqrt(np.mean(log_d**2) - np.mean(log_d) ** 2) * 100,
    )


def infer_ms_tta(model, img_pil, args, dev):
    """MS + optional hflip TTA, log-merge. Returns float32 depth at largest size."""
    H_target = max(args.ms_sizes)
    W_target = H_target
    log_accum = None
    weight_sum = 0.0
    for size, w in zip(args.ms_sizes, args.ms_weights):
        for flip in ([False, True] if args.hflip else [False]):
            img_r = img_pil.resize((size, size), Image.BILINEAR)
            if flip:
                img_r = img_r.transpose(Image.FLIP_LEFT_RIGHT)
            x = torch.from_numpy(np.array(img_r)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
            with torch.no_grad():
                out = model.model(x)
            if isinstance(out, dict):
                out = out["depth"]
            if out.ndim == 4:
                out = out[0, 0]
            elif out.ndim == 3:
                out = out[0]
            d = out.detach().cpu().numpy().astype(np.float32)
            if flip:
                d = d[:, ::-1].copy()
            d_resized = np.array(Image.fromarray(d).resize((W_target, H_target), Image.BILINEAR))
            log_d = np.log(np.clip(d_resized, 1e-6, None))
            log_accum = (w * log_d) if log_accum is None else (log_accum + w * log_d)
            weight_sum += w
    return np.exp(log_accum / weight_sum)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    dev = "cuda:0"
    model = YOLO(args.model, task="depth")
    model.model.eval().to(dev)

    pairs = []
    for img_path in sorted(glob.glob(f"{args.data_dir}/Test134/*.jpg")):
        stem = os.path.basename(img_path).replace("img-", "").replace(".jpg", "")
        mat_path = f"{args.data_dir}/Gridlaserdata/depth_sph_corr-{stem}.mat"
        if os.path.exists(mat_path):
            pairs.append((img_path, mat_path))
    print(f"Make3D pairs: {len(pairs)}")

    all_m = []
    for i, (img_path, mat_path) in enumerate(pairs):
        img = Image.open(img_path).convert("RGB")
        m = sio.loadmat(mat_path)
        # Make3D depth GT: Position3DGrid[:,:,3] is the depth (m). Shape ~ (55, 305).
        gt = m["Position3DGrid"][:, :, 3].astype(np.float32)
        Hd, Wd = gt.shape
        Wi, Hi = img.size
        target_w = Wi
        target_h = int(round(Wi * Hd / Wd))
        top = max(0, (Hi - target_h) // 2)
        img_crop = img.crop((0, top, Wi, min(Hi, top + target_h)))
        pred = infer_ms_tta(model, img_crop, args, dev)
        pred_at_gt = np.array(Image.fromarray(pred).resize((Wd, Hd), Image.BILINEAR))
        met = compute_metrics(pred_at_gt, gt, args.max_depth)
        if met is None:
            continue
        all_m.append(met)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(pairs)}  d1={met['d1']:.4f}")

    mean = {k: float(np.mean([m[k] for m in all_m])) for k in all_m[0].keys()}
    print(f"Make3D mean: {mean}")
    print(f"TSV\t{args.label}\tmake3d\t{mean['d1']:.4f}\t{mean['d2']:.4f}\t{mean['d3']:.4f}\t"
          f"{mean['abs_rel']:.4f}\t{mean['sq_rel']:.4f}\t{mean['rmse']:.4f}\t{mean['rmse_log']:.4f}\t{mean['silog']:.2f}")


if __name__ == "__main__":
    main()
