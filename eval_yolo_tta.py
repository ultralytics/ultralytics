"""YOLO depth eval with multi-scale TTA.
Usage: python eval_yolo_tta.py --checkpoint <path> --device 0
"""
import sys, os, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
import h5py

sys.path.insert(0, os.path.expanduser('~/ultralytics_depth_anything'))
from ultralytics.nn.tasks import DepthModel

EIGEN_CROP = (45, 471, 41, 601)

def load_nyu_eigen(mat_path='/data/depth_anything/nyu_depth_v2_labeled.mat'):
    with h5py.File(mat_path, 'r') as f:
        images = np.transpose(np.array(f['images']), (0, 3, 2, 1))
        depths = np.transpose(np.array(f['depths']), (0, 2, 1))
    indices = np.load('/data/depth_anything/eigen_test_indices.npy')
    return images[indices], depths[indices]

def load_model(ckpt_path, device):
    # Ensure dinov2 hub module is importable for DINOv2-based checkpoints
    import os, sys
    hub_dir = os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main')
    if os.path.isdir(hub_dir) and hub_dir not in sys.path:
        sys.path.insert(0, hub_dir)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = (ckpt.get('ema') or ckpt.get('model')).float()
    return model.to(device).eval()

def infer_single(model, img_rgb, device, imgsz):
    """Single-scale inference, returns (H, W) numpy array."""
    h, w = img_rgb.shape[:2]
    x = torch.from_numpy(img_rgb[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = F.interpolate(x, size=(imgsz, imgsz), mode='bilinear', align_corners=True).to(device)
    with torch.no_grad():
        pred = model(x)
    if isinstance(pred, dict): pred = pred['depth']
    if pred.ndim == 3: pred = pred.unsqueeze(1)
    pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
    return pred.squeeze().cpu().numpy()

def predict_tta(model, images, device, scales=(480, 518, 640), hflip=True):
    preds = []
    for i, img_rgb in enumerate(images):
        scale_preds = []
        for s in scales:
            p = infer_single(model, img_rgb, device, s)
            scale_preds.append(np.log(np.clip(p, 1e-8, None)))
            if hflip:
                p_f = infer_single(model, img_rgb[:, ::-1, :], device, s)
                scale_preds.append(np.log(np.clip(p_f[:, ::-1], 1e-8, None)))
        preds.append(np.exp(np.mean(scale_preds, axis=0)))
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(images)} images done')
    return preds

def align_log_ls(pred, gt, mask, n_iter=5, trim=0.012):
    lp = np.log(np.clip(pred[mask], 1e-8, None)).flatten()
    lg = np.log(gt[mask]).flatten()
    for _ in range(n_iter):
        A = np.stack([lp, np.ones_like(lp)], 1)
        s, t = np.linalg.lstsq(A, lg, rcond=None)[0]
        resid = np.abs(s * lp + t - lg)
        thr = np.percentile(resid, (1 - trim) * 100)
        lp, lg = lp[resid < thr], lg[resid < thr]
    return np.exp(s * np.log(np.clip(pred, 1e-8, None)) + t)

def compute_metrics(pred, gt):
    mask = (gt > 1e-3) & (gt < 10.0) & (pred > 1e-6)
    if mask.sum() < 100: return None
    p, g = pred[mask], gt[mask]
    ratio = np.maximum(p / g, g / p)
    delta1 = (ratio < 1.25).mean()
    delta2 = (ratio < 1.25**2).mean()
    delta3 = (ratio < 1.25**3).mean()
    abs_rel = np.mean(np.abs(p - g) / g)
    rmse = np.sqrt(np.mean((p - g)**2))
    log_p, log_g = np.log(p), np.log(g)
    d = log_p - log_g
    silog = np.sqrt(np.mean(d**2) - np.mean(d)**2) * 100
    return dict(delta1=delta1, delta2=delta2, delta3=delta3, abs_rel=abs_rel, rmse=rmse, silog=silog)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--scales', default='480,518,640')
    parser.add_argument('--no-hflip', action='store_true')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}')
    scales = [int(s) for s in args.scales.split(',')]
    hflip = not args.no_hflip

    print(f'Loading {args.checkpoint}')
    model = load_model(args.checkpoint, device)
    print(f'TTA: scales={scales}, hflip={hflip}')

    images, depths = load_nyu_eigen()
    print(f'Loaded {len(images)} NYU test images')

    t0 = time.time()
    preds = predict_tta(model, images, device, scales, hflip)
    print(f'TTA inference: {time.time()-t0:.1f}s')

    # Evaluate with robust log-LS alignment
    all_metrics = []
    for i in range(len(images)):
        t, b, l, r = EIGEN_CROP
        pc, gc = preds[i][t:b, l:r], depths[i][t:b, l:r]
        mask = (gc > 1e-3) & (gc < 10.0) & (pc > 1e-6)
        if mask.sum() < 100: continue
        aligned = align_log_ls(pc, gc, mask)
        aligned = np.clip(aligned, 1e-3, 10.0)
        m = compute_metrics(aligned, gc)
        if m: all_metrics.append(m)

    agg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    print(f'\n=== TTA results ({len(scales)} scales, hflip={hflip}) ===')
    print(f'  delta1={agg["delta1"]:.4f}  delta2={agg["delta2"]:.4f}  delta3={agg["delta3"]:.4f}')
    print(f'  abs_rel={agg["abs_rel"]:.4f}  rmse={agg["rmse"]:.4f}  silog={agg["silog"]:.4f}')
    print(f'  ({len(all_metrics)} valid images evaluated)')

if __name__ == '__main__':
    main()
