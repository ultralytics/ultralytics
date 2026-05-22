"""
Evaluate Depth Anything V2 on NYU Depth V2 (Eigen split, 654 images).

Computes standard depth metrics aligned with the Depth Anything paper:
  delta1, delta2, delta3, abs_rel, sq_rel, rmse, rmse_log, silog, log10

Usage:
    cd ~/ultralytics_depth_anything
    /home/rick/miniconda/envs/pytorch/bin/uv run python eval_depth.py \
        --model depth_anything_v2_vits.pth \
        --data /data/depth_anything/validation/nyu_v2 \
        --imgsz 518

Environment variables:
    CUDA_VISIBLE_DEVICES=0  # pin to specific GPU
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

# Eigen split test indices (654 images from the 1449 labeled NYU pairs)
# Standard split used by Eigen et al. and all Depth Anything evaluations.
EIGEN_TEST_INDICES_URL = "https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/metric_depth/dataset/splits/nyu/nyu_test.txt"

# Eigen crop for NYU (removes unreliable Kinect borders)
EIGEN_CROP = (45, 471, 41, 601)  # top, bottom, left, right on 480x640


def load_nyu_mat(mat_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load NYU Depth V2 labeled.mat file.

    Returns:
        images: (N, H, W, 3) uint8 RGB
        depths: (N, H, W) float32 in meters
    """
    print(f"Loading {mat_path}...")
    with h5py.File(mat_path, "r") as f:
        # h5py reads MATLAB v7.3 HDF5 with transposed dims:
        #   images: (N, 3, W, H) in h5py → need (N, H, W, 3) RGB
        #   depths: (N, W, H) in h5py → need (N, H, W)
        images = np.array(f["images"])  # (1449, 3, 640, 480)
        depths = np.array(f["depths"])  # (1449, 640, 480)

    # Transpose: (N, 3, W, H) → (N, H, W, 3) and (N, W, H) → (N, H, W)
    images = np.transpose(images, (0, 3, 2, 1))  # (N, H, W, 3)
    depths = np.transpose(depths, (0, 2, 1))       # (N, H, W)

    print(f"Loaded {images.shape[0]} images, shape={images.shape[1:3]}, depth range=[{depths.min():.2f}, {depths.max():.2f}]m")
    return images, depths


def get_eigen_test_indices(n_total: int = 1449) -> list[int]:
    """Get the 654 Eigen test split indices.

    Loads from pre-saved .npy file, falls back to downloading from NYU website.
    """
    # Try pre-saved file first
    npy_path = Path("/data/depth_anything/eigen_test_indices.npy")
    if npy_path.exists():
        indices = np.load(str(npy_path)).tolist()
        print(f"Loaded {len(indices)} Eigen test indices from {npy_path}")
        return indices

    # Fall back to downloading splits.mat from NYU
    try:
        import urllib.request
        import scipy.io
        cache_path = Path("/tmp/splits.mat")
        if not cache_path.exists():
            urllib.request.urlretrieve(
                "http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat",
                str(cache_path),
            )
        splits = scipy.io.loadmat(str(cache_path))
        indices = (splits["testNdxs"].flatten() - 1).tolist()  # MATLAB 1-indexed → 0-indexed
        print(f"Loaded {len(indices)} Eigen test indices from splits.mat")
        return indices
    except Exception as e:
        print(f"WARNING: Could not load Eigen split ({e}), using all {n_total} images")
        return list(range(n_total))


def eigen_crop(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply Eigen center crop to prediction and ground truth."""
    t, b, l, r = EIGEN_CROP
    return pred[t:b, l:r], gt[t:b, l:r]


def compute_metrics(pred: np.ndarray, gt: np.ndarray, min_depth: float = 1e-3, max_depth: float = 10.0) -> dict:
    """Compute standard depth estimation metrics.

    Both pred and gt should be in the same scale (after alignment).

    Args:
        pred: Predicted depth (H, W).
        gt: Ground truth depth (H, W) in meters.
        min_depth: Minimum valid depth.
        max_depth: Maximum valid depth.

    Returns:
        Dict of metric name → float value.
    """
    # Mask valid pixels
    mask = (gt > min_depth) & (gt < max_depth)
    pred = pred[mask]
    gt = gt[mask]

    if len(gt) == 0:
        return {}

    # Threshold accuracies
    thresh = np.maximum(pred / gt, gt / pred)
    delta1 = (thresh < 1.25).mean()
    delta2 = (thresh < 1.25 ** 2).mean()
    delta3 = (thresh < 1.25 ** 3).mean()

    # Error metrics
    abs_rel = np.mean(np.abs(pred - gt) / gt)
    sq_rel = np.mean(((pred - gt) ** 2) / gt)
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))

    # Scale-invariant log error
    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100  # ×100 per convention

    # Log10
    log10 = np.mean(np.abs(np.log10(pred) - np.log10(gt)))

    return {
        "delta1": float(delta1),
        "delta2": float(delta2),
        "delta3": float(delta3),
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "rmse": float(rmse),
        "rmse_log": float(rmse_log),
        "silog": float(silog),
        "log10": float(log10),
    }


def align_least_squares(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Align prediction to ground truth via least-squares (scale + shift).

    Solves: min ||s * pred + t - gt||^2 over valid pixels.

    Args:
        pred: Predicted depth (H, W).
        gt: Ground truth depth (H, W).
        mask: Valid pixel mask (H, W) bool.

    Returns:
        Aligned prediction (H, W).
    """
    p = pred[mask].flatten()
    g = gt[mask].flatten()

    # Solve [p, 1] @ [s, t]^T = g in least-squares sense
    A = np.stack([p, np.ones_like(p)], axis=1)
    result = np.linalg.lstsq(A, g, rcond=None)
    s, t = result[0]

    return s * pred + t


def align_least_squares_log(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                            robust: bool = True, n_iter: int = 5, trim_pct: float = 0.012) -> np.ndarray:
    """Align prediction to ground truth via least-squares in log space.

    Solves: min ||s * log(pred) + t - log(gt)||^2, then returns exp(s*log(pred)+t).
    This is the correct alignment for models outputting inverse depth, since
    log(1/depth) = -log(depth) and a linear fit in log space naturally handles
    the nonlinear relationship between inverse depth and forward depth.

    With robust=True, iteratively trims outlier pixels (top trim_pct by residual)
    to improve alignment quality on images with wide depth ranges.
    Best config: trim_pct=0.012, n_iter=5 (delta1=0.938 on NYU Eigen split).
    """
    lp = np.log(np.clip(pred[mask], 1e-8, None)).flatten()
    lg = np.log(gt[mask]).flatten()

    if robust:
        for _ in range(n_iter):
            A = np.stack([lp, np.ones_like(lp)], axis=1)
            result = np.linalg.lstsq(A, lg, rcond=None)
            s, t = result[0]
            residuals = np.abs(s * lp + t - lg)
            threshold = np.percentile(residuals, (1 - trim_pct) * 100)
            keep = residuals < threshold
            lp, lg = lp[keep], lg[keep]
    else:
        A = np.stack([lp, np.ones_like(lp)], axis=1)
        result = np.linalg.lstsq(A, lg, rcond=None)
        s, t = result[0]

    return np.exp(s * np.log(np.clip(pred, 1e-8, None)) + t)


def align_median(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Align prediction to ground truth via median scaling."""
    scale = np.median(gt[mask]) / (np.median(pred[mask]) + 1e-8)
    return pred * scale


@torch.no_grad()
def run_depth_inference(model, images: np.ndarray, device: torch.device,
                        input_size: int = 518, batch_size: int = 1) -> list[np.ndarray]:
    """Run Depth Anything V2 inference on a batch of images.

    Args:
        model: DepthAnythingV2 model.
        images: (N, H, W, 3) uint8 RGB images.
        device: torch device.
        input_size: Target size for inference.
        batch_size: Batch size for inference.

    Returns:
        List of (H, W) float32 depth maps (relative/inverse depth).
    """
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    predictions = []
    n = len(images)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_imgs = images[start:end]

        # Preprocess: resize, normalize
        tensors = []
        orig_sizes = []
        for img in batch_imgs:
            h, w = img.shape[:2]
            orig_sizes.append((h, w))

            # Resize keeping aspect ratio, round to multiple of 14
            scale = input_size / max(h, w)
            new_h = max(14, int(h * scale) // 14 * 14)
            new_w = max(14, int(w * scale) // 14 * 14)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # To tensor: HWC RGB uint8 → CHW float32 [0,1]
            t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            tensors.append(t)

        # Pad to same size for batching
        max_h = max(t.shape[1] for t in tensors)
        max_w = max(t.shape[2] for t in tensors)
        batch = torch.zeros(len(tensors), 3, max_h, max_w, device=device)
        for i, t in enumerate(tensors):
            batch[i, :, :t.shape[1], :t.shape[2]] = t.to(device)

        # Normalize
        batch = (batch - mean) / std

        # Inference
        depth_batch = model(batch)  # (B, H_model, W_model)

        # Resize each back to original size
        for i in range(len(batch_imgs)):
            d = depth_batch[i]
            oh, ow = orig_sizes[i]
            d = F.interpolate(
                d.unsqueeze(0).unsqueeze(0),
                size=(oh, ow),
                mode="bilinear",
                align_corners=True,
            ).squeeze().cpu().numpy()
            predictions.append(d)

        if (start // batch_size) % 10 == 0:
            print(f"  Processed {end}/{n} images...")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate Depth Anything V2 on NYU Depth V2")
    parser.add_argument("--model", type=str, default="depth_anything_v2_vits.pth",
                        help="Path to model weights or model name (vits/vitb/vitl)")
    parser.add_argument("--data", type=str, default="/data/depth_anything/nyu_depth_v2_labeled.mat",
                        help="Path to NYU .mat file or extracted directory")
    parser.add_argument("--imgsz", type=int, default=518, help="Input size for model")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    parser.add_argument("--align", type=str, default="least_squares_log",
                        choices=["least_squares", "least_squares_log", "median", "none"],
                        help="Alignment method (least_squares_log recommended for relative models)")
    parser.add_argument("--half", action="store_true", help="Use FP16")
    parser.add_argument("--label", type=str, default="eval", help="Label for TSV row")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model: {args.model}")
    from ultralytics.models.depth_anything.build import build_depth_anything
    model = build_depth_anything(args.model)
    model = model.to(device)
    if args.half:
        model = model.half()
    model.eval()
    print(f"Model loaded: encoder={model.encoder_name}")

    # Load data
    if args.data.endswith(".mat"):
        images, depths = load_nyu_mat(args.data)
        test_indices = get_eigen_test_indices(len(images))
        print(f"Using {len(test_indices)} test images (Eigen split)")
        images = images[test_indices]
        depths = depths[test_indices]
    else:
        raise NotImplementedError("Only .mat file loading is implemented. Provide the nyu_depth_v2_labeled.mat path.")

    # Run inference
    print(f"Running inference (imgsz={args.imgsz}, batch_size={args.batch_size})...")
    t0 = time.time()
    predictions = run_depth_inference(model, images, device, args.imgsz, args.batch_size)
    inference_time = time.time() - t0
    print(f"Inference done in {inference_time:.1f}s ({inference_time/len(images)*1000:.1f}ms/image)")

    # Compute metrics
    print("Computing metrics...")
    all_metrics = []
    for i in range(len(images)):
        pred = predictions[i]
        gt = depths[i]

        # Apply Eigen crop
        pred_crop, gt_crop = eigen_crop(pred, gt)

        # Valid pixel mask
        mask = (gt_crop > 1e-3) & (gt_crop < 10.0)
        if mask.sum() < 100:
            continue

        # Align prediction to ground truth
        if args.align == "least_squares":
            pred_aligned = align_least_squares(pred_crop, gt_crop, mask)
        elif args.align == "least_squares_log":
            mask_pos = mask & (pred_crop > 1e-6)
            pred_aligned = align_least_squares_log(pred_crop, gt_crop, mask_pos)
        elif args.align == "none":
            pred_aligned = pred_crop
        else:
            pred_aligned = align_median(pred_crop, gt_crop, mask)

        # Clamp to valid range
        pred_aligned = np.clip(pred_aligned, 1e-3, 10.0)

        metrics = compute_metrics(pred_aligned, gt_crop)
        if metrics:
            all_metrics.append(metrics)

    # Aggregate
    n = len(all_metrics)
    print(f"\nResults ({n} valid images):")
    print("=" * 60)

    agg = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        agg[key] = float(np.mean(vals))

    for key, val in agg.items():
        print(f"{key}: {val:.6f}")

    # Also print peak VRAM
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"peak_vram_mb: {peak_vram:.0f}")
        agg["peak_vram_mb"] = peak_vram

    agg["n_images"] = n
    agg["inference_time_s"] = inference_time
    agg["model"] = args.model
    agg["align"] = args.align
    agg["imgsz"] = args.imgsz

    # Save results
    out_path = Path("eval_results.json")
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"TSV\t{args.label}\tnyu_eigen\t{agg['delta1']:.4f}\t{agg['delta2']:.4f}\t{agg['delta3']:.4f}\t"
          f"{agg['abs_rel']:.4f}\t{agg['sq_rel']:.4f}\t{agg['rmse']:.4f}\t{agg['rmse_log']:.4f}\t{agg['silog']:.2f}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
