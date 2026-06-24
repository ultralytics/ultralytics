#!/usr/bin/env python
"""Build the NYU Depth V2 dataset in Ultralytics depth (YOLO) format from the
official labeled .mat file.

Layout produced (matches DepthDataset: images/<split>/<stem>.jpg paired with
depth/<split>/<stem>.npy, float32 metres):

    <out>/
        images/train/*.jpg   depth/train/*.npy   (795 images)
        images/val/*.jpg     depth/val/*.npy     (654 images, Eigen test split)

Transpose / split logic mirrors eval_depth.py so val matches the standard
Eigen test set used for benchmarking.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def load_nyu_mat(mat_path: str):
    """Return (images (N,H,W,3) uint8 RGB, depths (N,H,W) float32 metres)."""
    print(f"Loading {mat_path} ...")
    with h5py.File(mat_path, "r") as f:
        images = np.array(f["images"])  # (N, 3, W, H)
        depths = np.array(f["depths"])  # (N, W, H)
    images = np.transpose(images, (0, 3, 2, 1)).astype(np.uint8)  # (N, H, W, 3)
    depths = np.transpose(depths, (0, 2, 1)).astype(np.float32)   # (N, H, W)
    print(f"  {images.shape[0]} images, HxW={images.shape[1:3]}, depth=[{depths.min():.2f},{depths.max():.2f}]m")
    return images, depths


def eigen_test_indices(npy_path: str, n_total: int):
    p = Path(npy_path)
    if not p.exists():
        raise FileNotFoundError(f"Eigen test indices not found: {p}")
    idx = np.load(str(p)).astype(int).tolist()
    print(f"  {len(idx)} Eigen test (val) indices from {p}")
    return set(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="/data/depth_anything/nyu_depth_v2_labeled.mat")
    ap.add_argument("--eigen", default="/data/depth_anything/eigen_test_indices.npy")
    ap.add_argument("--out", default="/data/depth_anything/nyu-depth")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    images, depths = load_nyu_mat(args.mat)
    n = images.shape[0]
    val_set = eigen_test_indices(args.eigen, n)

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "depth" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for i in range(n):
        split = "val" if i in val_set else "train"
        stem = f"nyu_{i:04d}"
        Image.fromarray(images[i]).save(
            out / "images" / split / f"{stem}.jpg", quality=args.quality
        )
        # float16 metres: ~half the disk/zip size; loader upcasts to float32 on load.
        np.save(out / "depth" / split / f"{stem}.npy", depths[i].astype(np.float16))
        counts[split] += 1

    print(f"Done: train={counts['train']} val={counts['val']} -> {out}")


if __name__ == "__main__":
    main()
