# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Audit pseudo-labels against the COCO ground truth they were generated from.

The teacher's 3D is unverifiable without a 3D benchmark, but its **2D projection** is directly comparable to
COCO's human-annotated keypoints. Agreement there is a necessary (not sufficient) condition for the 3D being
right: a reconstruction that does not even reproject onto the person cannot be correct in depth either.

Reports OKS between pseudo and GT 2D keypoints over COCO-visible joints, plus the distribution of decoded root
depth and relative depth, so out-of-range encoding clipping shows up as a spike at the bounds.

Usage:
    python research/src/check_pseudo_labels.py --pseudo <labels3d dir> --gt <coco labels dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.pose3d import Z_REL_RANGE, Z_ROOT_MAX, decode_z


def main(pseudo: Path, gt: Path, limit: int) -> None:
    """Compare pseudo-labels with COCO GT and print OKS plus depth distributions."""
    oks_all, z_roots, z_rels, clipped = [], [], [], 0
    files = sorted(pseudo.glob("*.txt"))[:limit]
    for f in files:
        g_path = gt / f.name
        if not g_path.exists():
            continue
        p = np.array([x.split() for x in f.read_text().strip().splitlines()], dtype=np.float32)
        g = np.array([x.split() for x in g_path.read_text().strip().splitlines()], dtype=np.float32)
        g = g[g[:, 0] == 0]
        if len(p) != len(g):  # rows are written in GT order, so a length mismatch means a dropped person
            continue
        pk = p[:, 5:].reshape(len(p), -1, 4)
        gk = g[:, 5:].reshape(len(g), -1, 3)
        area = (g[:, 3] * g[:, 4]).clip(1e-6)  # normalized box area, matching normalized keypoint coords

        vis = gk[..., 2] > 0
        d2 = ((pk[:, :17, :2] - gk[..., :2]) ** 2).sum(-1)
        e = d2 / (2 * (OKS_SIGMA[None] ** 2) * (area[:, None] + 1e-9)) / 2
        oks = (np.exp(-e) * vis).sum(1) / vis.sum(1).clip(1)
        oks_all += oks[vis.sum(1) > 0].tolist()

        z_rel, z_root = decode_z(pk[:, :17, 3], pk[:, 17, 3])
        z_roots += z_root.tolist()
        z_rels += z_rel.flatten().tolist()
        clipped += int((pk[:, 17, 3] >= 1.0).sum() + (pk[:, :17, 3] <= 0.0).sum() + (pk[:, :17, 3] >= 1.0).sum())

    oks_all, z_roots, z_rels = np.array(oks_all), np.array(z_roots), np.array(z_rels)
    print(f"files: {len(files)}   persons scored: {len(oks_all)}")
    print(
        f"2D OKS vs COCO GT: mean {oks_all.mean():.3f}  median {np.median(oks_all):.3f}  "
        f"frac>0.5 {(oks_all > 0.5).mean():.3f}  frac>0.75 {(oks_all > 0.75).mean():.3f}"
    )
    print(
        f"root depth (m): p5 {np.percentile(z_roots, 5):.2f}  median {np.median(z_roots):.2f}  "
        f"p95 {np.percentile(z_roots, 95):.2f}  max {z_roots.max():.2f}  (encoding cap {Z_ROOT_MAX})"
    )
    print(
        f"relative depth (m): p1 {np.percentile(z_rels, 1):+.2f}  p99 {np.percentile(z_rels, 99):+.2f}  "
        f"absmax {np.abs(z_rels).max():.2f}  (encoding range +/-{Z_REL_RANGE})"
    )
    print(f"values at an encoding bound: {clipped} ({clipped / max(len(z_rels) + len(z_roots), 1):.4%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100000)
    main(**vars(parser.parse_args()))
