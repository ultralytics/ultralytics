# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Audit a pose3d label root: visibility histogram, depth ranges, encoding-bound hits, and drift against a reference root.

Usage:
    python research/src/audit_pose3d_labels.py <root> [--ref <root>] [--sample 20000]
"""

import argparse
import random
from pathlib import Path

import numpy as np

from ultralytics.utils.pose3d import decode_z

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("--ref", type=Path, default=None, help="root with the same label filenames, for root-depth drift")
ap.add_argument("--sample", type=int, default=20000)
a = ap.parse_args()
files = sorted((a.root / "labels").glob("*.txt"))
random.seed(0)
files = random.sample(files, min(a.sample, len(files)))
rows, drift = [], []
for f in files:
    x = np.loadtxt(f, ndmin=2)[:, 5:].reshape(-1, 18, 4)
    rows.append(x)
    if a.ref and (a.ref / "labels" / f.name).exists():
        r = np.loadtxt(a.ref / "labels" / f.name, ndmin=2)[:, 5:].reshape(-1, 18, 4)
        if len(r) == len(x):
            drift.append(x[:, -1, 3] / np.maximum(r[:, -1, 3], 1e-9))
k = np.concatenate(rows)
zr, zroot = decode_z(k[:, :-1, 3], k[:, -1, 3])
v = k[:, :-1, 2]
print(f"files {len(files)}  persons {len(k)}")
print("joint vis:", dict(zip(*[x.tolist() for x in np.unique(v, return_counts=True)])),
      " root vis:", dict(zip(*[x.tolist() for x in np.unique(k[:, -1, 2], return_counts=True)])))
lab = zr[v == 2]
print(f"z_rel (vis2) p1 {np.percentile(lab, 1):.3f} p50 {np.median(lab):.3f} p99 {np.percentile(lab, 99):.3f} "
      f"absmax {np.abs(lab).max():.3f}  bound hits {(np.abs(lab) >= 1.999).sum()}")
print(f"z_root p1 {np.percentile(zroot, 1):.2f} p50 {np.median(zroot):.2f} p99 {np.percentile(zroot, 99):.2f} "
      f"max {zroot.max():.2f}  bound hits {(zroot >= 49.99).sum()}")
if drift:
    d = np.concatenate(drift)
    print(f"root depth new/ref: median {np.median(d):.4f}  p5 {np.percentile(d, 5):.4f}  p95 {np.percentile(d, 95):.4f}  n {len(d)}")
