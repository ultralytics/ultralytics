# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Measure what focal length SAM 3D Body actually assumed when it produced our labels.

Metric depth is only defined together with a focal length, and the pipeline currently carries three different
ones: whatever the teacher assumed, the 1.1 declared in coco-pose3d.yaml, and 3DPW's measured 1.0239. This
prints the teacher's own `focal_length` against each image's dimensions.

The answer decides the size of the fix. If `focal / max(w, h)` is constant, every pseudo-label is already in
one self-consistent convention, the declared ratio is simply the wrong number, and the correction is
evaluation-side arithmetic with no retraining. If it varies per image, the labels carry per-image scale noise
and have to be re-keyed, which means a pass over the teacher again.

Usage:
    python research/src/teacher_focal_probe.py --images <dir> --labels <dir> --n 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(images: Path, labels: Path, n: int, device: str) -> None:
    """Run the teacher on a spread of image shapes and report its focal convention."""
    import cv2
    from snowpose.model import BodyModel

    model = BodyModel(device=device)
    files = sorted(labels.glob("*.txt"))
    rows = []
    for lb in files[:: max(1, len(files) // n)][:n]:
        img = images / f"{lb.stem}.jpg"
        if not img.exists():
            continue
        h, w = cv2.imread(str(img)).shape[:2]
        first = lb.read_text().strip().splitlines()[0].split()
        v = np.array(first, dtype=np.float32)
        cx, cy, bw, bh = v[1:5]
        box = np.array([[(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]], np.float32)
        out = model.est.process_one_image(str(img), bboxes=box)
        if not out:
            continue
        f = float(out[0]["focal_length"])
        rows.append((w, h, f, f / max(w, h), f / np.hypot(w, h)))

    a = np.array(rows)
    print(f"{'w':>6}{'h':>6}{'focal':>9}{'f/max':>9}{'f/diag':>9}")
    for w, h, f, r_max, r_diag in a[:12]:
        print(f"{int(w):6d}{int(h):6d}{f:9.1f}{r_max:9.4f}{r_diag:9.4f}")
    for name, col in (("f/max(w,h)", a[:, 3]), ("f/diagonal", a[:, 4])):
        print(
            f"{name:12s} mean {col.mean():.4f}  std {col.std():.4f}  min {col.min():.4f}  max {col.max():.4f}  "
            f"spread {100 * col.std() / col.mean():.2f}%"
        )
    print(f"\nn={len(a)} images, {len(set(map(tuple, a[:, :2].tolist())))} distinct shapes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("/data/shared-datasets/coco-pose/images/val2017"))
    parser.add_argument("--labels", type=Path, default=Path("/data/shared-datasets/coco-pose/labels/val2017"))
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    main(**vars(parser.parse_args()))
