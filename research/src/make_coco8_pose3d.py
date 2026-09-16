# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build the coco8-pose3d smoke dataset from coco8-pose.

The depth channel written here is SYNTHETIC: root depth is inferred from the person's box height under the
assumed pinhole camera, and per-joint relative depth is zero. It exists only to exercise the pose3d plumbing end
to end without waiting on SAM 3D Body access. Nothing measured on this dataset means anything about 3D accuracy.

Usage:
    python research/src/make_coco8_pose3d.py [--src <coco8-pose dir>] [--dst <output dir>]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from ultralytics.utils import SETTINGS
from ultralytics.utils.downloads import safe_download
from ultralytics.utils.pose3d import encode_z

PERSON_HEIGHT_M = 1.7  # a stand-in for the real thing; see module docstring
FOCAL_RATIO = 1.1  # must match focal_ratio in coco8-pose3d.yaml


def convert_label(line: str, img_h: int) -> str:
    """Convert one coco8-pose label row (17 x (x, y, v)) into a pose3d row (18 x (x, y, v, z))."""
    v = np.array(line.split(), dtype=np.float32)
    cls, box, kpts = v[0], v[1:5], v[5:].reshape(-1, 3)

    hips = kpts[[11, 12]]
    labelled = hips[hips[:, 2] > 0]
    root_xy = labelled[:, :2].mean(0) if len(labelled) else box[:2]
    root_v = 2.0 if len(labelled) else 1.0

    # Synthetic root depth from the pinhole relation z = f * H / h_px, clipped to a sane range.
    focal = FOCAL_RATIO * img_h
    z_root = float(np.clip(focal * PERSON_HEIGHT_M / max(box[3] * img_h, 1.0), 0.5, 40.0))
    z_rel = np.zeros(len(kpts), dtype=np.float32)
    z_rel_enc, z_root_enc = encode_z(z_rel, np.float32(z_root))

    out = [cls, *box]
    for (x, y, vis), z in zip(kpts, z_rel_enc):
        out += [x, y, vis, z]
    out += [root_xy[0], root_xy[1], root_v, float(z_root_enc)]
    return " ".join(f"{x:.6g}" for x in out)


def main(src: Path, dst: Path) -> None:
    """Copy coco8-pose images and rewrite its labels with synthetic depth channels."""
    import cv2

    if not src.exists():
        safe_download("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-pose.zip", dir=src.parent)
    if dst.exists():
        shutil.rmtree(dst)
    for split in ("train", "val"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in sorted((src / "images" / split).glob("*.jpg")):
            shutil.copy2(img, dst / "images" / split / img.name)
            h = cv2.imread(str(img)).shape[0]
            lb = (src / "labels" / split / f"{img.stem}.txt").read_text().strip().splitlines()
            rows = [convert_label(x, h) for x in lb if x.strip()]
            (dst / "labels" / split / f"{img.stem}.txt").write_text("\n".join(rows) + "\n")
    print(f"coco8-pose3d written to {dst}")


if __name__ == "__main__":
    root = Path(SETTINGS["datasets_dir"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=root / "coco8-pose")
    parser.add_argument("--dst", type=Path, default=root / "coco8-pose3d")
    args = parser.parse_args()
    main(args.src, args.dst)
