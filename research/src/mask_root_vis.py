# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Mark the root joint unlabelled when it sits outside the image.

`pseudo_label_sam3d.py` now writes the root's visibility from the same in-frame test every other joint gets,
but the labels already on disk were written before that and carry visibility 2 for every root. The information
is still recoverable: an out-of-frame root was clipped to exactly 0.0 or 1.0 by the clip pass, and a root that
landed on a border pre-clip is a measure-zero coincidence, so the clip marker is the out-of-frame flag.

That one channel is the whole change. `Pose3DLoss` masks the root-depth term, the root's 2D location term and
its objectness target off `keypoints[..., 2] != 0`, so setting it to 0 stops all three from supervising a point
the network cannot see — no change to the loss, the head or the dataloader.

H8 is the evidence: root-depth AbsRel on 3DPW rose 0.0398 -> 0.0445 -> 0.0456 as the frame-edge images entered
training, monotone in the data and unmoved by doubling the schedule.

Usage:
    python research/src/mask_root_vis.py --src <dataset root> --dst <new root>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def convert_split(src: Path, dst: Path, split: str) -> tuple[int, int, int]:
    """Rewrite one split with the root visibility masked. Returns (files, rows, rows masked)."""
    (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
    files = sorted((src / "labels" / split).glob("*.txt"))
    n_rows = n_masked = 0
    for lb in files:
        v = np.array([x.split() for x in lb.read_text().strip().splitlines()], dtype=np.float64)
        k = v[:, 5:].reshape(len(v), -1, 4)
        rx, ry = k[:, -1, 0], k[:, -1, 1]
        outside = (rx <= 0) | (rx >= 1) | (ry <= 0) | (ry >= 1)
        k[outside, -1, 2] = 0.0
        v[:, 5:] = k.reshape(len(v), -1)
        (dst / "labels" / split / lb.name).write_text("\n".join(" ".join(f"{x:.6g}" for x in row) for row in v) + "\n")
        n_rows += len(v)
        n_masked += int(outside.sum())
    return len(files), n_rows, n_masked


def main(src: Path, dst: Path, splits: list[str]) -> None:
    """Rewrite every split and relink the images."""
    (dst / "images").mkdir(parents=True, exist_ok=True)
    for split in splits:
        if not (src / "labels" / split).exists():
            continue
        link = dst / "images" / split
        if not link.exists():
            link.symlink_to((src / "images" / split).resolve())
        n_files, n_rows, n_masked = convert_split(src, dst, split)
        listing = "\n".join(f"./images/{split}/{p.stem}.jpg" for p in sorted((dst / "labels" / split).glob("*.txt")))
        (dst / f"{split}.txt").write_text(listing + "\n")
        print(f"{split}: {n_files} images, {n_rows} persons, {n_masked} roots masked ({100 * n_masked / n_rows:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("/home/rick/coco-pose3d-refocal"))
    parser.add_argument("--dst", type=Path, default=Path("/home/rick/coco-pose3d-rootmask"))
    parser.add_argument("--splits", nargs="+", default=["train2017", "val2017"])
    main(**vars(parser.parse_args()))
