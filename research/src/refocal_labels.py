# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Re-key pseudo-label depths from the teacher's diagonal focal onto one declared convention.

SAM 3D Body sets focal = the image diagonal (measured f/diagonal = 1.0000, sd 0.0000). That is self-consistent
per image but not across images: `diagonal / max(w, h)` runs 1.147 to 1.414 with aspect ratio alone, so two
identically-framed people in a 16:9 and a square photo get depth targets 23% apart for no physical reason. The
network cannot see the original aspect once the image is letterboxed, so that variance is unlearnable noise
sitting directly on the depth target.

This rewrites every depth into `focal = ratio * max(w, h)`, the convention the validator already assumes, by
scaling depths by `ratio * max(w, h) / diagonal`. Depth scales with focal because the pixels are fixed: a
point at (X, Y, Z) under focal f lands where (X, Y, Z * f'/f) lands under f'. Image dimensions are read from
the file headers, so the teacher never runs again.

Usage:
    python research/src/refocal_labels.py --src <dataset root> --dst <new root> --ratio 1.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ultralytics.utils.pose3d import Z_REL_RANGE, Z_ROOT_MAX, decode_z, encode_z


def convert_split(src: Path, dst: Path, split: str, ratio: float) -> tuple[int, int, list[float]]:
    """Rewrite one split's labels into the declared convention. Returns (files, rows, factors)."""
    (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
    files = sorted((src / "labels" / split).glob("*.txt"))
    n_rows, factors, clipped = 0, [], 0
    for lb in files:
        img = src / "images" / split / f"{lb.stem}.jpg"
        if not img.exists():
            continue
        with Image.open(img) as im:  # header only, no decode
            w, h = im.size
        factor = ratio * max(w, h) / float(np.hypot(w, h))
        factors.append(factor)

        v = np.array([x.split() for x in lb.read_text().strip().splitlines()], dtype=np.float64)
        k = v[:, 5:].reshape(len(v), -1, 4)
        z_rel, z_root = decode_z(k[:, :-1, 3], k[:, -1, 3])
        enc_rel, enc_root = encode_z(z_rel * factor, z_root * factor)
        clipped += int((z_root * factor >= Z_ROOT_MAX).sum() + (np.abs(z_rel * factor) >= Z_REL_RANGE).sum())
        k[:, :-1, 3], k[:, -1, 3] = enc_rel, enc_root
        v[:, 5:] = k.reshape(len(v), -1)
        (dst / "labels" / split / lb.name).write_text("\n".join(" ".join(f"{x:.6g}" for x in row) for row in v) + "\n")
        n_rows += len(v)
    if clipped:
        print(f"  {split}: {clipped} depth values hit an encoding bound after rescaling")
    return len(files), n_rows, factors


def main(src: Path, dst: Path, ratio: float, splits: list[str]) -> None:
    """Rewrite every split and relink the images."""
    (dst / "images").mkdir(parents=True, exist_ok=True)
    for split in splits:
        if not (src / "labels" / split).exists():
            continue
        link = dst / "images" / split
        if not link.exists():
            link.symlink_to((src / "images" / split).resolve())
        n_files, n_rows, factors = convert_split(src, dst, split, ratio)
        f = np.array(factors)
        listing = "\n".join(f"./images/{split}/{p.stem}.jpg" for p in sorted((dst / "labels" / split).glob("*.txt")))
        (dst / f"{split}.txt").write_text(listing + "\n")
        print(
            f"{split}: {n_files} images, {n_rows} persons, depth scaled by "
            f"mean {f.mean():.4f} (min {f.min():.4f}, max {f.max():.4f})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("/home/rick/coco-pose3d"))
    parser.add_argument("--dst", type=Path, default=Path("/home/rick/coco-pose3d-refocal"))
    parser.add_argument("--ratio", type=float, default=1.2, help="declared focal as a multiple of max(w, h)")
    parser.add_argument("--splits", nargs="+", default=["train2017", "val2017"])
    main(**vars(parser.parse_args()))
