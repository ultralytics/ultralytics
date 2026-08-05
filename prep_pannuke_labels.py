# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Decode the PanNuke folds into a YOLO detection dataset.

PanNuke ships three fold zips of float64 ``.npy`` arrays, 37GB uncompressed. ``images.npy`` is (N, 256, 256, 3) with
0-255 values, and ``masks.npy`` is (N, 256, 256, 6) where channels 0-4 hold per-nucleus integer instance ids for the
five classes and channel 5 is background. Boxes come from the extent of each instance id, so nuclei that touch stay
separate, which a binary mask plus connected components would not achieve. Ids are only ever compared within a channel,
so an id reused across two classes cannot merge two nuclei.

Each fold is extracted to a temporary directory, memory-mapped, decoded, and discarded before the next, so peak disk
stays near one fold rather than all three. The zips are never touched. Every patch trains except a deterministic 2%
slice that exists so the yaml is loadable and spot-checkable, which makes PanNuke val numbers non-reportable, and in
any case a fixed split is not the 3-fold cross-validation the source is benchmarked under.

Usage:
    python prep_pannuke_labels.py
    python prep_pannuke_labels.py --visualize 6
"""

from __future__ import annotations

import argparse
import random
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from ultralytics.utils import YAML
from ultralytics.utils.downloads import unzip_file
from ultralytics.utils.ops import xywhn2xyxy
from ultralytics.utils.plotting import Annotator, colors

PANNUKE_ROOT = Path("/data/shared-datasets/domain-det/pannuke")
NAMES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
EXPECTED_PATCHES = 7901
EXPECTED_NUCLEI = 205343  # Gamper et al., PanNuke paper. Printed as a delta, not asserted, since it moves with the rule
VAL_EVERY = 50  # every 50th patch to val, purely so the yaml loads
SPARSE_FILL = 0.2  # a box this empty suggests one id covering two disjoint nuclei


def instance_boxes(mask: np.ndarray) -> tuple[list, int]:
    """Turn one patch's instance-id mask into normalized YOLO boxes.

    Args:
        mask (np.ndarray): Array of shape (H, W, 6) whose first five channels hold per-nucleus integer instance ids.

    Returns:
        boxes (list): One (class index, cx, cy, w, h) per nucleus, normalized to the patch.
        sparse (int): Boxes whose pixels fill less than ``SPARSE_FILL`` of their extent.
    """
    h, w = mask.shape[:2]
    boxes, sparse = [], 0
    for c in range(len(NAMES)):
        chan = mask[..., c]
        for i in np.unique(chan):
            if i == 0:
                continue
            ys, xs = np.nonzero(chan == i)
            x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            sparse += len(xs) / ((x2 - x1) * (y2 - y1)) < SPARSE_FILL
            boxes.append((c, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
    return boxes, sparse


def decode_fold(root: Path, fold: int) -> Counter:
    """Extract one fold, write its patches as PNG and its nuclei as YOLO labels, then discard the extracted arrays.

    Args:
        root (Path): PanNuke root holding the fold zips, also the output root.
        fold (int): Fold number to decode.

    Returns:
        (Counter): Counters for patches written, nuclei written, patches with no nucleus, and suspiciously sparse boxes.
    """
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = Counter({k: 0 for k in ("patches", "nuclei", "empty", "sparse")})
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(unzip_file(root / f"fold_{fold}.zip", tmp))
        im_arr = np.load(next(work.rglob("images.npy")), mmap_mode="r")
        mk_arr = np.load(next(work.rglob("masks.npy")), mmap_mode="r")
        assert len(im_arr) == len(mk_arr), f"fold {fold} has {len(im_arr)} images against {len(mk_arr)} masks"
        assert im_arr.shape[1:] == (256, 256, 3), f"fold {fold} images are {im_arr.shape[1:]}, expected (256, 256, 3)"
        assert mk_arr.shape[1:] == (256, 256, 6), f"fold {fold} masks are {mk_arr.shape[1:]}, expected (256, 256, 6)"
        assert im_arr[0].max() > 1, f"fold {fold} pixels peak at {im_arr[0].max()}, expected a 0-255 range"

        for i in range(len(im_arr)):
            stem, split = f"fold{fold}_{i:05d}", "val" if i % VAL_EVERY == 0 else "train"
            # PanNuke arrays are RGB, cv2.imwrite expects BGR
            ok = cv2.imwrite(str(root / "images" / split / f"{stem}.png"), im_arr[i][..., ::-1].astype(np.uint8))
            assert ok, f"failed to write {stem}.png, the corpus would silently shrink"
            boxes, sparse = instance_boxes(np.asarray(mk_arr[i]))
            path = root / "labels" / split / f"{stem}.txt"
            if boxes:
                path.write_text("".join(f"{b[0]} {b[1]:.6g} {b[2]:.6g} {b[3]:.6g} {b[4]:.6g}\n" for b in boxes))
            else:
                path.unlink(missing_ok=True)  # keep output a pure function of the inputs across re-runs
                stats["empty"] += 1
            stats.update(patches=1, nuclei=len(boxes), sparse=sparse)
    return stats


def visualize(root: Path, n: int) -> int:
    """Render decoded boxes onto sample patches from every split for visual inspection.

    Args:
        root (Path): Dataset root containing ``images/`` and ``labels/``.
        n (int): Number of patches to render per split.

    Returns:
        (int): Number of previews rendered.
    """
    out_dir = root / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    written = 0
    for split_dir in sorted((root / "labels").iterdir()):
        labels = sorted(split_dir.glob("*.txt"))
        for lb in rng.sample(labels, min(n, len(labels))):
            im = cv2.imread(str(root / "images" / split_dir.name / f"{lb.stem}.png"))
            im = cv2.resize(im, (768, 768), interpolation=cv2.INTER_NEAREST)  # 256px is too small to inspect
            rows = [x.split() for x in lb.read_text().splitlines() if x.strip()]
            a = Annotator(im, line_width=1)
            for r in rows:
                xywhn = np.array([[float(v) for v in r[1:]]], dtype=np.float32)
                a.box_label(xywhn2xyxy(xywhn, w=im.shape[1], h=im.shape[0])[0], color=colors(int(r[0]), True))
            counts = Counter(NAMES[int(r[0])] for r in rows)
            a.text((8, 8), f"{lb.stem}  n={len(rows)}  " + " ".join(f"{k}:{v}" for k, v in counts.most_common()))
            cv2.imwrite(str(out_dir / f"{split_dir.name}__{lb.stem}.jpg"), a.result())
            written += 1
    return written


def main() -> None:
    """Decode every PanNuke fold to YOLO format and optionally render previews."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=PANNUKE_ROOT)
    p.add_argument("--visualize", type=int, default=0, help="render this many labelled patches per split")
    a = p.parse_args()

    total = Counter()
    for fold in (1, 2, 3):
        stats = decode_fold(a.root, fold)
        print(f"fold {fold}: " + "  ".join(f"{k} {v}" for k, v in stats.items()))
        total.update(stats)
    for k, v in total.items():
        print(f"{k:22s} {v}")
    print(f"{'patch delta':22s} {total['patches'] - EXPECTED_PATCHES:+d} vs {EXPECTED_PATCHES}")
    print(f"{'nuclei delta':22s} {total['nuclei'] - EXPECTED_NUCLEI:+d} vs {EXPECTED_NUCLEI} reported")

    splits = {s: f"images/{s}" for s in ("train", "val")}
    YAML.save(a.root / "data.yaml", {"path": str(a.root), **splits, "names": dict(enumerate(NAMES))})
    if a.visualize:
        print(f"previews: {visualize(a.root, a.visualize)}")


if __name__ == "__main__":
    main()
