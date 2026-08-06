# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build the merged enterprise detection corpus from the 21 converted sources.

Each source keeps its own subdirectory and its class ids are shifted into a slice of one joined label space, so a
dataset stays recoverable at train time. That is what the quota sampler and the owning-slice cls loss need, and it also
keeps mosaic and copy_paste inside one source. Images are symlinked, labels are rewritten.

Ordering is by source name and frozen in the emitted data.yaml, since the offsets are only meaningful against it.

Usage:
    python prep_enterprise_corpus.py
    python prep_enterprise_corpus.py --sources real-colon camus --viz 8
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from prep_domain_det import ROOT
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import YAML

EXTRA = [Path("/data/shared-datasets/SODA-A-split"), Path("/data/shared-datasets/yoloe26_data/Objects365v1")]
OUT = ROOT / "_merged"
WORKERS = 32  # every image costs a few NFS round trips, so this is latency-bound rather than CPU-bound
SEED = 1337


def dataset_yaml(root: Path) -> Path | None:
    """Return a source's data yaml, which is not always named data.yaml."""
    return next(iter(sorted(root.glob("*.yaml"))), None)


def sources() -> list[Path]:
    """Return the corpus source roots, sorted by name so the class offsets are reproducible."""
    roots = [y.parent for y in ROOT.glob("*/data.yaml")] + [e for e in EXTRA if dataset_yaml(e)]
    return sorted(roots, key=lambda p: p.name)


def split_dirs(base: Path, key) -> list[Path]:
    """Resolve a data.yaml split value, which is one directory or a list of them."""
    return [d for d in (base / r for r in ([key] if isinstance(key, str) else key)) if d.is_dir()]


def names_of(y: dict) -> list[str]:
    """Return class names as an ordered list, accepting the dict or list yaml form."""
    n = y["names"]
    return [n[i] for i in sorted(n)] if isinstance(n, dict) else list(n)


def link_one(src: Path, lf: Path | None, img_dst: Path, lab_dst: Path, offset: int) -> tuple[int, int]:
    """Symlink one image and write its labels with class ids shifted by ``offset``.

    Rows are copied through as class plus whatever coordinates follow, so oriented boxes and segmentation polygons
    survive untouched. An image with no label rows is a background image and gets no label written.

    Returns:
        (int): Boxes written.
        (int): 1 if the image is background, else 0.
    """
    with contextlib.suppress(FileExistsError):
        os.symlink(src, img_dst / src.name)
    rows = [
        f"{int(f[0]) + offset} {' '.join(f[1:])}"
        for line in (lf.read_text().splitlines() if lf else [])
        if len(f := line.split()) >= 5
    ]
    if rows:
        (lab_dst / f"{src.stem}.txt").write_text("\n".join(rows) + "\n")
    return len(rows), int(not rows)


def link_split(src_dirs: list[Path], dst: Path, offset: int) -> tuple[int, int, int]:
    """Merge one split into ``dst``, shifting every class id by ``offset``.

    Returns:
        (int): Images linked.
        (int): Boxes written.
        (int): Background images.
    """
    img_dst, lab_dst = dst / "images", dst / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lab_dst.mkdir(parents=True, exist_ok=True)
    jobs = []
    for d in src_dirs:
        lab_dir = Path(str(d).replace("/images", "/labels"))
        # One listing per directory, rather than an exists() round trip per image
        stems = {n[:-4] for n in os.listdir(lab_dir) if n.endswith(".txt")} if lab_dir.is_dir() else set()
        res = d.resolve()
        jobs += [
            (res / e.name, lab_dir / f"{stem}.txt" if (stem := e.name.rsplit(".", 1)[0]) in stems else None)
            for e in os.scandir(d)
            if e.name.rsplit(".", 1)[-1].lower() in IMG_FORMATS
        ]
    boxes = bg = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for b, g in ex.map(lambda j: link_one(*j, img_dst, lab_dst, offset), jobs):
            boxes, bg = boxes + b, bg + g
    return len(jobs), boxes, bg


def visualize(dst: Path, names: list[str], n: int) -> None:
    """Draw boxes on a few merged images so the offset rewrite is checked against pixels, not counts."""
    out = dst / "preview"
    out.mkdir(exist_ok=True)
    rnd = random.Random(SEED)
    labs = sorted((dst / "train" / "labels").glob("*.txt"))
    for lf in rnd.sample(labs, min(n, len(labs))):
        img_dir = dst / "train" / "images"
        p = next((q for e in IMG_FORMATS if (q := img_dir / f"{lf.stem}.{e}").exists()), None)
        if p is None or (im := cv2.imread(str(p))) is None:
            continue
        h, w = im.shape[:2]
        for line in lf.read_text().splitlines():
            f = line.split()
            c, xy = int(f[0]), [float(v) for v in f[1:]]
            if len(xy) == 4:
                cx, cy, bw, bh = xy
                x1, y1, x2, y2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
            else:
                xs, ys = xy[0::2], xy[1::2]
                x1, y1, x2, y2 = min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(im, f"{c}:{names[c]}", (int(x1), max(12, int(y1) - 4)), 0, 0.5, (255, 0, 0), 1)
        cv2.imwrite(str(out / p.name), im)


def main() -> None:
    """Merge every source into OUT and write the joined data.yaml."""
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--out", type=Path, default=OUT)
    a.add_argument("--sources", nargs="*", help="restrict to these source names")
    a.add_argument("--viz", type=int, default=12, help="preview images per source")
    args = a.parse_args()

    srcs = sources()
    ymls = {r.name: YAML.load(dataset_yaml(r)) for r in srcs}
    offsets, names, offset = {}, [], 0
    for r in srcs:  # offsets always come from the full list, so a partial run stays consistent
        offsets[r.name] = offset
        n = names_of(ymls[r.name])
        names += [f"{r.name}/{c}" for c in n]
        offset += len(n)
    print(f"{len(offsets)} sources, {offset} classes")

    totals = [0, 0, 0]
    for r in srcs:
        if args.sources and r.name not in args.sources:
            continue
        y = ymls[r.name]
        base = Path(y.get("path", r))
        dst = args.out / r.name
        row = []
        for split in ("train", "val"):
            if split not in y:
                continue
            i, b, g = link_split(split_dirs(base, y[split]), dst / split, offsets[r.name])
            row.append(f"{split} {i:,} imgs {b:,} boxes {g:,} bg")
            if split == "train":
                totals = [totals[0] + i, totals[1] + b, totals[2] + g]
        print(f"  {r.name:22} off={offsets[r.name]:4}  " + " | ".join(row), flush=True)
        if args.viz:
            visualize(dst, names, args.viz)

    YAML.save(
        args.out / "data.yaml",
        {
            "path": str(args.out),
            "train": [f"{r.name}/train/images" for r in srcs],
            "val": [f"{r.name}/val/images" for r in srcs],
            "names": dict(enumerate(names)),
            "offsets": offsets,  # source name to its first class id, the slice bounds the cls mask needs
        },
    )
    print(f"train totals: {totals[0]:,} images, {totals[1]:,} boxes, {totals[2]:,} background")
    print(f"wrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
