# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert the RICO semantic annotations into a YOLO detection dataset.

``semantic_annotations.zip`` holds one json per UI, a view tree whose nodes carry ``bounds`` and a ``componentLabel``
drawn from a 25-term vocabulary. Screenshots live in ``unique_uis.tar.gz`` under ``combined/``. Bounds are expressed in
a fixed 1440x2560 device space regardless of the jpg scale on disk, which varies between 1080x1920 and 540x960 at a
constant 0.5625 aspect, so boxes normalize against the device space rather than the image.

Every labelled node is kept, including nested ones, so a Toolbar and the Icons inside it are both emitted. That is the
multi-scale supervision the source provides, not a bug. The val split exists so the yaml is loadable and spot-checkable,
and it splits by screen rather than by app, so RICO val numbers are not a reportable metric.

Usage:
    python prep_rico_labels.py
    python prep_rico_labels.py --visualize 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tarfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

from ultralytics.utils import YAML
from ultralytics.utils.ops import xywhn2xyxy
from ultralytics.utils.plotting import Annotator, colors

RICO_ROOT = Path("/data/shared-datasets/domain-det/rico")
DEVICE_W, DEVICE_H = 1440, 2560  # RICO bounds are always in this space, verified against the shipped screenshots
VAL_PERCENT = 2
NAMES = [
    "Advertisement", "Background Image", "Bottom Navigation", "Button Bar", "Card",
    "Checkbox", "Date Picker", "Drawer", "Icon", "Image",
    "Input", "List Item", "Map View", "Modal", "Multi-Tab",
    "Number Stepper", "On/Off Switch", "Pager Indicator", "Radio Button", "Slider",
    "Text", "Text Button", "Toolbar", "Video", "Web View",
]  # fmt: skip


def walk(node: dict):
    """Yield every node of a RICO view tree depth-first.

    Args:
        node (dict): Root node of the tree.

    Yields:
        (dict): Each node, root included.
    """
    yield node
    for child in node.get("children") or []:
        yield from walk(child)


def split_of(uid: str) -> str:
    """Assign a UI to a split from its id alone, so a screen can never move between runs.

    Args:
        uid (str): RICO screen id.

    Returns:
        (str): Either ``train`` or ``val``.
    """
    return "val" if int(hashlib.md5(uid.encode()).hexdigest(), 16) % 100 < VAL_PERCENT else "train"


def read_annotations(zip_path: Path) -> tuple[dict[str, list], dict[str, int]]:
    """Read every UI json and turn its labelled nodes into normalized boxes.

    Args:
        zip_path (Path): Path to ``semantic_annotations.zip``.

    Returns:
        rows (dict): UI id to a list of (class index, cx, cy, w, h) with coordinates normalized to the device space.
        stats (dict): Counters for UIs read, boxes kept, UIs clipped, and boxes dropped as degenerate or duplicate.
    """
    index = {n: i for i, n in enumerate(NAMES)}
    stats = dict.fromkeys(("uis", "boxes", "clipped_uis", "degenerate", "duplicate"), 0)
    rows: dict[str, list] = {}
    with zipfile.ZipFile(zip_path) as z:
        for entry in z.namelist():
            if not entry.endswith(".json"):
                continue
            boxes, seen, clipped = [], set(), False
            for node in walk(json.loads(z.read(entry))):
                label = node.get("componentLabel")
                if label is None:
                    continue
                assert label in index, f"{entry} carries component label {label!r}, outside the known vocabulary"
                bounds = node["bounds"]
                x1, y1 = max(bounds[0], 0), max(bounds[1], 0)
                x2, y2 = min(bounds[2], DEVICE_W), min(bounds[3], DEVICE_H)
                clipped |= [x1, y1, x2, y2] != bounds
                if x2 <= x1 or y2 <= y1:
                    stats["degenerate"] += 1
                    continue
                w, h = (x2 - x1) / DEVICE_W, (y2 - y1) / DEVICE_H
                box = (index[label], (x1 + x2) / 2 / DEVICE_W, (y1 + y2) / 2 / DEVICE_H, w, h)
                if box in seen:  # an exact repeat of a box already emitted for this UI
                    stats["duplicate"] += 1
                    continue
                seen.add(box)
                boxes.append(box)
            rows[entry.rsplit("/", 1)[-1][:-5]] = boxes
            stats["uis"] += 1
            stats["boxes"] += len(boxes)
            stats["clipped_uis"] += clipped
    return rows, stats


def write_dataset(root: Path, rows: dict[str, list]) -> dict[str, int]:
    """Write label files, extract the matching screenshots, and emit the dataset yaml.

    Args:
        root (Path): RICO root holding the two source archives, also the output root.
        rows (dict): UI id to its normalized boxes.

    Returns:
        (dict): Counters for labelled UIs per split, empty UIs, images written or already present, images with no
            annotation, and annotated UIs whose screenshot never arrived.
    """
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    keys = ("labelled_train", "labelled_val", "empty", "image_written", "image_present", "image_without_anno")
    stats = dict.fromkeys(keys, 0)
    for uid, boxes in rows.items():
        path = root / "labels" / split_of(uid) / f"{uid}.txt"
        if not boxes:
            path.unlink(missing_ok=True)  # keep output a pure function of the inputs across re-runs
            stats["empty"] += 1
            continue
        path.write_text("".join(f"{b[0]} {b[1]:.6g} {b[2]:.6g} {b[3]:.6g} {b[4]:.6g}\n" for b in boxes))
        stats[f"labelled_{split_of(uid)}"] += 1

    with tarfile.open(root / "unique_uis.tar.gz", "r|gz") as t:  # streaming, the archive is 6.4GB
        for member in t:
            if not member.name.endswith(".jpg"):
                continue
            uid = member.name.rsplit("/", 1)[-1][:-4]
            if uid not in rows:
                stats["image_without_anno"] += 1
                continue
            dst = root / "images" / split_of(uid) / f"{uid}.jpg"
            if dst.exists():
                stats["image_present"] += 1
                continue
            dst.write_bytes(t.extractfile(member).read())
            stats["image_written"] += 1
    stats["anno_without_image"] = len(rows) - stats["image_written"] - stats["image_present"]

    splits = {s: f"images/{s}" for s in ("train", "val")}
    YAML.save(root / "data.yaml", {"path": str(root), **splits, "names": dict(enumerate(NAMES))})
    return stats


def visualize(root: Path, n: int) -> int:
    """Render converted boxes onto sample screenshots from every split for visual inspection.

    Args:
        root (Path): Dataset root containing ``images/`` and ``labels/``.
        n (int): Number of screenshots to render per split.

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
            im = cv2.imread(str(root / "images" / split_dir.name / f"{lb.stem}.jpg"))
            rows = [x.split() for x in lb.read_text().splitlines() if x.strip()]
            a = Annotator(im, line_width=2)
            for r in rows:
                xywhn = np.array([[float(v) for v in r[1:]]], dtype=np.float32)
                a.box_label(
                    xywhn2xyxy(xywhn, w=im.shape[1], h=im.shape[0])[0], NAMES[int(r[0])], colors(int(r[0]), True)
                )
            a.text((8, 8), f"{lb.stem}  n={len(rows)}")
            cv2.imwrite(str(out_dir / f"{split_dir.name}__{lb.stem}.jpg"), a.result())
            written += 1
    return written


def main() -> None:
    """Convert RICO semantic annotations to YOLO format and optionally render previews."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=RICO_ROOT)
    p.add_argument("--visualize", type=int, default=0, help="render this many labelled screenshots per split")
    a = p.parse_args()

    rows, stats = read_annotations(a.root / "semantic_annotations.zip")
    for k, v in sorted(stats.items()):
        print(f"{k:22s} {v}")
    for k, v in sorted(write_dataset(a.root, rows).items()):
        print(f"{k:22s} {v}")
    if a.visualize:
        print(f"previews: {visualize(a.root, a.visualize)}")


if __name__ == "__main__":
    main()
