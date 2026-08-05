# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Project SODA-A scene annotations onto the existing 1024px tiles.

The tiles under ``SODA-A-split/images/`` were produced with ``split_dota`` defaults but only the crops were kept, so no
labels exist. Each tile name encodes its own window (``{scene}__{crop}__{x}___{y}``), so the windows are parsed back out
of the filenames and the polygons are assigned with the same IoF rule the cropper used. Images are never re-cut, and
every tile on disk is visited by construction.

Labels are written as normalized 8-coord OBB polygons, matching ``split_dota`` output and the sibling DOTAv1-split.
Horizontal boxes are one ``reshape(-1, 4, 2).min/max`` away at consume time, but the reverse is not, so orientation is
kept here.

Usage:
    python prep_soda_a_labels.py
    python prep_soda_a_labels.py --visualize 8
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ultralytics.data.split_dota import get_window_obj
from ultralytics.utils import YAML
from ultralytics.utils.plotting import Annotator, colors

ANNO_ROOT = Path("/data/shared-datasets/SODA-A-raw/anno-extracted")
SPLIT_ROOT = Path("/data/shared-datasets/SODA-A-split")
IGNORE = "ignore"  # SODA-A crowd/unclear regions, excluded from training labels
TILE_RE = re.compile(r"(.+)__(\d+)__(\d+)___(\d+)$")


def load_scene(json_path: Path) -> tuple[dict, dict[int, str]]:
    """Read one SODA-A scene json into the anno dict ``get_window_obj`` expects.

    Polygons that are not 4-point are reduced to their minimum-area rectangle, since SODA-A ships a small number of
    5-point annotations that would otherwise make the label array ragged.

    Args:
        json_path (Path): Path to a per-scene SODA-A annotation json.

    Returns:
        anno (dict): Keys ``label`` (N, 9) with normalized polygons and ``ori_size`` as (height, width).
        names (dict): Category id to name, with the ignore class removed.
    """
    d = json.loads(json_path.read_text())
    h, w = d["images"]["height"], d["images"]["width"]
    names = {c["id"]: c["name"] for c in d["categories"] if c["name"] != IGNORE}
    polys = []
    for a in d["annotations"]:
        if a["category_id"] not in names:
            continue
        p = np.array(a["poly"], dtype=np.float32)
        if p.size != 8:
            p = cv2.boxPoints(cv2.minAreaRect(p.reshape(-1, 2)))
        polys.append([a["category_id"], *p.ravel()])
    label = np.array(polys, dtype=np.float32).reshape(-1, 9)
    label[:, 1::2] /= w  # get_window_obj re-multiplies by the scene size
    label[:, 2::2] /= h
    return {"label": label, "ori_size": (h, w)}, names


def project(anno_root: Path, split_root: Path, iof: float) -> dict[str, int]:
    """Write a label file for every tile on disk that contains at least one object.

    Args:
        anno_root (Path): Root of the per-scene annotation jsons.
        split_root (Path): Root containing ``images/<split>/`` tiles.
        iof (float): Intersection-over-foreground threshold for keeping an object in a window.

    Returns:
        (dict): Counters for scenes, tiles, labelled and background tiles, and boxes written.
    """
    scenes: dict[str, list] = defaultdict(list)
    for p in (split_root / "images").glob("*/*.jpg"):
        scene, crop, x, y = TILE_RE.match(p.stem).groups()
        scenes[scene].append((p, int(x), int(y), int(crop)))
    splits = sorted(d.name for d in (split_root / "images").iterdir() if d.is_dir())
    for split in splits:
        (split_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    annos = {p.stem: p for p in anno_root.rglob("*.json")}
    if missing := sorted(set(scenes) - set(annos)):
        raise FileNotFoundError(f"{len(missing)} tiled scenes have no annotation json, first: {missing[:5]}")

    stats = {"scenes": len(scenes), "tiles": sum(map(len, scenes.values())), "labelled": 0, "background": 0, "boxes": 0}
    names: dict[int, str] = {}
    for scene, tiles in sorted(scenes.items()):
        anno, scene_names = load_scene(annos[scene])
        names = names or scene_names
        assert scene_names == names, f"{scene} declares a different category block than the rest of the dataset"
        windows = np.array([[x, y, x + c, y + c] for _, x, y, c in tiles], dtype=np.int64)
        for (p, x0, y0, crop), ob in zip(tiles, get_window_obj(anno, windows, iof_thr=iof)):
            path = split_root / "labels" / p.parent.name / f"{p.stem}.txt"
            if not len(ob):
                path.unlink(missing_ok=True)  # keep output a pure function of the inputs across re-runs
                stats["background"] += 1
                continue
            ob[:, 1::2] = (ob[:, 1::2] - x0) / crop
            ob[:, 2::2] = (ob[:, 2::2] - y0) / crop
            path.write_text("".join(f"{int(o[0])} " + " ".join(f"{c:.6g}" for c in o[1:]) + "\n" for o in ob))
            stats["labelled"] += 1
            stats["boxes"] += len(ob)

    assert max(names) < len(names), f"class ids {sorted(names)} are not contiguous from 0"
    splits = {s: f"images/{s}" for s in splits}
    YAML.save(split_root / "data.yaml", {"path": str(split_root), **splits, "names": names})
    return stats


def visualize(split_root: Path, n: int) -> int:
    """Render projected polygons onto sample tiles from every split for visual inspection.

    Args:
        split_root (Path): Root containing ``images/`` and ``labels/``.
        n (int): Number of tiles to render per split.

    Returns:
        (int): Number of previews rendered.
    """
    out_dir = split_root / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    written = 0
    for split_dir in sorted((split_root / "labels").iterdir()):
        labels = sorted(split_dir.glob("*.txt"))
        for lb in rng.sample(labels, min(n, len(labels))):
            im = cv2.imread(str(split_root / "images" / split_dir.name / f"{lb.stem}.jpg"))
            rows = [r.split() for r in lb.read_text().splitlines() if r.strip()]
            a = Annotator(im, line_width=2)
            for r in rows:
                pts = np.array(r[1:], dtype=np.float32).reshape(4, 2) * [im.shape[1], im.shape[0]]
                a.box_label(pts.tolist(), label=r[0], color=colors(int(r[0]), True))
            a.text((8, 8), f"{lb.stem}  n={len(rows)}")
            cv2.imwrite(str(out_dir / f"{split_dir.name}__{lb.stem}.jpg"), a.result())
            written += 1
    return written


def main() -> None:
    """Project SODA-A annotations onto existing tiles and optionally render previews."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anno-root", type=Path, default=ANNO_ROOT)
    p.add_argument("--split-root", type=Path, default=SPLIT_ROOT)
    p.add_argument("--iof", type=float, default=0.7, help="keep an object with this fraction inside the window")
    p.add_argument("--visualize", type=int, default=0, help="render this many labelled tiles per split")
    a = p.parse_args()

    for k, v in project(a.anno_root, a.split_root, a.iof).items():
        print(f"{k:22s} {v}")
    if a.visualize:
        print(f"previews: {visualize(a.split_root, a.visualize)}")


if __name__ == "__main__":
    main()
