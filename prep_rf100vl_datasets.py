"""Download a curated subset of Roboflow100-VL datasets and convert them to YOLO format.

Per dataset:
    1. Skip if the final ``data.yaml`` already exists.
    2. Download COCO export of the latest version via the ``roboflow`` SDK.
    3. Strip Roboflow's class-0 placeholder and re-index annotations/categories.
    4. Convert each split to YOLO labels with ``ultralytics.data.converter.convert_coco``.
    5. Lay out ``images/{split}`` as symlinks and write ``data.yaml``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import YAML

DATASETS: list[tuple[str, str, str]] = [
    ("medical", "nih-xray-itazg-xeoi", "nih-xray"),
    ("medical", "crystal-clean-brain-tumors-mri-dataset-hzb2f-plsq", "crystal-clean-brain-tumors-mri-dataset"),
    ("defect", "wheel-defect-detection-e53jb-38chk-ytwg", "wheel-defect-detection"),
    ("defect", "l10ul502-6ann9-yumi", "l10ul502"),
    ("agri", "weeds4-evltl-grf8n-zfn7s-ecuv", "weeds4"),
    ("agri", "aerial-cows-kt2wd-3jxcj-uvfx", "aerial-cows"),
    ("agri", "pig-detection-kaimq-a8ret-abpd", "pig-detection"),
    ("underwater", "peixos-fish-eyltk-rpno", "peixos-fish"),
    ("commerce", "everdaynew-6ej0k-lyqxk-zzbi", "everdaynew"),
    ("aerial", "electric-pylon-detection-in-rsi-q6qra-psut", "electric-pylon-detection-in-rsi"),
    ("aerial", "zebrasatasturias-nzsnv-cqvl", "zebrasatasturias"),
    ("non-rgb", "infraredimageofpowerequipment-kt4us-zqnd", "infraredimageofpowerequipment"),
    ("non-rgb", "thermal-cheetah-my4dp-zvgwh-ooto", "thermal-cheetah"),
]

SPLITS = ("train", "valid", "test")


def get_clean_ann_data(data: dict) -> dict | None:
    """Drop the Roboflow class-0 placeholder and re-index categories and annotation IDs.

    Args:
        data (dict): Parsed COCO annotation dict.

    Returns:
        (dict | None): Cleaned COCO dict, or None when category 0 is not the Roboflow placeholder.
    """
    if not data.get("categories") or data["categories"][0].get("supercategory") != "none":
        return None
    cleaned = {}
    if data.get("info"):
        cleaned["info"] = data["info"]
    if data.get("licenses"):
        cleaned["licenses"] = data["licenses"]
    cleaned["categories"] = [
        {"id": c["id"] - 1, "name": c["name"], "supercategory": c["supercategory"]}
        for c in data["categories"]
        if c["id"] != 0
    ]
    cleaned["images"] = deepcopy(data["images"])
    cleaned["annotations"] = deepcopy(data["annotations"])
    if cleaned["annotations"]:
        shift = 1 - min(a["id"] for a in cleaned["annotations"])
        for ann in cleaned["annotations"]:
            ann["category_id"] = ann["category_id"] - 1
            ann["id"] = ann["id"] + shift
    return cleaned


def clean_coco_inplace(coco_root: Path) -> None:
    """Rewrite each split's ``_annotations.coco.json`` under ``coco_root`` after dropping class 0."""
    for split in SPLITS:
        ann_path = coco_root / split / "_annotations.coco.json"
        if not ann_path.exists():
            continue
        with ann_path.open() as f:
            data = json.load(f)
        cleaned = get_clean_ann_data(data)
        if cleaned is None:
            continue
        with ann_path.open("w") as f:
            json.dump(cleaned, f)


def download_coco(slug: str, dest: Path, api_key: str) -> None:
    """Download the latest version of ``slug`` from the ``rf100-vl`` workspace as COCO into ``dest``."""
    import roboflow

    rf = roboflow.Roboflow(api_key=api_key)
    project = rf.workspace("rf100-vl").project(slug)
    version = max(project.versions(), key=lambda v: v.id)
    dest.mkdir(parents=True, exist_ok=True)
    version.download(location=str(dest), model_format="coco", overwrite=True)


def convert_splits_to_yolo(coco_root: Path, labels_root: Path) -> None:
    """Convert all COCO splits under ``coco_root`` into YOLO ``labels/{split}/*.txt`` files."""
    from ultralytics.data.converter import convert_coco

    labels_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_in = Path(tmp) / "json"
        tmp_in.mkdir()
        present = []
        for split in SPLITS:
            src = coco_root / split / "_annotations.coco.json"
            if not src.exists():
                continue
            shutil.copy(src, tmp_in / f"{split}.json")
            present.append(split)
        if not present:
            raise RuntimeError(f"no COCO annotation files under {coco_root}")
        tmp_out = Path(tmp) / "yolo"
        convert_coco(
            labels_dir=str(tmp_in),
            save_dir=str(tmp_out),
            use_segments=False,
            use_keypoints=False,
            cls91to80=False,
        )
        for split in present:
            src_dir = tmp_out / "labels" / split
            dst_dir = labels_root / split
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.move(str(src_dir), str(dst_dir))


def symlink_images(coco_root: Path, images_root: Path) -> dict[str, int]:
    """Symlink images from each ``coco_root/{split}`` into ``images_root/{split}`` and return per-split counts."""
    counts = {}
    for split in SPLITS:
        src_dir = coco_root / split
        if not src_dir.is_dir():
            counts[split] = 0
            continue
        dst_dir = images_root / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for img in sorted(src_dir.iterdir()):
            if img.suffix.lower().lstrip(".") not in IMG_FORMATS:
                continue
            link = dst_dir / img.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(img.resolve())
            n += 1
        counts[split] = n
    return counts


def write_data_yaml(dataset_root: Path, names: dict[int, str]) -> None:
    """Write the ultralytics ``data.yaml`` for the prepared dataset."""
    YAML.save(
        dataset_root / "data.yaml",
        {
            "path": str(dataset_root.resolve()),
            "train": "images/train",
            "val": "images/valid",
            "test": "images/test",
            "names": names,
        },
    )


def load_names(coco_root: Path) -> dict[int, str]:
    """Return ``{id: name}`` for the first split that has a cleaned annotation file."""
    for split in SPLITS:
        ann_path = coco_root / split / "_annotations.coco.json"
        if not ann_path.exists():
            continue
        with ann_path.open() as f:
            data = json.load(f)
        cats = sorted(data["categories"], key=lambda c: c["id"])
        return {int(c["id"]): c["name"] for c in cats}
    raise RuntimeError(f"no annotations found under {coco_root}")


def prepare_one(slug: str, basename: str, root: Path, api_key: str) -> dict:
    """Run the full download/clean/convert/layout pipeline for a single dataset."""
    dataset_root = root / basename
    coco_root = dataset_root / "coco"
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"

    download_coco(slug, coco_root, api_key)
    clean_coco_inplace(coco_root)
    convert_splits_to_yolo(coco_root, labels_root)
    image_counts = symlink_images(coco_root, images_root)
    names = load_names(coco_root)
    write_data_yaml(dataset_root, names)
    return {"num_classes": len(names), **{f"num_{s}_images": image_counts.get(s, 0) for s in SPLITS}}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the prep script."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("/data/shared-datasets/rf100-vl"))
    parser.add_argument("--datasets", type=str, default=None, help="comma-separated basename filter")
    parser.add_argument("--api_key", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Prepare the curated RF100-VL subset under ``args.root``."""
    args = parse_args()
    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("provide --api_key or set ROBOFLOW_API_KEY")

    selected = DATASETS
    if args.datasets:
        wanted = {x.strip() for x in args.datasets.split(",") if x.strip()}
        unknown = wanted - {b for _, _, b in DATASETS}
        if unknown:
            raise SystemExit(f"unknown basenames: {sorted(unknown)}")
        selected = [d for d in DATASETS if d[2] in wanted]

    args.root.mkdir(parents=True, exist_ok=True)
    total = len(selected)
    summary: list[tuple[str, dict]] = []
    for i, (_, slug, basename) in enumerate(selected, 1):
        dataset_root = args.root / basename
        if (dataset_root / "data.yaml").exists():
            print(f"[{i}/{total}] {basename} ... SKIP (data.yaml exists)")
            cfg = YAML.load(dataset_root / "data.yaml")
            stats = {"num_classes": len(cfg.get("names") or {})}
            for split in SPLITS:
                d = dataset_root / "images" / split
                stats[f"num_{split}_images"] = sum(1 for _ in d.iterdir()) if d.is_dir() else 0
            summary.append((basename, stats))
            continue
        stats = prepare_one(slug, basename, args.root, api_key)
        print(f"[{i}/{total}] {basename} ... OK")
        summary.append((basename, stats))

    print("\nbasename, num_classes, num_train_images, num_val_images, num_test_images")
    for basename, s in summary:
        print(f"{basename}, {s['num_classes']}, {s['num_train_images']}, {s['num_valid_images']}, {s['num_test_images']}")


if __name__ == "__main__":
    main()
