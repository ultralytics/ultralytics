"""
Labelbox -> YOLO converter utilities.

Scope (v1):
- Modern Labelbox NDJSON export format
- Rectangular bounding boxes for detection tasks
- Simple YAML-based class mapping (one map per conversion)

Non-goals:
- Multi-map schemas
- Project management, uploads, or active learning
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import yaml

from ultralytics.utils import LOGGER, YAML


def load_class_map(path: str | Path) -> dict[str, int]:
    """Load class_name -> class_id mapping from a YAML file.

    YAML format:

    classes: class0:
        labelbox: ["labelbox.class0"]
    class1:
        labelbox: ["labelbox.class1"]
    class2:
        labelbox: ["labelbox.class2a", "labelbox.class2b"]
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    classes_cfg = cfg.get("classes", {})
    name_to_id: dict[str, int] = {}
    for idx, (class_name, _meta) in enumerate(classes_cfg.items()):
        name_to_id[class_name] = idx
    return name_to_id


def load_labelbox_mapping(path: str | Path) -> dict[str, int]:
    """Build a mapping from Labelbox label strings to YOLO class IDs.

    Uses the same YAML as load_class_map but flattens the labelbox entries.
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    classes_cfg = cfg.get("classes", {})
    mapping: dict[str, int] = {}
    for idx, (_class_name, meta) in enumerate(classes_cfg.items()):
        for lb_name in meta.get("labelbox", []):
            mapping[lb_name] = idx
    return mapping


def parse_labelbox_ndjson(path: str | Path) -> Iterable[dict]:
    """Yield each Labelbox data row (annotation) from an NDJSON export file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def convert_labelbox(
    labels_path: str | Path,
    images_dir: str | Path | None,
    save_dir: str | Path,
    class_map: str | Path,
):
    """Convert a Labelbox NDJSON export to a YOLO detection dataset (bounding boxes only).

    Notes:
        - v1 focuses on rectangular bounding boxes only.
        - Segmentation and polygons are intentionally left for a future extension.
        - The expected NDJSON structure is a simplified subset of Labelbox exports:

            {
              "externalId": "image_1.jpg",
              "imageWidth": 1280,
              "imageHeight": 720,
              "objects": [
                {
                  "className": "nho.pistol",
                  "bbox": {"left": 100, "top": 200, "width": 50, "height": 80}
                }
              ]
            }
    """
    if images_dir is None:
        raise ValueError("images_dir is required for convert_labelbox v1 (no automatic image downloading).")

    labels_path = Path(labels_path)
    images_dir = Path(images_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load class mappings from YAML
    lb_to_class = load_labelbox_mapping(class_map)
    name_to_id = load_class_map(class_map)

    images_out = save_dir / "images" / "train"
    labels_out = save_dir / "labels" / "train"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_boxes = 0

    for row in parse_labelbox_ndjson(labels_path):
        external_id = row.get("externalId")
        width = row.get("imageWidth")
        height = row.get("imageHeight")
        objects = row.get("objects", [])

        if not external_id or not width or not height:
            LOGGER.warning("Skipping row without required keys 'externalId', 'imageWidth', 'imageHeight'.")
            continue

        src_image = images_dir / external_id
        if not src_image.exists():
            LOGGER.warning(f"Image not found for Labelbox row, skipping: {src_image}")
            continue

        # Preserve subdirectory structure from externalId (e.g., "floor1/cam0.jpg")
        relative_path = Path(external_id)
        dst_image = images_out / relative_path
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        if not dst_image.exists():
            shutil.copy2(src_image, dst_image)

        label_lines: list[str] = []
        for obj in objects:
            class_name = obj.get("className")
            bbox = obj.get("bbox") or {}
            if not class_name or class_name not in lb_to_class:
                continue

            cls_id = lb_to_class[class_name]
            left = float(bbox.get("left", 0.0))
            top = float(bbox.get("top", 0.0))
            bw = float(bbox.get("width", 0.0))
            bh = float(bbox.get("height", 0.0))
            if bw <= 0 or bh <= 0:
                continue

            x_center = (left + bw / 2.0) / float(width)
            y_center = (top + bh / 2.0) / float(height)
            w_norm = bw / float(width)
            h_norm = bh / float(height)

            line = f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            label_lines.append(line)
            n_boxes += 1

        # Always write a label file (empty if no annotations) to match YOLO dataloader expectations
        label_file = labels_out / relative_path.with_suffix(".txt")
        label_file.parent.mkdir(parents=True, exist_ok=True)
        label_file.write_text("\n".join(label_lines) + "\n" if label_lines else "", encoding="utf-8")
        n_images += 1

    # Write dataset YAML (standard YOLO layout)
    names = {v: k for k, v in name_to_id.items()}
    data_yaml = {
        "path": str(save_dir),
        "train": "images/train",
        "names": names,
    }
    YAML.save(save_dir / "data.yaml", data_yaml)

    LOGGER.info(f"Labelbox data converted successfully: {n_images} images, {n_boxes} boxes. Saved to {save_dir}.")
    return save_dir / "data.yaml"
