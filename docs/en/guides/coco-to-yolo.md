---
comments: true
description: Learn how to convert COCO JSON annotations to YOLO format for object detection, instance segmentation, and pose estimation training. Complete guide with step-by-step examples, common pitfalls, and class ID mapping for custom datasets.
keywords: COCO to YOLO, convert COCO JSON to YOLO, COCO JSON format, YOLO annotation format, convert_coco, COCO dataset training, train YOLO on COCO, object detection dataset, instance segmentation dataset, pose estimation dataset, dataset conversion, annotation format, cls91to80, category_id, bounding box format, YOLO training data
---

# How to Convert COCO Annotations to YOLO Format

Training [Ultralytics YOLO](https://www.ultralytics.com/) models requires annotations in YOLO format, but many popular [annotation](https://www.ultralytics.com/glossary/data-labeling) tools export in [COCO JSON](https://cocodataset.org/#format-data) format instead. This guide shows you how to convert your COCO annotations to YOLO format and start training [object detection](https://www.ultralytics.com/glossary/object-detection), [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation), and [pose estimation](https://www.ultralytics.com/glossary/pose-estimation) models.

## Why Convert from COCO to YOLO?

The COCO JSON format stores all annotations in a single file, while [YOLO](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format) uses one text file per image with normalized coordinates. Converting is necessary because:

- **YOLO models require `.txt` label files** with one file per image, containing `class x_center y_center width height` in normalized coordinates.
- **COCO JSON uses pixel coordinates** in `[x_min, y_min, width, height]` format with a single JSON file for all images.
- **Class IDs differ** — COCO uses arbitrary `category_id` values, while YOLO requires zero-indexed class IDs.

| Feature          | COCO JSON                                 | YOLO TXT                                                |
| ---------------- | ----------------------------------------- | ------------------------------------------------------- |
| **Structure**    | Single JSON file for all images           | One `.txt` file per image                               |
| **Bbox format**  | `[x_min, y_min, width, height]` in pixels | `class x_center y_center width height` normalized (0-1) |
| **Class IDs**    | `category_id` (can start from any number) | Zero-indexed (starts from 0)                            |
| **Segmentation** | Polygon arrays in `segmentation` field    | Polygon coordinates after class ID                      |
| **Keypoints**    | `[x, y, visibility, ...]` in pixels       | `[x, y, visibility, ...]` normalized                    |

## Quick Start

The fastest way to convert COCO annotations and start training:

```python
from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="path/to/annotations/",  # directory containing your JSON files
    save_dir="path/to/output/",  # where to save converted labels
    cls91to80=False,  # IMPORTANT: set False for custom datasets
)
```

After conversion, [organize your directory structure](#3-organize-directory-structure), [create a dataset.yaml](#4-create-datasetyaml), and [start training](#5-train-your-yolo-model). See the full [step-by-step guide](#step-by-step-conversion-guide) below.

!!! warning "Custom datasets: always use `cls91to80=False`"

    The `cls91to80=True` default is designed **only** for the standard [COCO dataset](../datasets/detect/coco.md) with 80 object classes, which maps 91 non-contiguous category IDs to 80 contiguous class IDs. For any custom dataset, you **must** set `cls91to80=False` — otherwise your class IDs will be silently mapped incorrectly and your model will learn wrong classes.

## Step-by-Step Conversion Guide

### 1. Prepare Your COCO Dataset

A typical COCO-format dataset exported from annotation tools has the following structure:

```
my_dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── val/
│       ├── img_100.jpg
│       └── ...
└── annotations/
    ├── instances_train.json
    └── instances_val.json
```

Each JSON file follows the [COCO data format](https://cocodataset.org/#format-data) specification with three required fields — `images`, `annotations`, and `categories`:

```json
{
    "images": [{ "id": 1, "file_name": "img_001.jpg", "width": 640, "height": 480 }],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 50, 200, 150],
            "area": 30000,
            "iscrowd": 0
        }
    ],
    "categories": [
        { "id": 1, "name": "helmet" },
        { "id": 2, "name": "vest" }
    ]
}
```

### 2. Convert Annotations

Use the [`convert_coco()`](../reference/data/converter.md#ultralytics.data.converter.convert_coco) function to convert your COCO JSON annotations to YOLO `.txt` format:

!!! example "Convert COCO to YOLO format"

    === "Object Detection"

        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco(
            labels_dir="my_dataset/annotations/",
            save_dir="my_dataset/converted/",
            cls91to80=False,
        )
        ```

    === "Instance Segmentation"

        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco(
            labels_dir="my_dataset/annotations/",
            save_dir="my_dataset/converted/",
            use_segments=True,
            cls91to80=False,
        )
        ```

    === "Pose Estimation"

        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco(
            labels_dir="my_dataset/annotations/",
            save_dir="my_dataset/converted/",
            use_keypoints=True,
            cls91to80=False,
        )
        ```

### 3. Organize Directory Structure

After conversion, label files need to be placed alongside your images. YOLO expects a `labels/` directory that mirrors the `images/` directory:

```python
import shutil
from pathlib import Path

# Paths
converted_dir = Path("my_dataset/converted/labels")
dataset_dir = Path("my_dataset")

# Move labels next to images for each split
for split in ["train", "val"]:
    src = converted_dir / split  # convert_coco strips "instances_" prefix from JSON filename
    dst = dataset_dir / "labels" / split
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.txt"):
        shutil.move(str(f), str(dst / f.name))
```

Your final [dataset structure](../datasets/detect/index.md#ultralytics-yolo-format) should look like:

```
my_dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── val/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img_001.txt
│   │   └── ...
│   └── val/
│       └── ...
└── dataset.yaml
```

### 4. Create dataset.yaml

Create a `dataset.yaml` configuration file that maps your COCO categories to YOLO class names. This file tells YOLO where your data is and what classes to detect:

```python
import json
from pathlib import Path

import yaml

# Read categories from your COCO JSON
with open("my_dataset/annotations/instances_train.json") as f:
    coco = json.load(f)

# Build class names matching convert_coco output (category_id - 1)
categories = sorted(coco["categories"], key=lambda x: x["id"])
names = {cat["id"] - 1: cat["name"] for cat in categories}
# NOTE: convert_coco maps class IDs as category_id - 1, so category_id must
# start from 1. If your categories start from 0, add 1 to each ID first.

# Create dataset.yaml
dataset = {
    "path": str(Path("my_dataset").resolve()),
    "train": "images/train",
    "val": "images/val",
    "names": names,
}

with open("my_dataset/dataset.yaml", "w") as f:
    yaml.dump(dataset, f, default_flow_style=False)
```

The resulting YAML file:

```yaml
path: /absolute/path/to/my_dataset
train: images/train
val: images/val
names:
    0: helmet
    1: vest
```

For more details on the dataset YAML format, see the [dataset configuration guide](../datasets/detect/index.md).

### 5. Train Your YOLO Model

With your converted dataset ready, train a YOLO model:

!!! example "Train on converted COCO data"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")  # load a pretrained model
        results = model.train(data="my_dataset/dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train model=yolo26n.pt data=my_dataset/dataset.yaml epochs=100 imgsz=640
        ```

For training tips and best practices, see the [model training guide](model-training-tips.md).

### 6. Verify Your Conversion

Before training, spot-check a few label files to confirm class IDs and coordinates are correct:

```python
from pathlib import Path

label_file = Path("my_dataset/labels/train/img_001.txt")
for line in label_file.read_text().strip().splitlines():
    parts = line.split()
    cls_id = int(parts[0])
    coords = [float(v) for v in parts[1:5]]
    assert cls_id >= 0, f"Negative class ID {cls_id} — category_id in your JSON may start from 0"
    assert all(0 <= v <= 1 for v in coords), f"Coordinates out of [0, 1] range: {coords}"
```

!!! tip

    If you see negative class IDs, your COCO JSON likely uses `category_id` starting from 0. Add 1 to all `category_id` values in your JSON before running `convert_coco()`, since it maps class IDs as `category_id - 1`.

## Troubleshooting Common Issues

### Wrong Class IDs After Conversion

If your model trains but detects wrong object classes, you're likely using `cls91to80=True` (default) on a custom dataset. This maps your `category_id` values through the COCO 91-to-80 lookup table, which is only correct for the standard [COCO dataset](../datasets/detect/coco.md).

**Solution**: Always use `cls91to80=False` for custom datasets.

### No Labels Found During Training

If training shows `WARNING: No labels found` or `0 images, N backgrounds`, your label files are not in the expected directory. `convert_coco()` saves labels to a separate output directory (e.g., `save_dir/labels/train/`), but YOLO expects `labels/` parallel to `images/` inside your dataset directory.

**Solution**: Move label files to match the expected [directory structure](#3-organize-directory-structure). Make sure `labels/train/` is a sibling of `images/train/`.

### KeyError During Conversion

If you get `KeyError: 'bbox'` or similar errors when running `convert_coco()`, your `labels_dir` likely contains non-instance JSON files (e.g., `captions_train2017.json`) that have a different annotation structure.

**Solution**: Only place instance annotation JSON files (e.g., `instances_train2017.json`) in the `labels_dir`.

### Empty Label Files After Conversion

If conversion completes but `.txt` files are empty or missing, all annotations may have `iscrowd: 1` (common with [SAM](../models/sam.md)-generated masks), or [bounding boxes](https://www.ultralytics.com/glossary/bounding-box) have zero width or height.

**Solution**: Inspect your JSON annotations for `iscrowd` values. If using SAM masks, preprocess the JSON to set `iscrowd: 0`.

### Class ID Gaps in Converted Labels

If class IDs in label files are non-contiguous (e.g., 0, 4, 9 instead of 0, 1, 2), your annotation tool uses non-contiguous `category_id` values.

**Solution**: Verify the class IDs in your `.txt` files match the `names` dictionary in `dataset.yaml`. Remap IDs to contiguous values if needed.

For full API details and parameter descriptions, see the [`convert_coco` API reference](../reference/data/converter.md#ultralytics.data.converter.convert_coco).

## FAQ

### How do I convert COCO JSON annotations to YOLO format?

Use the `convert_coco()` function from Ultralytics to convert COCO JSON annotations to YOLO `.txt` format. Set `cls91to80=False` for custom datasets:

```python
from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="path/to/annotations/", save_dir="output/", cls91to80=False)
```

After conversion, reorganize your label files so `labels/` mirrors the `images/` directory, then create a `dataset.yaml` file. See the [step-by-step guide](#step-by-step-conversion-guide) for the complete workflow.

### Why does YOLO training show "No labels found" after COCO conversion?

This happens because `convert_coco()` saves labels to a subdirectory inside `save_dir/labels/` (e.g., `save_dir/labels/train/`) rather than directly into your dataset's `labels/train/` alongside `images/train/`. YOLO expects labels to sit parallel to images — for example, `images/train/img.jpg` needs `labels/train/img.txt`. Move your converted labels to match this structure. See [fixing the directory structure](#3-organize-directory-structure).

### What does `cls91to80` do in `convert_coco()`?

The `cls91to80` parameter controls how COCO `category_id` values are mapped to YOLO class IDs. When `True` (default), it uses a lookup table designed for the standard [COCO dataset](../datasets/detect/coco.md), which has 80 classes with non-contiguous IDs (1-90). For **custom datasets**, always set `cls91to80=False` — this simply subtracts 1 from each `category_id` to create zero-indexed class IDs.

### Can I train YOLO directly on COCO JSON without converting?

Not with the current YOLO training pipeline — annotations must be in YOLO `.txt` format with one file per image. Use `convert_coco()` to convert your COCO JSON first, then follow this [guide](#step-by-step-conversion-guide) to organize and train. For more on supported formats, see [dataset formats](../datasets/detect/index.md).

### Can I convert COCO segmentation annotations to YOLO format?

Yes, use `use_segments=True` when calling `convert_coco()` to include polygon segmentation masks in the converted YOLO labels. This produces label files compatible with [YOLO segmentation models](../tasks/segment.md):

```python
from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="annotations/", save_dir="output/", use_segments=True, cls91to80=False)
```

### How do I convert COCO keypoint annotations to YOLO format?

Use `use_keypoints=True` to convert COCO keypoint annotations for [pose estimation](../tasks/pose.md) training:

```python
from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="annotations/", save_dir="output/", use_keypoints=True, cls91to80=False)
```

Note that if both `use_segments` and `use_keypoints` are set to `True`, only keypoints will be written to the label files — segments are silently ignored.
