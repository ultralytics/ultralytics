---
comments: true
description: Learn how to prepare semantic segmentation datasets for Ultralytics YOLO, including PNG mask labels, dataset YAML fields, ignore labels, and supported datasets.
keywords: Ultralytics, YOLO, semantic segmentation, semantic, dataset format, pixel masks, Cityscapes, ADE20K, Pascal VOC
title: Semantic Segmentation Datasets
---

# Semantic Segmentation Datasets Overview

[Semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation) assigns one class label to every pixel in an image. Unlike [instance segmentation](../segment/index.md), semantic segmentation does not separate individual objects of the same class. The training target is a dense class map where each pixel stores a class ID.

This guide explains the dataset format used by Ultralytics YOLO semantic segmentation models and lists the built-in dataset configurations available for training and validation.

## Supported Dataset Formats

Two label formats are supported. The dataset loader picks the path based on whether the dataset YAML defines a `masks_dir` key.

### PNG mask format

Semantic segmentation datasets use one image file and one mask file per sample. The mask is a single-channel image, usually PNG, where each pixel value is the class index for the corresponding image pixel.

- Pixel values `0`, `1`, `2`, ... represent class IDs from the dataset `names` mapping.
- Pixel value `255` is treated as the ignore label and is excluded from loss and metric computation.
- Mask files should use the same stem as their matching image file, for example `frankfurt_000000_000294.png`.
- Masks are resolved as `.png` by default; if missing, other supported image extensions are also accepted. Use lossless formats such as `.png` or `.tiff`, since lossy compression (e.g. `.jpg`) corrupts the class ID pixel values.

The default layout keeps images and masks in parallel folders. The `masks_dir` value from the dataset YAML replaces the `images` path component to find masks.

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── masks/
    ├── train/
    └── val/
```

For example, an image at `images/train/aachen_000000_000019.png` is paired with a mask at `masks/train/aachen_000000_000019.png` when `masks_dir: masks`.

### YOLO polygon label format

If your dataset already has Ultralytics YOLO polygon labels (one `.txt` per image with `<class-index> <x1> <y1> <x2> <y2> ...` rows), you can train semantic segmentation directly from them — no PNG mask conversion needed. See the [instance segmentation dataset format](../segment/index.md#ultralytics-yolo-format) for the row-level layout.

This path is selected automatically when the dataset YAML **omits** `masks_dir`. Behavior:

- Polygons are converted to a per-image semantic mask at load time, sorted by area so smaller objects override larger ones in overlap regions.
- **Multi-class** (`N > 1` in `names`): an extra `background` class is appended after your declared classes for pixels not covered by any polygon. The model is built with `N + 1` output channels and the last channel is background.
- **Single-class** (`N == 1` in `names`): still trained as 1 class. The mask is binary, with your declared class shown as `1` and pixels not covered by any polygon as `0`. No extra background class is added to `names`.
- Pixels added by augmentation padding (e.g. random crop) still use `255` as the ignore label.

Use this path when your data is already labeled as instance polygons and you want a semantic segmentation model from the same files.

### Dataset YAML format

Semantic segmentation datasets are configured with YAML files. The main fields are:

| Key             | Description                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------- |
| `path`          | Dataset root directory.                                                                           |
| `train`         | Training image path relative to `path`, or an absolute path.                                      |
| `val`           | Validation image path relative to `path`, or an absolute path.                                    |
| `test`          | Optional test image path.                                                                         |
| `masks_dir`     | Directory name used for semantic masks. Omit this key to switch to the YOLO polygon label format. |
| `names`         | Class ID to class name mapping.                                                                   |
| `label_mapping` | Optional mapping from source dataset IDs to training IDs or `ignore_label`.                       |

!!! example "ultralytics/cfg/datasets/cityscapes8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/cityscapes8.yaml"
    ```

Use `label_mapping` when the source mask IDs do not already match contiguous training class IDs. Cityscapes and ADE20K include mappings that convert original label IDs into YOLO semantic segmentation train IDs and ignore unused labels.

## Usage

Train a YOLO26 semantic segmentation model with Python or CLI:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained semantic segmentation model
        model = YOLO("yolo26n-sem.pt")

        # Train on the Cityscapes8 semantic segmentation dataset
        results = model.train(data="cityscapes8.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        yolo semantic train data=cityscapes8.yaml model=yolo26n-sem.pt epochs=100 imgsz=1024
        ```

## Supported Datasets

Ultralytics provides semantic segmentation dataset YAML files for these datasets:

- [Cityscapes](cityscapes.md): Urban street-scene semantic segmentation dataset with 19 train classes.
- [Cityscapes8](cityscapes8.md): An 8-image Cityscapes subset for quick tests and CI checks.
- [ADE20K](ade20k.md): Scene parsing dataset with 150 semantic classes.

## Adding Your Own Dataset

### Option A — PNG masks

1. Save your images under split folders such as `images/train` and `images/val`.
2. Save one single-channel mask per image under the mirrored mask folders, such as `masks/train` and `masks/val`.
3. Ensure mask pixel values are class IDs. Use `255` for pixels that should be ignored.
4. Create a dataset YAML with `path`, `train`, `val`, `masks_dir`, and `names`.
5. Add `label_mapping` only when your mask IDs need conversion to contiguous train IDs.

```yaml
path: path/to/my-semantic-dataset
train: images/train
val: images/val
masks_dir: masks

names:
    0: background
    1: road
    2: building
```

### Option B — Polygon labels

1. Lay out images and `.txt` polygon files exactly as for [instance segmentation](../segment/index.md).
2. Create a dataset YAML with `path`, `train`, `val`, and `names` — **omit** `masks_dir`.
3. Do not add a "background" entry to `names`. For multi-class datasets the loader appends one automatically; for single-class datasets training stays at 1 class — your declared class becomes `1` in the mask and uncovered pixels become `0`.

```yaml
path: path/to/my-polygon-dataset
train: images/train
val: images/val

names:
    0: person
    1: car
```

## FAQ

### What is the difference between semantic segmentation masks and instance segmentation labels?

Semantic segmentation masks are dense pixel maps. Each pixel stores a class ID, and there is one mask image per training image. Instance segmentation labels in Ultralytics YOLO use text files with polygon coordinates, one row per object instance.

### What pixel value is ignored during training?

Pixel value `255` is used as the ignore label. These pixels are skipped during loss and metric computation, which is useful for void regions, unlabeled pixels, or classes outside the training label set.

### Do mask file names need to match image file names?

Yes. Each semantic mask should have the same file stem as the corresponding image. The dataset loader replaces the `images` directory component with `masks_dir` and searches for matching mask files.

### Can I use original dataset label IDs directly?

Yes, if they already match your `names` class IDs. If the source dataset uses non-contiguous IDs or includes labels that should be ignored, add a `label_mapping` section to convert source pixel values to training IDs.

### Can I use my instance segmentation dataset to train semantic segmentation?

Yes. Instance segmentation datasets use Ultralytics YOLO polygon labels (one `.txt` per image with `<class-index> <x1> <y1> <x2> <y2> ...` rows), and the same files can be reused for semantic segmentation — just **omit** `masks_dir` from the dataset YAML. The loader converts polygons to per-image masks on the fly. For multi-class datasets (`N > 1`) an extra `background` class is appended and the model is built with `N + 1` output channels. For single-class datasets (`N == 1`) training stays at 1 class — the mask shows your declared class as `1` and uncovered pixels as `0`.
