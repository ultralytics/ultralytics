---
comments: true
description: Prepare datasets for Ultralytics YOLOA anomaly detection, including the normal-image fit set, the optional labeled-defect train set, and the validation layout.
keywords: Ultralytics, YOLOA, anomaly detection dataset, defect detection, normal images, one-class learning, memory bank, dataset format
---

# Anomaly Detection Datasets Overview

[Anomaly detection](https://www.ultralytics.com/glossary/anomaly-detection) with Ultralytics YOLOA uses two distinct kinds of data: a **fit set** of normal images that builds the memory bank, and an optional **train set** of labeled defects for gradient fine-tuning. Keeping the two apart is the single most important dataset decision — the fit set needs no labels at all, while the train set is a standard YOLO detection dataset.

## Fit dataset (normal images)

The fit set is a plain folder of good, defect-free images. No label files, no dataset YAML — structurally the same "train from a folder" convention as [classification datasets](../classify/index.md).

```text
bottle/
└── train/
    └── good/
        ├── 000.png
        ├── 001.png
        └── ...
```

Point `fit()` at the folder (or pass an explicit list of image paths):

```python
from ultralytics.yoloa import YOLOA

model = YOLOA("yolo26n-anomaly.yaml")
model.fit("bottle/train/good", name="bottle")
```

Supported image extensions: `avif`, `bmp`, `dng`, `heic`, `heif`, `jp2`, `jpeg`, `jpeg2000`, `jpg`, `mpo`, `png`, `tif`, `tiff`, `webp`. Video sources are not supported for fitting.

## Train dataset (labeled defects, optional)

The optional gradient fine-tune consumes a standard [YOLO detection dataset](../detect/index.md): images plus one `.txt` label file per image with normalized `class x_center y_center width height` rows, described by a dataset YAML.

```yaml
# defects.yaml
path: ../datasets/defects
train: images/train
val: images/val
nc: 1
names: [defect]
```

During training, the anomaly prior mask is rendered automatically from the ground-truth boxes — you do not need to supply masks. If your labels are polygon segments, setting `seg_target_polygon: true` in the model YAML `anomaly` block renders the prior from the polygon union instead of boxes.

## Validation layout

Validation uses the same detection-style YAML, with the `val` split containing defect images and their labeled boxes:

```yaml
# bottle.yaml
path: ../datasets/bottle
train: train/good # normal images; satisfies the required train key
val: images/val # defect images with standard YOLO box labels
nc: 1
names: [defect]
```

The memory bank comes from your `fit()` call or a fitted checkpoint — not from the YAML — so fit before validating:

```python
from ultralytics.yoloa import YOLOA

model = YOLOA("yolo26n-anomaly.yaml")
model.fit("bottle/train/good", name="bottle")
metrics = model.val(data="bottle.yaml")
```

With a fitted bank, the validator adds mAP10 and mAP25 columns (IoU 0.10 and 0.25) for coarse defect localization alongside standard detection metrics. Note that `val()` resets the bank when it finishes — re-fit (cached banks reload instantly) before predicting with the same model instance.

## Supported datasets

Downloadable dataset configurations for anomaly detection have not been published yet. Prepare your own data using the layouts above: collect normal images of your product or scene for the fit set, and optionally label defect boxes in [standard YOLO format](../detect/index.md) for fine-tuning and validation. You can annotate, manage, and version your defect datasets on the [Ultralytics Platform](https://platform.ultralytics.com).

## FAQ

### Do I need labeled images to use YOLOA?

No — the required fit set is a plain folder of normal images with no labels and no YAML. Labels are only needed for the optional `train()` fine-tune and for computing validation metrics, both of which use the standard YOLO detection format.

### How many normal images should the fit set contain?

The memory bank keeps at most 10,000 feature vectors after coreset compression, and each image contributes a grid of patch features, so even a few dozen representative normal images produce a full bank. Cover the natural variation of your normal class — lighting, orientation, texture — rather than maximizing raw image count.

### Why does my fit set have no dataset YAML?

`fit()` performs feature extraction, not training: it only needs images, so it takes a folder path or a list of image paths directly. A dataset YAML describes labeled splits for training and validation, which the fit step does not use. This mirrors how classification trains from a folder without a YAML.

### Can I use an existing detection dataset with YOLOA?

Yes, for the optional fine-tune and for validation: any standard YOLO detection dataset of defect boxes works unchanged with `model.train(data="defects.yaml")` and `model.val(data="defects.yaml")`. For the fit set, extract the defect-free images into their own folder — the memory bank must see only normal samples.
