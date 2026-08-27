---
title: Train YOLO on COCO JSON Without Conversion
comments: true
description: Train Ultralytics YOLO directly on COCO JSON annotations without converting to YOLO format, using a custom dataset and trainer with complete working code.
keywords: COCO JSON training, train YOLO on COCO JSON, COCO JSON without conversion, custom YOLO dataset, custom YOLO trainer, COCO annotations YOLO, direct COCO training, Ultralytics YOLO, object detection training, YOLODataset subclass, COCO format training, skip annotation conversion
---

# How to Train YOLO on COCO JSON Without Converting

[Annotations](https://www.ultralytics.com/glossary/data-labeling) in [COCO JSON](https://cocodataset.org/#format-data) format can be used directly for [Ultralytics YOLO](https://www.ultralytics.com) training without converting to `.txt` files first. This works by subclassing [`YOLODataset`](../reference/data/dataset.md#ultralytics.data.dataset.YOLODataset) to parse COCO JSON on the fly and wiring it into the training pipeline through a custom trainer.

## Why Train Directly on COCO JSON

This approach keeps the COCO JSON as the single source of truth — no `convert_coco()` call, no directory reorganization, no intermediate label files. [YOLO26](../models/yolo26.md) and all other Ultralytics YOLO detection models are supported. Segmentation and pose models require additional label fields (see [FAQ](#does-this-support-segmentation-and-pose-estimation)).

!!! tip "Looking for a one-time conversion instead?"

    See the [COCO to YOLO Conversion guide](coco-to-yolo.md) for the standard `convert_coco()` workflow.

## Architecture Overview

Two classes are needed:

1. **`COCODataset`** — reads COCO JSON and converts [bounding boxes](https://www.ultralytics.com/glossary/bounding-box) to YOLO format in memory during training
2. **`COCOTrainer`** — overrides `build_dataset()` to use `COCODataset` instead of the default `YOLODataset`

The implementation is a simplified version of the built-in [`GroundingDataset`](../reference/data/dataset.md#ultralytics.data.dataset.GroundingDataset), which also reads JSON annotations directly. Three methods are overridden here — `get_img_files()`, `cache_labels()`, and `get_labels()` — where `GroundingDataset` overrides more, including its own cache-hash and instance-count checks.

## Building the COCO JSON Dataset Class

The `COCODataset` class inherits from `YOLODataset` and overrides the label loading logic. Instead of reading `.txt` files from a labels directory, it opens the COCO JSON file, iterates over annotations grouped by image, and converts each bounding box from COCO pixel format `[x_min, y_min, width, height]` to YOLO normalized center format `[x_center, y_center, width, height]`. Crowd annotations (`iscrowd: 1`) and zero-area boxes are skipped automatically.

The `get_img_files()` method returns an empty list because image paths are resolved from the JSON `file_name` field inside `cache_labels()`. Category IDs are sorted and remapped to zero-indexed class indices, so both 1-based (standard COCO) and non-contiguous ID schemes work correctly.

```python
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import get_hash, load_dataset_cache_file, save_dataset_cache_file
from ultralytics.utils import TQDM


class COCODataset(YOLODataset):
    """Dataset that reads COCO JSON annotations directly without conversion to .txt files."""

    def __init__(self, *args, json_file="", **kwargs):
        """Initialize the dataset with a COCO JSON annotation file."""
        self.json_file = json_file
        super().__init__(*args, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path):
        """Image paths are resolved from the JSON file, not from scanning a directory."""
        self.fraction = 1.0  # fraction is applied while scanning a directory, which this dataset skips
        return []

    def cache_labels(self, path=Path("./labels.cache")):
        """Parse COCO JSON and convert annotations to YOLO format. Results are saved to a .cache file."""
        x = {"labels": []}
        with open(self.json_file) as f:
            coco = json.load(f)

        # Sort categories by ID and map to 0-indexed classes
        categories = {cat["id"]: i for i, cat in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))}

        img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        for img_info in TQDM(coco["images"], desc="reading annotations"):
            h, w = img_info["height"], img_info["width"]
            im_file = Path(self.img_path) / img_info["file_name"]
            if not im_file.exists():
                continue

            self.im_files.append(str(im_file))
            bboxes = []
            for ann in img_to_anns.get(img_info["id"], []):
                if ann.get("iscrowd", False):
                    continue
                # COCO: [x, y, w, h] top-left in pixels -> YOLO: [cx, cy, w, h] center normalized
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2  # top-left to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:
                    continue
                cls = categories[ann["category_id"]]
                bboxes.append([cls, *box.tolist()])

            lb = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 5), dtype=np.float32)
            x["labels"].append(
                {
                    "im_file": str(im_file),
                    "shape": (h, w),
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:],
                    "segments": [],
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        if not x["labels"]:
            raise RuntimeError(f"No images listed in {self.json_file} were found in {self.img_path}")
        x["hash"] = get_hash([self.json_file, str(self.img_path)])
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """Load labels from .cache file if available, otherwise parse JSON and create the cache."""
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache = load_dataset_cache_file(cache_path)
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash([self.json_file, str(self.img_path)])
            self.im_files = [lb["im_file"] for lb in cache["labels"]]
        except (FileNotFoundError, AssertionError, AttributeError, KeyError, ModuleNotFoundError):
            cache = self.cache_labels(cache_path)
        cache.pop("hash", None)
        cache.pop("version", None)
        return cache["labels"]
```

Parsed labels are saved to a `.cache` file next to the JSON (e.g. `instances_train.cache`). On subsequent training runs, the cache is loaded directly, skipping JSON parsing.

!!! warning "The cache key is the JSON's file size, not its contents"

    `get_hash()` hashes file sizes and paths rather than file contents, so a re-run re-parses the JSON only when the JSON's byte count changes. Adding or removing images may also shift the image directory's own size and trigger a rebuild, but do not rely on it — the hash never inspects individual image files, so swapping one image for another can leave the size unchanged. An edit that preserves the byte count — nudging a coordinate, flipping `iscrowd`, swapping two equal-length class names — leaves the stale cache in place and trains on the old annotations with no warning, and replacing an image in place is invisible for the same reason. Delete the `.cache` file after editing annotations or images in place.

## Connecting the Dataset to the Training Pipeline

The only change needed in the trainer is overriding `build_dataset()`. The default `DetectionTrainer` builds a `YOLODataset` that scans for `.txt` label files. By replacing it with `COCODataset`, the trainer reads from the COCO JSON instead.

The JSON file path is pulled from a custom `train_json` / `val_json` field in the data config (see [Configuring dataset.yaml](#configuring-datasetyaml-for-coco-json)). During training, `mode="train"` resolves to `train_json`; during validation, `mode="val"` resolves to `val_json`. Both keys are required — the two splits read different image directories, so the training JSON cannot stand in for a missing `val_json`.

The dataset also resets `fraction` to `1.0`. `BaseDataset` applies that argument while scanning an image directory, a step `COCODataset` skips, so it cannot honor a partial-dataset request; resetting it keeps the dataset from appearing to accept a value it ignores. The built-in `GroundingDataset` makes the same compromise for the same reason.

```python
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr


class COCOTrainer(DetectionTrainer):
    """Trainer that uses COCODataset for direct COCO JSON training."""

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build a COCODataset for the given split using the JSON file from the data config."""
        json_file = self.data["train_json"] if mode == "train" else self.data["val_json"]
        return COCODataset(
            img_path=img_path,
            json_file=json_file,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(self.model.stride.max()) if hasattr(self, "model") and self.model else 32,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )
```

## Configuring dataset.yaml for COCO JSON

The `dataset.yaml` uses the standard `path`, `train`, and `val` fields to locate image directories. Note that `path` points at the image root here, so `train` and `val` are bare split names — unlike the [conversion guide](coco-to-yolo.md), where `path` is the dataset root and the splits carry an `images/` prefix. Two additional fields, `train_json` and `val_json`, specify the COCO annotation files that `COCOTrainer` reads. The `names` field lists the class names in the sorted order of `categories` in the JSON, and the class count is derived from it, so there is no need to set `nc`.

```yaml
path: /path/to/my_dataset/images # root with train/ and val/ image subfolders
train: train
val: val

# COCO JSON annotation files (use absolute paths; these custom keys are not resolved against `path`)
train_json: /path/to/my_dataset/annotations/instances_train.json
val_json: /path/to/my_dataset/annotations/instances_val.json

names:
    0: person
    1: bicycle
    # ... remaining class names
```

Expected directory structure:

```text
my_dataset/
  images/
    train/
      img_001.jpg
      ...
    val/
      img_100.jpg
      ...
  annotations/
    instances_train.json
    instances_val.json
  dataset.yaml
```

## Running Training on COCO JSON

With the dataset class, trainer class, and YAML config in place, training works through the standard `model.train()` call. The only difference from a normal training run is the `trainer=COCOTrainer` argument, which tells Ultralytics to use the custom dataset loader instead of the default one.

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(data="dataset.yaml", epochs=100, imgsz=640, trainer=COCOTrainer)
```

The full [training](../modes/train.md) pipeline runs as expected, including in-training [validation](../modes/val.md), checkpoint saving, and metric logging.

!!! note "Standalone `model.val()` needs its own override"

    Only training-time validation goes through `COCOTrainer.build_dataset`. A separate `model.val()` call builds the stock `YOLODataset`, which scans for `.txt` labels beside the images and finds none. It does not raise: the images are counted as backgrounds, so validation runs to completion and reports every metric as `0`, warning `No labels found in ...` and `no labels found in detect set, cannot compute metrics without labels`. To validate outside a training run, subclass the validator with the same `build_dataset` override and pass it to `model.val(validator=...)`.

## Full Implementation

For convenience, the full implementation is provided below as a single copy-paste script. It includes the custom dataset, custom trainer, and the training call. Save this alongside your `dataset.yaml` and run it directly.

```python
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ultralytics import YOLO
from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import get_hash, load_dataset_cache_file, save_dataset_cache_file
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import TQDM, colorstr


class COCODataset(YOLODataset):
    """Dataset that reads COCO JSON annotations directly without conversion to .txt files."""

    def __init__(self, *args, json_file="", **kwargs):
        """Initialize the dataset with a COCO JSON annotation file."""
        self.json_file = json_file
        super().__init__(*args, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path):
        """Image paths are resolved from the JSON file, not from scanning a directory."""
        self.fraction = 1.0  # fraction is applied while scanning a directory, which this dataset skips
        return []

    def cache_labels(self, path=Path("./labels.cache")):
        """Parse COCO JSON and convert annotations to YOLO format. Results are saved to a .cache file."""
        x = {"labels": []}
        with open(self.json_file) as f:
            coco = json.load(f)

        categories = {cat["id"]: i for i, cat in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))}

        img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        for img_info in TQDM(coco["images"], desc="reading annotations"):
            h, w = img_info["height"], img_info["width"]
            im_file = Path(self.img_path) / img_info["file_name"]
            if not im_file.exists():
                continue

            self.im_files.append(str(im_file))
            bboxes = []
            for ann in img_to_anns.get(img_info["id"], []):
                if ann.get("iscrowd", False):
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= w
                box[[1, 3]] /= h
                if box[2] <= 0 or box[3] <= 0:
                    continue
                cls = categories[ann["category_id"]]
                bboxes.append([cls, *box.tolist()])

            lb = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 5), dtype=np.float32)
            x["labels"].append(
                {
                    "im_file": str(im_file),
                    "shape": (h, w),
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:],
                    "segments": [],
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        if not x["labels"]:
            raise RuntimeError(f"No images listed in {self.json_file} were found in {self.img_path}")
        x["hash"] = get_hash([self.json_file, str(self.img_path)])
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """Load labels from .cache file if available, otherwise parse JSON and create the cache."""
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache = load_dataset_cache_file(cache_path)
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash([self.json_file, str(self.img_path)])
            self.im_files = [lb["im_file"] for lb in cache["labels"]]
        except (FileNotFoundError, AssertionError, AttributeError, KeyError, ModuleNotFoundError):
            cache = self.cache_labels(cache_path)
        cache.pop("hash", None)
        cache.pop("version", None)
        return cache["labels"]


class COCOTrainer(DetectionTrainer):
    """Trainer that uses COCODataset for direct COCO JSON training."""

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build a COCODataset for the given split using the JSON file from the data config."""
        json_file = self.data["train_json"] if mode == "train" else self.data["val_json"]
        return COCODataset(
            img_path=img_path,
            json_file=json_file,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(self.model.stride.max()) if hasattr(self, "model") and self.model else 32,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )


model = YOLO("yolo26n.pt")
model.train(data="dataset.yaml", epochs=100, imgsz=640, trainer=COCOTrainer)
```

You now have a minimal dataset and trainer that train Ultralytics YOLO directly on COCO JSON, with annotations staying the single source of truth and no intermediate `.txt` files. Extend the `cache_labels()` method with `segments` or `keypoints` to cover segmentation and pose, and see the [Model Training Tips](model-training-tips.md) guide for [hyperparameter](https://www.ultralytics.com/glossary/hyperparameter-tuning) tuning recommendations.

## FAQ

### What is the difference between this and convert_coco()?

[`convert_coco()`](../reference/data/converter.md#ultralytics.data.converter.convert_coco) writes `.txt` label files to disk as a one-time conversion. This approach parses the JSON at the start of each training run and converts annotations in memory. Use `convert_coco()` when permanent YOLO-format labels are preferred; use this approach to keep the COCO JSON as the single source of truth without generating additional files.

### Can YOLO train on COCO JSON without custom code?

Not with the current Ultralytics pipeline, which expects YOLO `.txt` labels by default. This guide provides the minimal custom code needed — one dataset class and one trainer class. Once defined, training requires only a standard `model.train()` call.

### Does this support segmentation and pose estimation?

This guide covers [object detection](https://www.ultralytics.com/glossary/object-detection). To add [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) support, include the `segmentation` polygon data from COCO annotations in the `segments` field of each label dictionary. For [pose estimation](https://www.ultralytics.com/glossary/pose-estimation), include `keypoints`. The [`GroundingDataset`](../reference/data/dataset.md#ultralytics.data.dataset.GroundingDataset) [source code](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/dataset.py) provides a reference implementation for handling segments.

### Do augmentations work with this custom dataset?

Yes. `COCODataset` extends `YOLODataset`, so all built-in [data augmentations](yolo-data-augmentation.md) — [mosaic](yolo-data-augmentation.md#mosaic-mosaic), [mixup](yolo-data-augmentation.md#mixup-mixup), [copy-paste](yolo-data-augmentation.md#copy-paste-copy_paste), and others — run without modification.

### How are category IDs mapped to class indices?

Categories are sorted by `id` and mapped to sequential indices starting from 0. This handles 1-based IDs (standard COCO), 0-based IDs, and non-contiguous IDs. The `names` dictionary in `dataset.yaml` should follow the same sorted order as the COCO `categories` array.

### Is there a performance overhead compared to pre-converted labels?

The COCO JSON is parsed once on the first training run. Parsed labels are saved to a `.cache` file, so subsequent runs load instantly without re-parsing. Training speed is identical to standard YOLO training since annotations are held in memory. The cache is keyed on the JSON's file size, so delete the `.cache` file after any edit that leaves the file the same length.
