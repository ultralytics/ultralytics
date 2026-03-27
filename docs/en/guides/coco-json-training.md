---
comments: true
description: Train Ultralytics YOLO directly on COCO JSON annotations without converting to YOLO format. Custom dataset and trainer example with complete working code for detection training.
keywords: COCO JSON training, train YOLO on COCO JSON, COCO JSON without conversion, custom YOLO dataset, custom YOLO trainer, COCO annotations YOLO, direct COCO training, Ultralytics YOLO, object detection training, YOLODataset subclass, COCO format training, skip annotation conversion
---

# How to Train YOLO on COCO JSON Without Converting

## Why Train Directly on COCO JSON

[Annotations](https://www.ultralytics.com/glossary/data-labeling) in [COCO JSON](https://cocodataset.org/#format-data) format can be used directly for [Ultralytics YOLO](https://www.ultralytics.com/) training without converting to `.txt` files first. This is done by subclassing `YOLODataset` to parse COCO JSON on the fly and wiring it into the training pipeline through a custom trainer.

This approach keeps the COCO JSON as the single source of truth — no `convert_coco()` call, no directory reorganization, no intermediate label files. [YOLO26](../models/yolo26.md) and all other Ultralytics YOLO detection models are supported. Segmentation and pose models require additional label fields (see [FAQ](#does-this-support-segmentation-and-pose-estimation)).

!!! tip "Looking for a one-time conversion instead?"

    See the [COCO to YOLO Conversion guide](coco-to-yolo.md) for the standard `convert_coco()` workflow.

## Architecture Overview

Two classes are needed:

1. **`COCOJSONDataset`** — reads COCO JSON and converts [bounding boxes](https://www.ultralytics.com/glossary/bounding-box) to YOLO format in memory during training
2. **`COCOJSONTrainer`** — overrides `build_dataset()` to use `COCOJSONDataset` instead of the default `YOLODataset`

The implementation follows the same pattern as the built-in `GroundingDataset`, which also reads JSON annotations directly. Three methods are overridden: `get_img_files()`, `cache_labels()`, and `get_labels()`.

## Building the COCO JSON Dataset Class

The `COCOJSONDataset` class inherits from `YOLODataset` and overrides the label loading logic. Instead of reading `.txt` files from a labels directory, it opens the COCO JSON file, iterates over annotations grouped by image, and converts each bounding box from COCO pixel format `[x_min, y_min, width, height]` to YOLO normalized center format `[x_center, y_center, width, height]`. Crowd annotations (`iscrowd: 1`) and zero-area boxes are skipped automatically.

The `get_img_files()` method returns an empty list because image paths are resolved from the JSON `file_name` field inside `cache_labels()`. Category IDs are sorted and remapped to zero-indexed class indices, so both 1-based (standard COCO) and non-contiguous ID schemes work correctly.

```python
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import get_hash, load_dataset_cache_file, save_dataset_cache_file
from ultralytics.utils import TQDM


class COCOJSONDataset(YOLODataset):
    """Dataset that reads COCO JSON annotations directly without conversion to .txt files."""

    def __init__(self, *args, json_file="", **kwargs):
        self.json_file = json_file
        super().__init__(*args, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path):
        """Image paths are resolved from the JSON file, not from scanning a directory."""
        return []

    def cache_labels(self, path=Path("./labels.cache")):
        """Parse COCO JSON and convert annotations to YOLO format. Results are saved to a .cache file."""
        x = {"labels": []}
        with open(self.json_file) as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}

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

Parsed labels are saved to a `.cache` file next to the JSON (e.g. `instances_train.cache`). On subsequent training runs, the cache is loaded directly, skipping JSON parsing. If the JSON file changes, the hash check fails and the cache is rebuilt automatically.

## Connecting the Dataset to the Training Pipeline

The only change needed in the trainer is overriding `build_dataset()`. The default `DetectionTrainer` builds a `YOLODataset` that scans for `.txt` label files. By replacing it with `COCOJSONDataset`, the trainer reads from the COCO JSON instead.

The JSON file path is pulled from a custom `train_json` / `val_json` field in the data config (see Step 3). During training, `mode="train"` resolves to `train_json`; during validation, `mode="val"` resolves to `val_json`. If `val_json` is not set, it falls back to `train_json`.

```python
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr


class COCOJSONTrainer(DetectionTrainer):
    """Trainer that uses COCOJSONDataset for direct COCO JSON training."""

    def build_dataset(self, img_path, mode="train", batch=None):
        json_file = self.data["train_json"] if mode == "train" else self.data.get("val_json", self.data["train_json"])
        return COCOJSONDataset(
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

The `dataset.yaml` uses the standard `path`, `train`, and `val` fields to locate image directories. Two additional fields, `train_json` and `val_json`, specify the COCO annotation files that `COCOJSONTrainer` reads. The `nc` and `names` fields define the number of classes and their names, matching the sorted order of `categories` in the JSON.

```yaml
path: /path/to/images # root directory with train/ and val/ subfolders
train: train
val: val

# COCO JSON annotation files
train_json: /path/to/annotations/instances_train.json
val_json: /path/to/annotations/instances_val.json

nc: 80
names:
    0: person
    1: bicycle
    # ... remaining class names
```

Expected directory structure:

```
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

With the dataset class, trainer class, and YAML config in place, training works through the standard `model.train()` call. The only difference from a normal training run is the `trainer=COCOJSONTrainer` argument, which tells Ultralytics to use the custom dataset loader instead of the default one.

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(data="dataset.yaml", epochs=100, imgsz=640, trainer=COCOJSONTrainer)
```

The full [training](../modes/train.md) pipeline runs as expected, including [validation](../modes/val.md), checkpoint saving, and metric logging.

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


class COCOJSONDataset(YOLODataset):
    """Dataset that reads COCO JSON annotations directly without conversion to .txt files."""

    def __init__(self, *args, json_file="", **kwargs):
        self.json_file = json_file
        super().__init__(*args, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path):
        return []

    def cache_labels(self, path=Path("./labels.cache")):
        x = {"labels": []}
        with open(self.json_file) as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}
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
        x["hash"] = get_hash([self.json_file, str(self.img_path)])
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
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


class COCOJSONTrainer(DetectionTrainer):
    """Trainer that uses COCOJSONDataset for direct COCO JSON training."""

    def build_dataset(self, img_path, mode="train", batch=None):
        json_file = self.data["train_json"] if mode == "train" else self.data.get("val_json", self.data["train_json"])
        return COCOJSONDataset(
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
model.train(data="dataset.yaml", epochs=100, imgsz=640, trainer=COCOJSONTrainer)
```

For [hyperparameter](https://www.ultralytics.com/glossary/hyperparameter-tuning) recommendations, see the [Model Training Tips](model-training-tips.md) guide.

## FAQ

### What is the difference between this and convert_coco()?

[`convert_coco()`](../reference/data/converter.md#ultralytics.data.converter.convert_coco) writes `.txt` label files to disk as a one-time conversion. This approach parses the JSON at the start of each training run and converts annotations in memory. Use `convert_coco()` when permanent YOLO-format labels are preferred; use this approach to keep the COCO JSON as the single source of truth without generating additional files.

### Can YOLO train on COCO JSON without custom code?

Not with the current Ultralytics pipeline, which expects YOLO `.txt` labels by default. This guide provides the minimal custom code needed — one dataset class and one trainer class. Once defined, training requires only a standard `model.train()` call.

### Does this support segmentation and pose estimation?

This guide covers [object detection](https://www.ultralytics.com/glossary/object-detection). To add [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) support, include the `segmentation` polygon data from COCO annotations in the `segments` field of each label dictionary. For [pose estimation](https://www.ultralytics.com/glossary/pose-estimation), include `keypoints`. The `GroundingDataset` [source code](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/dataset.py) provides a reference implementation for handling segments.

### Do augmentations work with this custom dataset?

Yes. `COCOJSONDataset` extends `YOLODataset`, so all built-in [data augmentations](yolo-data-augmentation.md) — [mosaic](yolo-data-augmentation.md#mosaic-mosaic), [mixup](yolo-data-augmentation.md#mixup-mixup), [copy-paste](yolo-data-augmentation.md#copy-paste-copy_paste), and others — run without modification.

### How are category IDs mapped to class indices?

Categories are sorted by `id` and mapped to sequential indices starting from 0. This handles 1-based IDs (standard COCO), 0-based IDs, and non-contiguous IDs. The `names` dictionary in `dataset.yaml` should follow the same sorted order as the COCO `categories` array.

### Is there a performance overhead compared to pre-converted labels?

The COCO JSON is parsed once on the first training run. Parsed labels are saved to a `.cache` file, so subsequent runs load instantly without re-parsing. Training speed is identical to standard YOLO training since annotations are held in memory. The cache is rebuilt automatically if the JSON file changes.
