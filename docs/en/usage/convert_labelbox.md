# convert_labelbox

`convert_labelbox` converts a Labelbox NDJSON export into a YOLO **detection** dataset (bounding boxes only).

## Prerequisites and Workflow

- You have already **exported your project from Labelbox as NDJSON**.
- You have **downloaded the corresponding images** referenced by `externalId` into a local `images_dir`.
- You have created a simple **class map YAML** (e.g. `class_map.yaml`) that maps Labelbox label strings → YOLO class IDs.

This utility is a **conversion helper**, not a full Labelbox integration. It:

- Reads your NDJSON and class map.
- Writes YOLO-format label TXT files and a `data.yaml` under `save_dir`.

It **does not**:

- Call the Labelbox API or download images for you.
- Handle segmentation/polygon annotations in v1 (bounding boxes only).
- Automatically adapt to arbitrary NDJSON schemas; it assumes a Labelbox-style schema with fields like `externalId`, `imageWidth`, `imageHeight`, and `objects[*].bbox`.

## Python

```python
from ultralytics.data.labelbox import convert_labelbox

convert_labelbox(
    labels_path="export.ndjson",
    images_dir="images/",
    save_dir="datasets/myproject",
    class_map="class_map.yaml",
)
```

## CLI (proposed)

```bash
yolo convert labelbox \
  labels=export.ndjson \
  images_dir=images/ \
  save_dir=datasets/myproject \
  class_map=class_map.yaml
```

This will create a YOLO-style folder structure under `save_dir` and a dataset YAML using the class names and IDs from `class_map.yaml`.
