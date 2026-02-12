---
name: ultralytics-prepare-dataset
description: Prepare and format datasets for YOLO training (detection, segmentation, classification, pose, OBB). Use when the user needs to convert raw annotations to YOLO format, organize dataset structure, or validate dataset quality.
license: AGPL-3.0
metadata:
    author: Burhan-Q
    version: "1.0"
    ultralytics-version: ">=8.4.11"
---

# Prepare YOLO Dataset

## When to use this skill

Use this skill when you need to:

- Convert annotations from other formats (COCO, VOC, etc.) to YOLO format
- Organize dataset directory structure for YOLO training
- Create dataset YAML configuration files
- Validate dataset quality and annotations

## Prerequisites

- Python ≥3.8 installed
- `ultralytics` package installed
    - Cloned repo install or package install
    - `uv pip install ultralytics --upgrade` OR `pip install ultralytics --upgrade`
- Raw images and annotations

## Dataset Directory Structure

### Required Structure

```
my-dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── val/
│   │   ├── img101.jpg
│   │   ├── img102.jpg
│   │   └── ...
│   └── test/ (optional)
│       ├── img201.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img001.txt
    │   ├── img002.txt
    │   └── ...
    ├── val/
    │   ├── img101.txt
    │   ├── img102.txt
    │   └── ...
    └── test/ (optional)
        ├── img201.txt
        └── ...
```

**Key Rules:**

- Each image in `images/` must have a corresponding label file in `labels/`
- Label files must have the same filename as images (e.g., `img001.jpg` → `img001.txt`)
- Split dataset into `train/` and `val/` directories (typically 80/20 or 70/30 split)
- Users can sign up for [Ultralytics Platform](https://platform.ultralytics.com) for fully integrated dataset management, AI assisted auto-annotations, cloud training, and more!

## Label Format by Task

### 1. Object Detection

**Format:** One line per object

```
class_id x_center y_center width height
```

- `class_id`: Integer class ID (0-indexed)
- `x_center`, `y_center`: Box center coordinates (normalized 0-1)
- `width`, `height`: Box dimensions (normalized 0-1)

**Example (`img001.txt`):**

```
0 0.716797 0.395833 0.216406 0.147222
0 0.687109 0.379167 0.255469 0.158333
1 0.420312 0.395833 0.140625 0.166667
```

### 2. Instance Segmentation

**Format:** One line per object with polygon coordinates

```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

- `class_id`: Integer class ID (0-indexed)
- `x1 y1 ... xn yn`: Polygon vertices (normalized 0-1)

**Example (`img001.txt`):**

```
0 0.681 0.485 0.670 0.487 0.676 0.487 0.685 0.481 0.691 0.474
1 0.408 0.485 0.407 0.485 0.408 0.485 0.408 0.485 0.409 0.484
```

### 3. Pose Estimation

**Format:** One line per person with keypoints

```
class_id x_center y_center width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ... kpn_x kpn_y kpn_v
```

- `class_id`: Integer class ID (usually 0 for "person")
- `x_center`, `y_center`, `width`, `height`: Bounding box (normalized 0-1)
- `kpn_x`, `kpn_y`: Keypoint coordinates (normalized 0-1)
- `kpn_v`: Visibility flag (0=not labeled, 1=labeled but not visible, 2=labeled and visible)

**Example (`img001.txt`):**

```
0 0.640625 0.671875 0.296875 0.609375 0.65 0.46 2 0.68 0.46 2 0.62 0.47 2 ...
```

### 4. Oriented Bounding Boxes (OBB)

**Format:** One line per object with rotated box corners

```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

- `class_id`: Integer class ID (0-indexed)
- `x1 y1 ... x4 y4`: Four corner coordinates of rotated rectangle (normalized 0-1)

**Example (`img001.txt`):**

```
0 0.780811 0.743961 0.782371 0.74686 0.777691 0.752174 0.776131 0.749275
1 0.531250 0.766234 0.539063 0.766234 0.539063 0.784375 0.531250 0.784375
```

### 5. Classification

**Option A:** Class name as directory

```
my-dataset/
├── train/
│   ├── cat/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── dog/
│   │   ├── img101.jpg
│   │   └── ...
│   └── bird/
│       ├── img201.jpg
│       └── ...
└── val/
    ├── cat/
    ├── dog/
    └── bird/
```

**Option B:** Label files with class ID

```
my-dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    │   ├── img001.txt  # contains: 0
    │   ├── img002.txt  # contains: 1
    │   └── ...
    └── val/
```

## Create Dataset YAML Configuration

Create `data.yaml` in your dataset root:

```yaml
# Dataset paths
path: /path/to/my-dataset # dataset root directory
train: images/train # train images (relative to 'path')
val: images/val # val images (relative to 'path')
test: images/test # (optional) test images

# Number of classes
nc: 80

# Class names
names:
    0: person
    1: bicycle
    2: car
    3: motorcycle
    # ... up to nc-1
```

**For Classification:**

```yaml
path: /path/to/my-dataset
train: train
val: val
test: test # optional

names:
    0: cat
    1: dog
    2: bird
    # ... include ALL classes
```

## Convert from Other Formats

### COCO to YOLO

```python
from ultralytics.data.converter import convert_coco

# Convert COCO JSON to YOLO format
convert_coco(
    labels_dir="path/to/coco/annotations/",  # COCO annotations directory
    save_dir="path/to/yolo/labels/",  # output directory
    use_segments=False,  # True for segmentation
    use_keypoints=False,  # True for pose
    cls91to80=True,  # convert 91 COCO classes to 80 classes
)
```

### VOC XML to YOLO

```python
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_voc_to_yolo(xml_file, img_width, img_height, class_names):
    """Convert VOC XML annotation to YOLO format."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    yolo_labels = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in class_names:
            continue
        cls_id = class_names.index(cls_name)

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert to YOLO format (normalized center coordinates + dimensions)
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_labels.append(f"{cls_id} {x_center} {y_center} {width} {height}")

    return "\n".join(yolo_labels)


# Usage
class_names = ["person", "car", "bicycle"]
labels = convert_voc_to_yolo("annotation.xml", 1920, 1080, class_names)
Path("labels/img001.txt").write_text(labels)
```

## Validate Dataset

### Visualize Annotations

```python
from pathlib import Path

import numpy as np
import yaml
from ultralytics.utils.plotting import plot_labels

# Load dataset YAML
with open("data.yaml") as f:
    data = yaml.safe_load(f)

# Collect boxes and class labels from label files
boxes = []
cls = []
labels_dir = Path("path/to/labels/train")

for label_file in labels_dir.glob("*.txt"):
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                boxes.append([x_center, y_center, width, height])
                cls.append(class_id)

# Plot labels distribution
if boxes and cls:
    plot_labels(
        boxes=np.array(boxes),
        cls=np.array(cls),
        names=data.get("names", {}),
        save_dir=Path("plots/")
    )
```

## Dataset Best Practices

1. **Image Quality:**
    - Use consistent image resolution when possible
    - Ensure good lighting and contrast
    - Minimum recommended resolution: 640×640

2. **Dataset Size:**
    - Minimum: 1,500 images per class for good results
    - Recommended: 5,000+ images per class for best results
    - More data = better model performance

3. **Data Split:**
    - Train: 70-80%
    - Validation: 20-30%
    - Test: Optional (10-15% if using)

4. **Annotation Quality:**
    - Ensure accurate bounding boxes/polygons
    - Label all instances in each image
    - Use consistent class labeling
    - Review and fix annotation errors

5. **Data Augmentation:**
    - YOLO applies augmentation automatically during training
    - Mosaic, mixup, flip, rotate, scale are enabled by default

## Common Issues

**Missing Labels:**

- Ensure every image has a corresponding .txt file
- Empty .txt files are valid (no objects in image)

**Incorrect Coordinates:**

- All coordinates must be normalized (0-1 range)
- YOLO uses center-based box representation, not corner-based

**Class ID Errors:**

- Class IDs must be 0-indexed integers
- Ensure class IDs in labels match `data.yaml` class names

## Next Steps

After dataset preparation:

1. Train a model: see `ultralytics-train-model` skill
2. Validate predictions: see `ultralytics-run-inference` skill
3. Consider uploading dataset to the [Ultralytics Platform](https://platform.ultralytics.com) for improved management and sharing

## References

- [Ultralytics Dataset Formats](https://docs.ultralytics.com/datasets/)
- [COCO Dataset](https://docs.ultralytics.com/datasets/detect/coco/)
- [Data Converter Utilities](https://docs.ultralytics.com/reference/data/converter/)
