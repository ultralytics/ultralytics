---
comments: true
description: Learn how to create, annotate, and prepare custom datasets for YOLO11 training, with step-by-step instructions and examples.
keywords: Ultralytics, YOLO11, custom dataset, object detection, segmentation, pose estimation, multi-object tracking, data preparation, annotation, training, guide, tutorial
---

# Create and Prepare Custom Datasets for YOLO

## Introduction

A well-structured [dataset](/datasets/index.md) is the cornerstone of effective object detection with YOLO. In this guide, we’ll walk you through every step to create, annotate, and prepare your custom dataset for YOLO11 training. By the end, you’ll know how to structure your dataset, annotate your images, apply augmentations, and train your model confidently.

<p align="center">
  <br>
  <iframe width="560" height="315" src="https://www.youtube.com/embed/ZN3nRZT7b24?si=EW65YwT9ZUY7m8kz" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
  <br>
  <strong>Watch:</strong> How to Train Ultralytics YOLO11 Model on Custom Dataset using Google Colab Notebook
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</p>

## Data Collection

Begin by gathering a diverse set of images that represent the objects or scenes you aim to detect. Ensure variability in lighting conditions, angles, and backgrounds to enhance the model's robustness. You can source images from public datasets, capture your own, or utilize platforms like [Roboflow](https://docs.ultralytics.com/integrations/roboflow/) to streamline the process.

## Data Annotation

Accurate annotations are crucial for training. Use tools such as [CVAT](https://github.com/cvat-ai/cvat), [LabelImg](https://github.com/HumanSignal/labelImg), or [Roboflow](https://docs.ultralytics.com/integrations/roboflow/) to label your images. For object detection tasks, draw bounding boxes around objects and assign appropriate class labels. Ensure consistency in labeling to maintain dataset quality.

_Quick Tip:_ Checkout this comprehensive guide on [data collection and annotation strategies](https://docs.ultralytics.com/guides/data-collection-and-annotation/).

## Export Annotations

After annotating, export the annotations in the YOLO format, which consists of a .txt file for each image. Each line in the file corresponds to an object, formatted as

```
<class_id> <x_center> <y_center> <width> <height>
```

This format is essential for YOLO training.

## Organize Dataset Structure

A clean, consistent folder structure is key to avoiding errors during training.

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image3.jpg
│       ├── image4.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   └── val/
│       ├── image3.txt
│       ├── image4.txt
│       └── ...
└── dataset.yaml
```

Ensure that each image has a corresponding .txt annotation file in the respective labels directory.

## Create Dataset Configuration File (dataset.yaml)

YOLO needs to know some basic metadata about your dataset. The dataset.yaml file tells the model where your images and labels are, and what classes exist.

```yaml
# dataset.yaml
path: ../datasets/custom # dataset root directory (relative to training script)
train: images/train # train images (relative to 'path')
val: images/val # validation images (relative to 'path')

# Classes
nc: 3 # number of classes
names: # class names
    0: person
    1: car
    2: dog
```

This file specifies the paths to training and validation images, the number of classes (nc), and the class names.

## Annotation Format

### YOLO Annotation Format

YOLO uses normalized coordinates in text files. Each line represents one object:

```
class_id x_center y_center width height
```

**Example annotations:**

```
0 0.5 0.5 0.2 0.3    # person at center, 20% width, 30% height
1 0.8 0.6 0.15 0.25  # car at right side, 15% width, 25% height
```

**Key points:**

- **class_id**: Integer starting from 0
- **x_center, y_center**: Center coordinates (0.0 to 1.0)
- **width, height**: Bounding box dimensions (0.0 to 1.0)
- **Normalized**: All values are relative to image dimensions

### Converting from Other Formats

If your dataset is in another format like COCO or Pascal VOC, you can convert it to YOLO format easily.

#### From COCO Format

```python
from ultralytics.data.converter import convert_coco

# Convert COCO annotations to YOLO format
convert_coco(
    labels_dir="path/to/coco/labels",
    save_dir="path/to/yolo/labels",
    classes=["person", "car", "dog"],  # your class names
)
```

Find the documentation [data converter here](https://docs.ultralytics.com/reference/data/converter/#ultralytics.data.converter.convert_coco).

#### From Pascal VOC

```python
from ultralytics.data.converter import convert_voc

# Convert Pascal VOC annotations to YOLO format
convert_voc(labels_dir="path/to/voc/labels", save_dir="path/to/yolo/labels", classes=["person", "car", "dog"])
```

Find more about [VOC dataset](https://docs.ultralytics.com/datasets/detect/voc/) here.

### Quality Control

Before training, always check your dataset for errors. YOLO provides a simple utility:

```python
from ultralytics.data.utils import check_dataset

# Validate your dataset structure and annotations
check_dataset("path/to/dataset.yaml")
```

## Data Augmentation

Data augmentation is a technique used to artificially expand the size of a dataset by creating modified versions of images in the dataset. This helps improve the robustness and generalization of the model.

Find out more about [data augmentation for YOLO model training](https://docs.ultralytics.com/guides/yolo-data-augmentation/) in the documentation.

### Built-in YOLO Augmentation

Ultralytics YOLO provides built-in augmentation during training. You can configure it in your training script:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Train with custom augmentation settings
results = model.train(
    data="dataset.yaml",
    epochs=100,
    augment=True,  # Enable augmentation
    degrees=10.0,  # Image rotation (+/- degrees)
    translate=0.1,  # Image translation (+/- fraction)
    scale=0.5,  # Image scale (+/- fraction)
    shear=0.0,  # Image shear (+/- degrees)
    perspective=0.0,  # Image perspective (+/- fraction)
    flipud=0.0,  # Image flip up-down (probability)
    fliplr=0.5,  # Image flip left-right (probability)
    mosaic=1.0,  # Image mosaic (probability)
    mixup=0.0,  # Image mixup (probability)
    copy_paste=0.0,  # Segment copy-paste (probability)
)
```

### External Augmentation Tools

#### Using Albumentations

[Albumentations](https://docs.ultralytics.com/integrations/albumentations/) is a versatile Python library for image augmentation that supports transformations like rotation, brightness/contrast adjustments, blur, and geometric distortions. Using Albumentations with YOLO11 helps create a more diverse and realistic dataset, which improves your model’s robustness and generalization to unseen data.

**Why use it:** It’s ideal when you want fine-grained control over augmentation strategies and need to simulate a wide variety of real-world conditions for your training images.

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define augmentation pipeline
transform = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.OneOf(
            [
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ],
            p=0.2,
        ),
        A.Normalize(),
        ToTensorV2(),
    ]
)

# Apply to images
augmented = transform(image=image, bboxes=bboxes, labels=labels)
```

#### Using Roboflow

[Roboflow](https://docs.ultralytics.com/integrations/roboflow/) is a platform for dataset management and augmentation that lets you upload images, apply automated transformations (rotation, brightness, noise, etc.), and export them in YOLO-ready formats.

**Why use it:** It’s perfect for quickly preparing high-quality, consistent datasets without writing custom augmentation code, saving time while ensuring your data is robust and diverse.

```python
from roboflow import Roboflow

rf = Roboflow(api_key="your_api_key")
project = rf.workspace("your_workspace").project("your_project")

# Generate augmented dataset
project.generate(
    num_images=1000,
    augmentation={
        "rotation": {"min": -15, "max": 15},
        "brightness": {"min": 0.8, "max": 1.2},
        "noise": {"min": 0, "max": 0.1},
    },
)
```

## Dataset Splitting

### Train/Validation Split

Split your dataset into training and validation sets:

```python
from ultralytics.data.utils import autosplit

# Automatically split dataset into train/val
autosplit(
    path="path/to/dataset",
    weights=(0.8, 0.2),  # (train, val) split
    annotated_only=True,  # Only split images with annotations
)
```

### Manual Splitting

```python
import os
import shutil

from sklearn.model_selection import train_test_split


def split_dataset(images_dir, labels_dir, train_ratio=0.8):
    """Split dataset into train and validation sets."""
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

    # Split into train and val
    train_files, val_files = train_test_split(image_files, train_size=train_ratio, random_state=42)

    # Create directories
    os.makedirs("train/images", exist_ok=True)
    os.makedirs("train/labels", exist_ok=True)
    os.makedirs("val/images", exist_ok=True)
    os.makedirs("val/labels", exist_ok=True)

    # Move files
    for file in train_files:
        base_name = os.path.splitext(file)[0]
        shutil.copy(f"{images_dir}/{file}", f"train/images/{file}")
        if os.path.exists(f"{labels_dir}/{base_name}.txt"):
            shutil.copy(f"{labels_dir}/{base_name}.txt", f"train/labels/{base_name}.txt")

    for file in val_files:
        base_name = os.path.splitext(file)[0]
        shutil.copy(f"{images_dir}/{file}", f"val/images/{file}")
        if os.path.exists(f"{labels_dir}/{base_name}.txt"):
            shutil.copy(f"{labels_dir}/{base_name}.txt", f"val/labels/{base_name}.txt")
```

## Training with Custom Dataset

Once your dataset is prepared, you can train your YOLO model:

### Basic Training

```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolo11n.pt")

# Train on your custom dataset
results = model.train(data="dataset.yaml", epochs=100, imgsz=640, batch=16, name="custom_model")
```

### Advanced Training Configuration

```python
# Custom training configuration
results = model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="custom_model",
    # Optimization
    lr0=0.01,  # Initial learning rate
    lrf=0.01,  # Final learning rate
    momentum=0.937,  # SGD momentum/Adam beta1
    weight_decay=0.0005,  # Optimizer weight decay
    warmup_epochs=3,  # Warmup epochs
    warmup_momentum=0.8,  # Warmup initial momentum
    warmup_bias_lr=0.1,  # Warmup initial bias lr
    # Augmentation
    augment=True,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    # Validation
    val=True,
    save_period=10,  # Save checkpoint every x epochs
    # Logging
    plots=True,  # Save plots
    save=True,  # Save checkpoints
    verbose=True,
)
```

For advanced training configurations, refer to the [Ultralytics YOLO training guide](https://docs.ultralytics.com/modes/train/).

## Validation and Testing

### Validation During Training

```python
# Validate after training
results = model.val(data="dataset.yaml")
```

### Custom Validation

```python
# Validate on specific data
results = model.val(
    data="dataset.yaml",
    split="val",
    imgsz=640,
    batch=16,
    conf=0.001,  # Confidence threshold
    iou=0.6,  # NMS IoU threshold
    max_det=300,  # Maximum detections per image
    verbose=True,
)
```

For advanced model validation configurations, refer to the [Ultralytics YOLO validation guide](https://docs.ultralytics.com/modes/val/).

## FAQ

### How many images do I need per class?

- **Minimum**: 100 images per class
- **Recommended**: 1000+ images per class
- **Optimal**: 2000+ images per class with diverse conditions

### What image formats are supported?

YOLO supports JPG, PNG, JPEG. Avoid WebP, TIFF, and other formats.

### How do I handle class imbalance?

Use weighted loss, data augmentation, or collect more samples for underrepresented classes.

### Can I use images of different sizes?

Yes, but ensure minimum resolution of 640x640. YOLO will automatically resize images during training.

### How do I validate my dataset structure?

Use `ultralytics.data.utils.check_dataset('dataset.yaml')` to validate your dataset.
