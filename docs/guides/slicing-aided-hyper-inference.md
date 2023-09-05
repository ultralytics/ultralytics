---
comments: true
description: A comprehensive guide on how to use YOLOv8 with SAHI for standard and sliced inference in object detection tasks.
keywords: YOLOv8, SAHI, Sliced Inference, Object Detection, Ultralytics, Large Scale Image Analysis, High-Resolution Imagery
---

# Ultralytics Docs: Using YOLOv8 with SAHI for Sliced Inference

Welcome to the Ultralytics documentation on how to use YOLOv8 with SAHI (Slicing Aided Hyper Inference). In this comprehensive guide, we'll discuss what SAHI is, the benefits of sliced inference, and how to use SAHI with YOLOv8 for object detection tasks.

![SAHI Sliced Inference](https://raw.githubusercontent.com/obss/sahi/main/resources/sliced_inference.gif)

## Table of Contents

1. [Introduction to SAHI](#introduction-to-sahi)
2. [What is Sliced Inference?](#what-is-sliced-inference)
3. [Installation and Preparation](#installation-and-preparation)
4. [Standard Inference with YOLOv8](#standard-inference-with-yolov8)
5. [Sliced Inference with YOLOv8](#sliced-inference-with-yolov8)
6. [Handling Prediction Results](#handling-prediction-results)
7. [Batch Prediction](#batch-prediction)

## Introduction to SAHI

SAHI is a powerful library aimed at performing efficient and accurate object detection over slices of an image, particularly useful for large scale and high-resolution imagery. It integrates seamlessly with YOLO models and allows for a more efficient usage of computational resources.

## What is Sliced Inference?

Sliced Inference is a technique that divides a large image into smaller slices, performs object detection on each slice, and then aggregates the results back onto the original image. This method is especially beneficial when dealing with high-resolution images as it significantly reduces the computational load without sacrificing detection accuracy.

## Installation and Preparation

### Installation

To get started, install the latest versions of SAHI and Ultralytics:

```bash
pip install -U ultralytics sahi
```

### Import Modules and Download Resources

Here's how to import the necessary modules and download a YOLOv8 model and some test images:

```python
from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from pathlib import Path
from IPython.display import Image

# Download YOLOv8 model
yolov8_model_path = "models/yolov8s.pt"
download_yolov8s_model(yolov8_model_path)

# Download test images
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data/small-vehicles1.jpeg')
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png', 'demo_data/terrain2.png')
```

## Standard Inference with YOLOv8

### Instantiate the Model

You can instantiate a YOLOv8 model for object detection like this:

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.3,
    device="cpu",  # or 'cuda:0'
)
```

### Perform Standard Prediction

Perform standard inference using an image path or a numpy image.

```python
# With an image path
result = get_prediction("demo_data/small-vehicles1.jpeg", detection_model)

# With a numpy image
result = get_prediction(read_image("demo_data/small-vehicles1.jpeg"), detection_model)
```

### Visualize Results

Export and visualize the predicted bounding boxes and masks:

```python
result.export_visuals(export_dir="demo_data/")
Image("demo_data/prediction_visual.png")
```

## Sliced Inference with YOLOv8

Perform sliced inference by specifying the slice dimensions and overlap ratios:

```python
result = get_sliced_prediction(
    "demo_data/small-vehicles1.jpeg",
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)
```

## Handling Prediction Results

SAHI provides a `PredictionResult` object, which can be converted into various annotation formats:

```python
# Access the object prediction list
object_prediction_list = result.object_prediction_list

# Convert to COCO annotation, COCO prediction, imantics, and fiftyone formats
result.to_coco_annotations()[:3]
result.to_coco_predictions(image_id=1)[:3]
result.to_imantics_annotations()[:3]
result.to_fiftyone_detections()[:3]
```

## Batch Prediction

For batch prediction on a directory of images:

```python
predict(
    model_type="yolov8",
    model_path="path/to/yolov8n.pt",
    model_device="cpu",  # or 'cuda:0'
    model_confidence_threshold=0.4,
    source="path/to/dir",
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

That's it! Now you're equipped to use YOLOv8 with SAHI for both standard and sliced inference.
