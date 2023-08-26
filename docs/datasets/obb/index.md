---
comments: true
description: Dive deep into various oriented bounding box (OBB) dataset formats compatible with Ultralytics YOLO models. Grasp the nuances of using and converting datasets to this format.
keywords: Ultralytics, YOLO, oriented bounding boxes, OBB, dataset formats, label formats, DOTA v2, data conversion
---

# Oriented Bounding Box (OBB) Datasets Overview

Training a precise object detection model with oriented bounding boxes (OBB) requires a thorough dataset. This guide explains the various OBB dataset formats compatible with Ultralytics YOLO models, offering insights into their structure, application, and methods for format conversions.

## Supported OBB Dataset Formats

### YOLO OBB Format

The YOLO OBB format designates bounding boxes by their four corner points with coordinates normalized between 0 and 1. It follows this format:

```bash
class_index, x1, y1, x2, y2, x3, y3, x4, y4
```

Internally, YOLO processes losses and outputs in the `xywhr` format, which represents the bounding box's center point (xy), width, height, and rotation.

<p align="center"><img width="800" src="https://user-images.githubusercontent.com/26833433/259471881-59020fe2-09a4-4dcc-acce-9b0f7cfa40ee.png"></p>

An example of a `*.txt` label file for the above image, which contains an object of class `0` in OBB format, could look like:

```bash
0 0.780811 0.743961 0.782371 0.74686 0.777691 0.752174 0.776131 0.749758
```

## Usage

To train a model using these OBB formats:

!!! example ""

    === "Python"

        ```python
        from ultralytics import YOLO

        # Create a new YOLOv8n-OBB model from scratch
        model = YOLO('yolov8n-obb.yaml')

        # Train the model on the DOTAv2 dataset
        results = model.train(data='DOTAv2.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Train a new YOLOv8n-OBB model on the DOTAv2 dataset
        yolo detect train data=DOTAv2.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Supported Datasets

Currently, the following datasets with Oriented Bounding Boxes are supported:

- [**DOTA v2**](./dota-v2.md): DOTA (A Large-scale Dataset for Object Detection in Aerial Images) version 2, emphasizes detection from aerial perspectives and contains oriented bounding boxes with 1.7 million instances and 11,268 images.

### Incorporating your own OBB dataset

For those looking to introduce their own datasets with oriented bounding boxes, ensure compatibility with the "YOLO OBB format" mentioned above. Convert your annotations to this required format and detail the paths, classes, and class names in a corresponding YAML configuration file.

## Convert Label Formats

### DOTA Dataset Format to YOLO OBB Format

Transitioning labels from the DOTA dataset format to the YOLO OBB format can be achieved with this script:

!!! example ""

    === "Python"

        ```python
        from ultralytics.data.converter import convert_dota_to_yolo_obb
        
        convert_dota_to_yolo_obb('path/to/DOTA')
        ```

This conversion mechanism is instrumental for datasets in the DOTA format, ensuring alignment with the Ultralytics YOLO OBB format.

It's imperative to validate the compatibility of the dataset with your model and adhere to the necessary format conventions. Properly structured datasets are pivotal for training efficient object detection models with oriented bounding boxes.
