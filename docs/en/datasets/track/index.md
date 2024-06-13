---
comments: true
description: Learn how to use Multi-Object Tracking with YOLO. Explore dataset formats and see upcoming features for training trackers. Start with Python or CLI examples.
keywords: YOLO, Multi-Object Tracking, Tracking Datasets, Python Tracking Example, CLI Tracking Example, Object Detection, Ultralytics, AI, Machine Learning
---

# Multi-object Tracking Datasets Overview

## Dataset Format (Coming Soon)

Multi-Object Detector doesn't need standalone training and directly supports pre-trained detection, segmentation or Pose models. Support for training trackers alone is coming soon

## Usage

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
        ```
    === "CLI"

        ```bash
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.3, iou=0.5 show
        ```
