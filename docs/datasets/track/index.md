---
comments: true
description: Understand multi-object tracking datasets, upcoming features and how to use them with YOLO in Python and CLI. Dive in now!.
keywords: Ultralytics, YOLO, multi-object tracking, datasets, detection, segmentation, pose models, Python, CLI
---

# Multi-object Tracking Datasets Overview

## Dataset Format (Coming Soon)

Multi-Object Detector doesn't need standalone training and directly supports pre-trained detection, segmentation or Pose models.
Support for training trackers alone is coming soon

## Usage

!!! example ""

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/Zgi9g1ksQHc", conf=0.3, iou=0.5, show=True)
        ```
    === "CLI"

        ```bash
        yolo track model=yolov8n.pt source="https://youtu.be/Zgi9g1ksQHc" conf=0.3, iou=0.5 show
        ```
