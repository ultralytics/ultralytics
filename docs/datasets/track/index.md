---
comments: true
description: Discover the datasets compatible with Multi-Object Detector. Train your trackers and make your detections more efficient with Ultralytics' YOLO.
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