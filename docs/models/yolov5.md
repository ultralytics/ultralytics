---
comments: true
description: Detect objects faster and more accurately using Ultralytics YOLOv5u. Find pre-trained models for each task, including Inference, Validation and Training.
---

# YOLOv5u

## Overview

YOLOv5u is an updated version of YOLOv5 that incorporates the anchor-free split Ultralytics head used in the YOLOv8 models. It retains the same backbone and neck architecture as YOLOv5 but offers improved accuracy-speed tradeoff for object detection tasks.

## Key Features

- **Anchor-free Split Ultralytics Head:** YOLOv5u replaces the traditional anchor-based detection head with an anchor-free split Ultralytics head, resulting in improved performance.
- **Optimized Accuracy-Speed Tradeoff:** The updated model offers a better balance between accuracy and speed, making it more suitable for a wider range of applications.
- **Variety of Pre-trained Models:** YOLOv5u offers a range of pre-trained models tailored for various tasks, including Inference, Validation, and Training.

## Supported Tasks

| Model Type | Pre-trained Weights                                                                                                         | Task      |
|------------|-----------------------------------------------------------------------------------------------------------------------------|-----------|
| YOLOv5u    | `yolov5nu`, `yolov5su`, `yolov5mu`, `yolov5lu`, `yolov5xu`, `yolov5n6u`, `yolov5s6u`, `yolov5m6u`, `yolov5l6u`, `yolov5x6u` | Detection |

## Supported Modes

| Mode       | Supported          |
|------------|--------------------|
| Inference  | :heavy_check_mark: |
| Validation | :heavy_check_mark: |
| Training   | :heavy_check_mark: |

??? Performance

    === "Detection"

        | Model                                                                                    | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
        | ---------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv5nu](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5nu.pt)   | 640                   | 34.3                 | 73.6                           | 1.06                                | 2.6                | 7.7               |
        | [YOLOv5su](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5su.pt)   | 640                   | 43.0                 | 120.7                          | 1.27                                | 9.1                | 24.0              |
        | [YOLOv5mu](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5mu.pt)   | 640                   | 49.0                 | 233.9                          | 1.86                                | 25.1               | 64.2              |
        | [YOLOv5lu](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5lu.pt)   | 640                   | 52.2                 | 408.4                          | 2.50                                | 53.2               | 135.0             |
        | [YOLOv5xu](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5xu.pt)   | 640                   | 53.2                 | 763.2                          | 3.81                                | 97.2               | 246.4             |
        |                                                                                          |                       |                      |                                |                                     |                    |                   |
        | [YOLOv5n6u](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5n6u.pt) | 1280                  | 42.1                 | -                              | -                                   | 4.3                | 7.8               |
        | [YOLOv5s6u](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5s6u.pt) | 1280                  | 48.6                 | -                              | -                                   | 15.3               | 24.6              |
        | [YOLOv5m6u](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5m6u.pt) | 1280                  | 53.6                 | -                              | -                                   | 41.2               | 65.7              |
        | [YOLOv5l6u](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5l6u.pt) | 1280                  | 55.7                 | -                              | -                                   | 86.1               | 137.4             |
        | [YOLOv5x6u](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5x6u.pt) | 1280                  | 56.8                 | -                              | -                                   | 155.4              | 250.7             |