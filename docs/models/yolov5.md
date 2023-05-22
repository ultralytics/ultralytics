---
comments: true
description: YOLOv5u by Ultralytics explained. Discover the evolution of this model and its key specifications. Experience faster and more accurate object detection.
---

# YOLOv5u

## Overview

YOLOv5u is an enhanced version of the [YOLOv5](https://github.com/ultralytics/yolov5) object detection model from Ultralytics. This iteration incorporates the anchor-free, objectness-free split head that is featured in the [YOLOv8](./yolov8.md) models. Although it maintains the same backbone and neck architecture as YOLOv5, YOLOv5u provides an improved accuracy-speed tradeoff for object detection tasks, making it a robust choice for numerous applications.

## Key Features

- **Anchor-free Split Ultralytics Head:** YOLOv5u replaces the conventional anchor-based detection head with an anchor-free split Ultralytics head, boosting performance in object detection tasks.

- **Optimized Accuracy-Speed Tradeoff:** By delivering a better balance between accuracy and speed, YOLOv5u is suitable for a diverse range of real-time applications, from autonomous driving to video surveillance.

- **Variety of Pre-trained Models:** YOLOv5u includes numerous pre-trained models for tasks like Inference, Validation, and Training, providing the flexibility to tackle various object detection challenges.

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

## Usage

You can use YOLOv5u for object detection tasks using the Ultralytics repository. The following is a sample code snippet showing how to use YOLOv5u model for inference:

```python
from ultralytics import YOLO

# Load the model
model = YOLO('yolov5n.pt')  # load a pretrained model

# Perform inference
results = model('image.jpg')

# Print the results
results.print()
```

## Citations and Acknowledgments

If you use YOLOv5 or YOLOv5u in your research, please cite the Ultralytics YOLOv5 repository as follows:

```bibtex
@software{yolov5,
  title = {YOLOv5 by Ultralytics},
  author = {Glenn Jocher},
  year = {2020},
  version = {7.0},
  license = {AGPL-3.0},
  url = {https://github.com/ultralytics/yolov5},
  doi = {10.5281/zenodo.3908559},
  orcid = {0000-0001-5950-6979}
}
```

Special thanks to Glenn Jocher and the Ultralytics team for their work on developing and maintaining the YOLOv5 and YOLOv5u models.