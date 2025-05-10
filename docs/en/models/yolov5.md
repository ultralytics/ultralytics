---
comments: true
description: Explore YOLOv5u, an advanced object detection model with optimized accuracy-speed tradeoff, featuring anchor-free Ultralytics head and various pre-trained models.
keywords: YOLOv5, YOLOv5u, object detection, Ultralytics, anchor-free, pre-trained models, accuracy, speed, real-time detection
---

# Ultralytics YOLOv5

## Overview

YOLOv5u represents an advancement in [object detection](https://www.ultralytics.com/glossary/object-detection) methodologies. Originating from the foundational architecture of the [YOLOv5](https://github.com/ultralytics/yolov5) model developed by Ultralytics, YOLOv5u integrates the anchor-free, objectness-free split head, a feature previously introduced in the [YOLOv8](yolov8.md) models. This adaptation refines the model's architecture, leading to an improved accuracy-speed tradeoff in object detection tasks. Given the empirical results and its derived features, YOLOv5u provides an efficient alternative for those seeking robust solutions in both research and practical applications.

![Ultralytics YOLOv5](https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov5-splash.avif)

## Key Features

- **Anchor-free Split Ultralytics Head:** Traditional object detection models rely on predefined anchor boxes to predict object locations. However, YOLOv5u modernizes this approach. By adopting an anchor-free split Ultralytics head, it ensures a more flexible and adaptive detection mechanism, consequently enhancing the performance in diverse scenarios.

- **Optimized Accuracy-Speed Tradeoff:** Speed and accuracy often pull in opposite directions. But YOLOv5u challenges this tradeoff. It offers a calibrated balance, ensuring real-time detections without compromising on accuracy. This feature is particularly invaluable for applications that demand swift responses, such as [autonomous vehicles](https://www.ultralytics.com/glossary/autonomous-vehicles), [robotics](https://www.ultralytics.com/glossary/robotics), and real-time video analytics.

- **Variety of Pre-trained Models:** Understanding that different tasks require different toolsets, YOLOv5u provides a plethora of pre-trained models. Whether you're focusing on Inference, Validation, or Training, there's a tailor-made model awaiting you. This variety ensures you're not just using a one-size-fits-all solution, but a model specifically fine-tuned for your unique challenge.

## Supported Tasks and Modes

The YOLOv5u models, with various pre-trained weights, excel in [Object Detection](../tasks/detect.md) tasks. They support a comprehensive range of modes, making them suitable for diverse applications, from development to deployment.

| Model Type | Pre-trained Weights                                                                                                         | Task                                   | Inference | Validation | Training | Export |
| ---------- | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | --------- | ---------- | -------- | ------ |
| YOLOv5u    | `yolov5nu`, `yolov5su`, `yolov5mu`, `yolov5lu`, `yolov5xu`, `yolov5n6u`, `yolov5s6u`, `yolov5m6u`, `yolov5l6u`, `yolov5x6u` | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |

This table provides a detailed overview of the YOLOv5u model variants, highlighting their applicability in object detection tasks and support for various operational modes such as [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), and [Export](../modes/export.md). This comprehensive support ensures that users can fully leverage the capabilities of YOLOv5u models in a wide range of object detection scenarios.

## Performance Metrics

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLOv5"]'></canvas>

!!! tip "Performance"

    === "Detection (COCO)"

    See [Detection Docs](../tasks/detect.md) for usage examples with these models trained on [COCO](../datasets/detect/coco.md), which include 80 pre-trained classes.

    | Model                                                                                       | YAML                                                                                                           | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
    |---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-----------------------|----------------------|--------------------------------|-------------------------------------|--------------------|-------------------|
    | [yolov5nu.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5nu.pt)   | [yolov5n.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 34.3                 | 73.6                           | 1.06                                | 2.6                | 7.7               |
    | [yolov5su.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5su.pt)   | [yolov5s.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 43.0                 | 120.7                          | 1.27                                | 9.1                | 24.0              |
    | [yolov5mu.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5mu.pt)   | [yolov5m.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 49.0                 | 233.9                          | 1.86                                | 25.1               | 64.2              |
    | [yolov5lu.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5lu.pt)   | [yolov5l.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 52.2                 | 408.4                          | 2.50                                | 53.2               | 135.0             |
    | [yolov5xu.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5xu.pt)   | [yolov5x.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 53.2                 | 763.2                          | 3.81                                | 97.2               | 246.4             |
    |                                                                                             |                                                                                                                |                       |                      |                                |                                     |                    |                   |
    | [yolov5n6u.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5n6u.pt) | [yolov5n6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 42.1                 | 211.0                          | 1.83                                | 4.3                | 7.8               |
    | [yolov5s6u.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5s6u.pt) | [yolov5s6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 48.6                 | 422.6                          | 2.34                                | 15.3               | 24.6              |
    | [yolov5m6u.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5m6u.pt) | [yolov5m6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 53.6                 | 810.9                          | 4.36                                | 41.2               | 65.7              |
    | [yolov5l6u.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5l6u.pt) | [yolov5l6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 55.7                 | 1470.9                         | 5.47                                | 86.1               | 137.4             |
    | [yolov5x6u.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5x6u.pt) | [yolov5x6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 56.8                 | 2436.5                         | 8.98                                | 155.4              | 250.7             |

## Usage Examples

This example provides simple YOLOv5 training and inference examples. For full documentation on these and other [modes](../modes/index.md) see the [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) and [Export](../modes/export.md) docs pages.

!!! example

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) pretrained `*.pt` models as well as configuration `*.yaml` files can be passed to the `YOLO()` class to create a model instance in python:

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLOv5n model
        model = YOLO("yolov5n.pt")

        # Display model information (optional)
        model.info()

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # Run inference with the YOLOv5n model on the 'bus.jpg' image
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        CLI commands are available to directly run the models:

        ```bash
        # Load a COCO-pretrained YOLOv5n model and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolov5n.pt data=coco8.yaml epochs=100 imgsz=640

        # Load a COCO-pretrained YOLOv5n model and run inference on the 'bus.jpg' image
        yolo predict model=yolov5n.pt source=path/to/bus.jpg
        ```

## Citations and Acknowledgements

!!! tip "Ultralytics YOLOv5 Publication"

    Ultralytics has not published a formal research paper for YOLOv5 due to the rapidly evolving nature of the models. We focus on advancing the technology and making it easier to use, rather than producing static documentation. For the most up-to-date information on YOLO architecture, features, and usage, please refer to our [GitHub repository](https://github.com/ultralytics/ultralytics) and [documentation](https://docs.ultralytics.com/).

If you use YOLOv5 or YOLOv5u in your research, please cite the Ultralytics YOLOv5 repository as follows:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @software{yolov5,
          title = {Ultralytics YOLOv5},
          author = {Glenn Jocher},
          year = {2020},
          version = {7.0},
          license = {AGPL-3.0},
          url = {https://github.com/ultralytics/yolov5},
          doi = {10.5281/zenodo.3908559},
          orcid = {0000-0001-5950-6979}
        }
        ```

Please note that YOLOv5 models are provided under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) and [Enterprise](https://www.ultralytics.com/license) licenses.

## FAQ

### What is Ultralytics YOLOv5u and how does it differ from YOLOv5?

Ultralytics YOLOv5u is an advanced version of YOLOv5, integrating the anchor-free, objectness-free split head that enhances the [accuracy](https://www.ultralytics.com/glossary/accuracy)-speed tradeoff for real-time object detection tasks. Unlike the traditional YOLOv5, YOLOv5u adopts an anchor-free detection mechanism, making it more flexible and adaptive in diverse scenarios. For more detailed information on its features, you can refer to the [YOLOv5 Overview](#overview).

### How does the anchor-free Ultralytics head improve object detection performance in YOLOv5u?

The anchor-free Ultralytics head in YOLOv5u improves object detection performance by eliminating the dependency on predefined anchor boxes. This results in a more flexible and adaptive detection mechanism that can handle various object sizes and shapes with greater efficiency. This enhancement directly contributes to a balanced tradeoff between accuracy and speed, making YOLOv5u suitable for real-time applications. Learn more about its architecture in the [Key Features](#key-features) section.

### Can I use pre-trained YOLOv5u models for different tasks and modes?

Yes, you can use pre-trained YOLOv5u models for various tasks such as [Object Detection](../tasks/detect.md). These models support multiple modes, including [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), and [Export](../modes/export.md). This flexibility allows users to leverage the capabilities of YOLOv5u models across different operational requirements. For a detailed overview, check the [Supported Tasks and Modes](#supported-tasks-and-modes) section.

### How do the performance metrics of YOLOv5u models compare on different platforms?

The performance metrics of YOLOv5u models vary depending on the platform and hardware used. For example, the YOLOv5nu model achieves a 34.3 mAP on COCO dataset with a speed of 73.6 ms on CPU (ONNX) and 1.06 ms on A100 TensorRT. Detailed performance metrics for different YOLOv5u models can be found in the [Performance Metrics](#performance-metrics) section, which provides a comprehensive comparison across various devices.

### How can I train a YOLOv5u model using the Ultralytics Python API?

You can train a YOLOv5u model by loading a pre-trained model and running the training command with your dataset. Here's a quick example:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLOv5n model
        model = YOLO("yolov5n.pt")

        # Display model information (optional)
        model.info()

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Load a COCO-pretrained YOLOv5n model and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolov5n.pt data=coco8.yaml epochs=100 imgsz=640
        ```

For more detailed instructions, visit the [Usage Examples](#usage-examples) section.
