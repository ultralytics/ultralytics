---
comments: true
description: "YOLOv3, YOLOv3-Ultralytics, and YOLOv3u compared: multiscale detection, the anchor-free YOLOv8 head in YOLOv3u, supported modes, and usage examples."
keywords: YOLOv3, YOLOv3-Ultralytics, YOLOv3u, object detection, Ultralytics, computer vision, AI models, deep learning
---

# YOLOv3 and YOLOv3u

## Overview

This document presents an overview of three closely related object detection models, namely [YOLOv3](https://pjreddie.com/darknet/yolo/), [YOLOv3-Ultralytics](https://github.com/ultralytics/yolov3), and [YOLOv3u](https://github.com/ultralytics/ultralytics).

1. **YOLOv3:** This is the [third version](https://www.ultralytics.com/blog/the-evolution-of-object-detection-and-ultralytics-yolo-models) of the You Only Look Once (YOLO) object detection algorithm. Originally developed by Joseph Redmon, YOLOv3 improved on its predecessors by introducing predictions at three different scales.

2. **YOLOv3-Ultralytics:** This is the Ultralytics PyTorch implementation of YOLOv3, maintained in the [ultralytics/yolov3](https://github.com/ultralytics/yolov3) repository. It keeps the original [anchor-based](https://www.ultralytics.com/glossary/anchor-based-detectors) architecture while adding more pretrained models and easier customization.

3. **YOLOv3u:** This is an updated version of YOLOv3-Ultralytics that incorporates the [anchor-free](https://www.ultralytics.com/glossary/anchor-free-detectors), objectness-free split head used in YOLOv8 models. YOLOv3u maintains the same [backbone](https://www.ultralytics.com/glossary/backbone) and neck architecture as YOLOv3 but with the updated [detection head](https://www.ultralytics.com/glossary/detection-head) from YOLOv8.

![YOLOv3 object detection model banner](https://cdn.ul.run/i/86bbfe2f973e8746de786a46f9ccaf5d.avif)

## Key Features

- **YOLOv3:** Introduced detection at three different scales, with 13x13, 26x26, and 52x52 output grids for a 416x416 input. This significantly improved detection accuracy for objects of different sizes. Additionally, YOLOv3 added multi-label predictions for each [bounding box](https://www.ultralytics.com/glossary/bounding-box) and a stronger feature extractor network, Darknet-53.

- **YOLOv3u:** This updated model incorporates the anchor-free, objectness-free split head from YOLOv8. By eliminating the need for pre-defined [anchor boxes](https://www.ultralytics.com/glossary/anchor-boxes) and objectness scores, this detection head design can improve the model's ability to detect objects of varying sizes and shapes. This makes YOLOv3u more robust and accurate for object detection tasks.

## Supported Tasks and Modes

YOLOv3 is designed specifically for [object detection](https://www.ultralytics.com/glossary/object-detection) tasks. Ultralytics supports three variants of YOLOv3: `yolov3u`, `yolov3-tinyu` and `yolov3-sppu`. The `u` in the name signifies that these utilize the anchor-free head of YOLOv8, unlike their original architecture which is anchor-based. The variants trade off accuracy and speed: YOLOv3-Tiny is the smallest and fastest, while YOLOv3-SPP adds spatial pyramid pooling to the standard YOLOv3 model.

All three models support the [Training](../modes/train.md), [Validation](../modes/val.md), [Inference](../modes/predict.md), and [Export](../modes/export.md) modes, covering every stage from development to [model deployment](https://www.ultralytics.com/glossary/model-deployment).

| Model Type     | Pretrained Weights | Tasks Supported                        | Training | Validation | Inference | Export |
| -------------- | ------------------ | -------------------------------------- | -------- | ---------- | --------- | ------ |
| YOLOv3(u)      | `yolov3u.pt`       | [Object Detection](../tasks/detect.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOv3-Tiny(u) | `yolov3-tinyu.pt`  | [Object Detection](../tasks/detect.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOv3-SPP(u)  | `yolov3-sppu.pt`   | [Object Detection](../tasks/detect.md) | ✅       | ✅         | ✅        | ✅     |

This table summarizes the supported tasks and modes for each YOLOv3 variant.

## Usage Examples

This example provides simple YOLOv3 [training](../modes/train.md) and [inference](../modes/predict.md) examples.

!!! example

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) pretrained `*.pt` models as well as configuration `*.yaml` files can be passed to the `YOLO()` class to create a model instance in Python:

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLOv3u model
        model = YOLO("yolov3u.pt")

        # Display model information (optional)
        model.info()

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # Run inference with the YOLOv3u model on the 'bus.jpg' image
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        CLI commands are available to directly run the models:

        ```bash
        # Load a COCO-pretrained YOLOv3u model and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolov3u.pt data=coco8.yaml epochs=100 imgsz=640

        # Load a COCO-pretrained YOLOv3u model and run inference on the 'bus.jpg' image
        yolo predict model=yolov3u.pt source=path/to/bus.jpg
        ```

## Citations and Acknowledgments

If you use YOLOv3 in your research, please cite the original YOLO papers and the Ultralytics YOLOv3 repository:

!!! quote "Citation"

    === "BibTeX"

        ```bibtex
        @article{redmon2018yolov3,
          title={YOLOv3: An Incremental Improvement},
          author={Redmon, Joseph and Farhadi, Ali},
          journal={arXiv preprint arXiv:1804.02767},
          year={2018}
        }
        ```

Thank you to Joseph Redmon and Ali Farhadi for developing the original YOLOv3.

## FAQ

### What are the differences between YOLOv3, YOLOv3-Ultralytics, and YOLOv3u?

YOLOv3 is the third iteration of the YOLO (You Only Look Once) object detection algorithm developed by Joseph Redmon, known for its balance of [accuracy](https://www.ultralytics.com/glossary/accuracy) and speed, predicting at three different scales (13x13, 26x26, and 52x52 grids for a 416x416 input). YOLOv3-Ultralytics is Ultralytics' adaptation of YOLOv3 that adds support for more pretrained models and facilitates easier model customization. YOLOv3u is an upgraded variant of YOLOv3-Ultralytics, integrating the anchor-free, objectness-free split head from YOLOv8, improving detection robustness and accuracy for various object sizes. For more details on the variants, refer to the [YOLOv3 series](https://github.com/ultralytics/yolov3).

### How can I train a YOLOv3 model using Ultralytics?

You can train a YOLOv3u model with either Python or the CLI:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLOv3u model
        model = YOLO("yolov3u.pt")

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Load a COCO-pretrained YOLOv3u model and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolov3u.pt data=coco8.yaml epochs=100 imgsz=640
        ```

For more comprehensive training options and guidelines, visit our [Train mode documentation](../modes/train.md).

### What makes YOLOv3u more accurate for object detection tasks?

YOLOv3u improves upon YOLOv3 and YOLOv3-Ultralytics by incorporating the anchor-free, objectness-free split head used in YOLOv8 models. This upgrade eliminates the need for pre-defined anchor boxes and objectness scores, enhancing its capability to detect objects of varying sizes and shapes more precisely. This makes YOLOv3u a better choice for complex and diverse object detection tasks. For more information, refer to the [Key Features](#key-features) section.

### How can I use YOLOv3 models for inference?

You can run inference with YOLOv3 models from either Python or the CLI:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLOv3u model
        model = YOLO("yolov3u.pt")

        # Run inference with the YOLOv3u model on the 'bus.jpg' image
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Load a COCO-pretrained YOLOv3u model and run inference on the 'bus.jpg' image
        yolo predict model=yolov3u.pt source=path/to/bus.jpg
        ```

Refer to the [Inference mode documentation](../modes/predict.md) for more details on running YOLO models.

### What tasks are supported by YOLOv3 and its variants?

YOLOv3, YOLOv3-Tiny and YOLOv3-SPP primarily support object detection tasks. These models support the Training, Validation, Inference, and Export modes. For a comprehensive set of tasks supported and more in-depth details, visit our [Object Detection tasks documentation](../tasks/detect.md).

### Where can I find resources to cite YOLOv3 in my research?

If you use YOLOv3 in your research, please cite the original YOLO papers and the Ultralytics YOLOv3 repository. Example BibTeX citation:

!!! quote "Citation"

    === "BibTeX"

        ```bibtex
        @article{redmon2018yolov3,
          title={YOLOv3: An Incremental Improvement},
          author={Redmon, Joseph and Farhadi, Ali},
          journal={arXiv preprint arXiv:1804.02767},
          year={2018}
        }
        ```

For more citation details, refer to the [Citations and Acknowledgments](#citations-and-acknowledgments) section.
