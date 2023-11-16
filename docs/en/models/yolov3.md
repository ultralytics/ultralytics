---
comments: true
description: Get an overview of YOLOv3, YOLOv3-Ultralytics and YOLOv3u. Learn about their key features, usage, and supported tasks for object detection.
keywords: YOLOv3, YOLOv3-Ultralytics, YOLOv3u, Object Detection, Inference, Training, Ultralytics
---

# YOLOv3, YOLOv3-Ultralytics, and YOLOv3u

## Overview

This document presents an overview of three closely related object detection models, namely [YOLOv3](https://pjreddie.com/darknet/yolo/), [YOLOv3-Ultralytics](https://github.com/ultralytics/yolov3), and [YOLOv3u](https://github.com/ultralytics/ultralytics).

1. **YOLOv3:** This is the third version of the You Only Look Once (YOLO) object detection algorithm. Originally developed by Joseph Redmon, YOLOv3 improved on its predecessors by introducing features such as multiscale predictions and three different sizes of detection kernels.

2. **YOLOv3-Ultralytics:** This is Ultralytics' implementation of the YOLOv3 model. It reproduces the original YOLOv3 architecture and offers additional functionalities, such as support for more pre-trained models and easier customization options.

3. **YOLOv3u:** This is an updated version of YOLOv3-Ultralytics that incorporates the anchor-free, objectness-free split head used in YOLOv8 models. YOLOv3u maintains the same backbone and neck architecture as YOLOv3 but with the updated detection head from YOLOv8.

![Ultralytics YOLOv3](https://raw.githubusercontent.com/ultralytics/assets/main/yolov3/banner-yolov3.png)

## Key Features

- **YOLOv3:** Introduced the use of three different scales for detection, leveraging three different sizes of detection kernels: 13x13, 26x26, and 52x52. This significantly improved detection accuracy for objects of different sizes. Additionally, YOLOv3 added features such as multi-label predictions for each bounding box and a better feature extractor network.

- **YOLOv3-Ultralytics:** Ultralytics' implementation of YOLOv3 provides the same performance as the original model but comes with added support for more pre-trained models, additional training methods, and easier customization options. This makes it more versatile and user-friendly for practical applications.

- **YOLOv3u:** This updated model incorporates the anchor-free, objectness-free split head from YOLOv8. By eliminating the need for pre-defined anchor boxes and objectness scores, this detection head design can improve the model's ability to detect objects of varying sizes and shapes. This makes YOLOv3u more robust and accurate for object detection tasks.

## Supported Tasks

YOLOv3, YOLOv3-Ultralytics, and YOLOv3u all support the following tasks:

- Object Detection

## Supported Modes

All three models support the following modes:

- Inference
- Validation
- Training
- Export

## Performance

Below is a comparison of the performance of the three models. The performance is measured in terms of the Mean Average Precision (mAP) on the COCO dataset:

TODO

## Usage

You can use YOLOv3 for object detection tasks using the Ultralytics repository. The following is a sample code snippet showing how to use YOLOv3 model for inference:

!!! example ""

    This example provides simple inference code for YOLOv3. For more options including handling inference results see [Predict](../modes/predict.md) mode. For using YOLOv3 with additional modes see [Train](../modes/train.md), [Val](../modes/val.md) and [Export](../modes/export.md).

    === "Python"

        PyTorch pretrained `*.pt` models as well as configuration `*.yaml` files can be passed to the `YOLO()` class to create a model instance in python:

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLOv3n model
        model = YOLO('yolov3n.pt')

        # Display model information (optional)
        model.info()

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Run inference with the YOLOv3n model on the 'bus.jpg' image
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLI commands are available to directly run the models:

        ```bash
        # Load a COCO-pretrained YOLOv3n model and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolov3n.pt data=coco8.yaml epochs=100 imgsz=640

        # Load a COCO-pretrained YOLOv3n model and run inference on the 'bus.jpg' image
        yolo predict model=yolov3n.pt source=path/to/bus.jpg
        ```

## Citations and Acknowledgements

If you use YOLOv3 in your research, please cite the original YOLO papers and the Ultralytics YOLOv3 repository:

!!! note ""

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
