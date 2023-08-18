---
comments: true
description: Learn about the YOLO family, SAM, MobileSAM, FastSAM, YOLO-NAS, and RT-DETR models supported by Ultralytics, with examples on how to use them via CLI and Python.
keywords: Ultralytics, documentation, YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, models, architectures, Python, CLI
---

# Models

Ultralytics supports many models and architectures with more to come in the future. Want to add your model architecture? [Here's](../help/contributing.md) how you can contribute.

In this documentation, we provide information on four major models:

1. [YOLOv3](./yolov3.md): The third iteration of the YOLO model family originally by Joseph Redmon, known for its efficient real-time object detection capabilities.
2. [YOLOv4](./yolov3.md): A darknet-native update to YOLOv3 released by Alexey Bochkovskiy in 2020.
3. [YOLOv5](./yolov5.md): An improved version of the YOLO architecture by Ultralytics, offering better performance and speed tradeoffs compared to previous versions.
4. [YOLOv6](./yolov6.md): Released by [Meituan](https://about.meituan.com/) in 2022 and is in use in many of the company's autonomous delivery robots.
5. [YOLOv7](./yolov7.md): Updated YOLO models released in 2022 by the authors of YOLOv4.
6. [YOLOv8](./yolov8.md): The latest version of the YOLO family, featuring enhanced capabilities such as instance segmentation, pose/keypoints estimation, and classification.
7. [Segment Anything Model (SAM)](./sam.md): Meta's Segment Anything Model (SAM).
8. [Mobile Segment Anything Model (MobileSAM)](./mobile-sam.md): MobileSAM for mobile applications by Kyung Hee University.
9. [Fast Segment Anything Model (FastSAM)](./fast-sam.md): FastSAM by Image & Video Analysis Group, Institute of Automation, Chinese Academy of Sciences.
10. [YOLO-NAS](./yolo-nas.md): YOLO Neural Architecture Search (NAS) Models.
11. [Realtime Detection Transformers (RT-DETR)](./rtdetr.md): Baidu's PaddlePaddle Realtime Detection Transformer (RT-DETR) models.

You can use many of these models directly in the Command Line Interface (CLI) or in a Python environment. Below are examples of how to use the models with CLI and Python:

## Usage

This example provides simple inference code for YOLO, SAM and RTDETR models. For more options including handling inference results see [Predict](../modes/predict.md) mode. For using models with additional modes see [Train](../modes/train.md), [Val](../modes/val.md) and [Export](../modes/export.md).

!!! example ""

    === "Python"

        PyTorch pretrained `*.pt` models as well as configuration `*.yaml` files can be passed to the `YOLO()`, `SAM()`, `NAS()` and `RTDETR()` classes to create a model instance in python:

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLOv8n model
        model = YOLO('yolov8n.pt')

        # Display model information (optional)
        model.info()

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Run inference with the YOLOv8n model on the 'bus.jpg' image
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLI commands are available to directly run the models:

        ```bash
        # Load a COCO-pretrained YOLOv8n model and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # Load a COCO-pretrained YOLOv8n model and run inference on the 'bus.jpg' image
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

For more details on each model, their supported tasks, modes, and performance, please visit their respective documentation pages linked above.
