---
comments: true
description: Learn about the supported models and architectures, such as YOLOv3, YOLOv5, and YOLOv8, and how to contribute your own model to Ultralytics.
keywords: Ultralytics YOLO, YOLOv3, YOLOv4, YOLOv5, YOLOv6, YOLOv7, YOLOv8, SAM, YOLO-NAS, RT-DETR, object detection, instance segmentation, detection transformers, real-time detection, computer vision, CLI, Python
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
8. [YOLO-NAS](./yolo-nas.md): YOLO Neural Architecture Search (NAS) Models.
9. [Realtime Detection Transformers (RT-DETR)](./rtdetr.md): Baidu's PaddlePaddle Realtime Detection Transformer (RT-DETR) models.

You can use these models directly in the Command Line Interface (CLI) or in a Python environment. Below are examples of how to use the models with CLI and Python:

## CLI Example

```bash
yolo task=detect mode=train model=yolov8n.yaml data=coco128.yaml epochs=100
```

## Python Example

```python
from ultralytics import YOLO

model = YOLO("model.yaml")  # build a YOLOv8n model from scratch
# YOLO("model.pt")  use pre-trained model if available
model.info()  # display model information
model.train(data="coco128.yaml", epochs=100)  # train the model
```

For more details on each model, their supported tasks, modes, and performance, please visit their respective documentation pages linked above.