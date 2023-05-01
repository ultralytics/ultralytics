---
comments: true
---

# Models

Ultralytics supports many models and architectures with more to come in the future. Want to add your model architecture? [Here's](../help/contributing.md) how you can contribute.

In this documentation, we provide information on four major models:

1. [YOLOv3](./yolov3.md): The third iteration of the YOLO model family, known for its efficient real-time object detection capabilities.
2. [YOLOv5](./yolov5.md): An improved version of the YOLO architecture, offering better performance and speed tradeoffs compared to previous versions.
3. [YOLOv8](./yolov8.md): The latest version of the YOLO family, featuring enhanced capabilities such as instance segmentation, pose/keypoints estimation, and classification.
4. [Segment Anything Model (SAM)](./sam.md): Meta's Segment Anything Model (SAM).

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