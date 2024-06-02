---
comments: true
description: Discover a variety of models supported by Ultralytics, including YOLOv3 to YOLOv10, NAS, SAM, and RT-DETR for detection, segmentation, and more.
keywords: Ultralytics, supported models, YOLOv3, YOLOv4, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, SAM, NAS, RT-DETR, object detection, image segmentation, classification, pose estimation, multi-object tracking
---

# Models Supported by Ultralytics

Welcome to Ultralytics' model documentation! We offer support for a wide range of models, each tailored to specific tasks like [object detection](../tasks/detect.md), [instance segmentation](../tasks/segment.md), [image classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [multi-object tracking](../modes/track.md). If you're interested in contributing your model architecture to Ultralytics, check out our [Contributing Guide](../help/contributing.md).

## Featured Models

Here are some of the key models supported:

1. **[YOLOv3](yolov3.md)**: The third iteration of the YOLO model family, originally by Joseph Redmon, known for its efficient real-time object detection capabilities.
2. **[YOLOv4](yolov4.md)**: A darknet-native update to YOLOv3, released by Alexey Bochkovskiy in 2020.
3. **[YOLOv5](yolov5.md)**: An improved version of the YOLO architecture by Ultralytics, offering better performance and speed trade-offs compared to previous versions.
4. **[YOLOv6](yolov6.md)**: Released by [Meituan](https://about.meituan.com/) in 2022, and in use in many of the company's autonomous delivery robots.
5. **[YOLOv7](yolov7.md)**: Updated YOLO models released in 2022 by the authors of YOLOv4.
6. **[YOLOv8](yolov8.md) NEW 🚀**: The latest version of the YOLO family, featuring enhanced capabilities such as instance segmentation, pose/keypoints estimation, and classification.
7. **[YOLOv9](yolov9.md)**: An experimental model trained on the Ultralytics [YOLOv5](yolov5.md) codebase implementing Programmable Gradient Information (PGI).
8. **[YOLOv10](yolov10.md)**: By Tsinghua University, featuring NMS-free training and efficiency-accuracy driven architecture, delivering state-of-the-art performance and latency. 
9. **[Segment Anything Model (SAM)](sam.md)**: Meta's Segment Anything Model (SAM).
10. **[Mobile Segment Anything Model (MobileSAM)](mobile-sam.md)**: MobileSAM for mobile applications, by Kyung Hee University.
11. **[Fast Segment Anything Model (FastSAM)](fast-sam.md)**: FastSAM by Image & Video Analysis Group, Institute of Automation, Chinese Academy of Sciences.
12. **[YOLO-NAS](yolo-nas.md)**: YOLO Neural Architecture Search (NAS) Models.
13. **[Realtime Detection Transformers (RT-DETR)](rtdetr.md)**: Baidu's PaddlePaddle Realtime Detection Transformer (RT-DETR) models.
14. **[YOLO-World](yolo-world.md)**: Real-time Open Vocabulary Object Detection models from Tencent AI Lab.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Run Ultralytics YOLO models in just a few lines of code.
</p>

## Getting Started: Usage Examples

This example provides simple YOLO training and inference examples. For full documentation on these and other [modes](../modes/index.md) see the [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) and [Export](../modes/export.md) docs pages.

Note the below example is for YOLOv8 [Detect](../tasks/detect.md) models for object detection. For additional supported tasks see the [Segment](../tasks/segment.md), [Classify](../tasks/classify.md) and [Pose](../tasks/pose.md) docs.

!!! Example

    === "Python"

        PyTorch pretrained `*.pt` models as well as configuration `*.yaml` files can be passed to the `YOLO()`, `SAM()`, `NAS()` and `RTDETR()` classes to create a model instance in Python:

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLOv8n model
        model = YOLO("yolov8n.pt")

        # Display model information (optional)
        model.info()

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # Run inference with the YOLOv8n model on the 'bus.jpg' image
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        CLI commands are available to directly run the models:

        ```bash
        # Load a COCO-pretrained YOLOv8n model and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # Load a COCO-pretrained YOLOv8n model and run inference on the 'bus.jpg' image
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## Contributing New Models

Interested in contributing your model to Ultralytics? Great! We're always open to expanding our model portfolio.

1. **Fork the Repository**: Start by forking the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics).

2. **Clone Your Fork**: Clone your fork to your local machine and create a new branch to work on.

3. **Implement Your Model**: Add your model following the coding standards and guidelines provided in our [Contributing Guide](../help/contributing.md).

4. **Test Thoroughly**: Make sure to test your model rigorously, both in isolation and as part of the pipeline.

5. **Create a Pull Request**: Once you're satisfied with your model, create a pull request to the main repository for review.

6. **Code Review & Merging**: After review, if your model meets our criteria, it will be merged into the main repository.

For detailed steps, consult our [Contributing Guide](../help/contributing.md).
