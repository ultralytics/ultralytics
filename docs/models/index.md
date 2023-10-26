---
comments: true
description: Explore the diverse range of YOLO family, SAM, MobileSAM, FastSAM, YOLO-NAS, and RT-DETR models supported by Ultralytics. Get started with examples for both CLI and Python usage.
keywords: Ultralytics, documentation, YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, models, architectures, Python, CLI
---

# Models Supported by Ultralytics

Welcome to Ultralytics' model documentation! We offer support for a wide range of models, each tailored to specific tasks like [object detection](../tasks/detect.md), [instance segmentation](../tasks/segment.md), [image classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [multi-object tracking](../modes/track.md). If you're interested in contributing your model architecture to Ultralytics, check out our [Contributing Guide](../help/contributing.md).

## Featured Models

Here are some of the key models supported:

1. **[YOLOv3](./yolov3.md)**: YOLO's third iteration, renowned for real-time object detection. Created by Joseph Redmon.
2. **[YOLOv4](./yolov4.md)**: An advancement over YOLOv3, released by Alexey Bochkovskiy.
3. **[YOLOv5](./yolov5.md)**: Ultralytics' enhancement of YOLO, balancing performance and speed.
4. **[YOLOv6](./yolov6.md)**: Developed by [Meituan](https://about.meituan.com/), deployed in their autonomous delivery robots.
5. **[YOLOv7](./yolov7.md)**: Latest from YOLOv4 authors, released in 2022.
6. **[YOLOv8](./yolov8.md)**: Newest in the YOLO family with added capabilities like instance segmentation and pose estimation.
7. **[SAM (Segment Anything Model)](./sam.md)**: Developed by Meta, tailored for segmentation tasks.
8. **[MobileSAM](./mobile-sam.md)**: Optimized for mobile applications, by Kyung Hee University.
9. **[FastSAM](./fast-sam.md)**: Developed by the Chinese Academy of Sciences for rapid segmentation.
10. **[YOLO-NAS](./yolo-nas.md)**: Neural Architecture Search (NAS) models within the YOLO framework.
11. **[RT-DETR (Realtime Detection Transformers)](./rtdetr.md)**: By Baidu's PaddlePaddle, offering real-time detection.

<p align="center">
  <iframe width="720" height="405" src="https://www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
  <strong>Watch:</strong> Run Ultralytics YOLO models in just a few lines of code.
</p>

## Getting Started: Usage Examples

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

## Contributing New Models

Interested in contributing your model to Ultralytics? Great! We're always open to expanding our model portfolio.

1. **Fork the Repository**: Start by forking the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics).

2. **Clone Your Fork**: Clone your fork to your local machine and create a new branch to work on.

3. **Implement Your Model**: Add your model following the coding standards and guidelines provided in our [Contributing Guide](../help/contributing.md).

4. **Test Thoroughly**: Make sure to test your model rigorously, both in isolation and as part of the pipeline.

5. **Create a Pull Request**: Once you're satisfied with your model, create a pull request to the main repository for review.

6. **Code Review & Merging**: After review, if your model meets our criteria, it will be merged into the main repository.

For detailed steps, consult our [Contributing Guide](../help/contributing.md).
