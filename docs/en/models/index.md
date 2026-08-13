---
comments: true
description: Discover a variety of models supported by Ultralytics, including YOLO26 back to YOLOv3, NAS, SAM, and RT-DETR for detection, segmentation, semantic segmentation, depth estimation, and more.
keywords: Ultralytics, supported models, YOLO26, YOLO12, YOLO11, YOLOv10, YOLOv9, YOLOv8, YOLOv7, YOLOv6, YOLOv5, YOLOv4, YOLOv3, SAM3, SAM2, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, YOLO-World, YOLOE, object detection, image segmentation, semantic segmentation, depth estimation, classification, pose estimation, multi-object tracking
---

# Models Supported by Ultralytics

Welcome to Ultralytics' model documentation! We offer support for a wide range of models, each tailored to specific tasks like [object detection](../tasks/detect.md), [instance segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [image classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [multi-object tracking](../modes/track.md). If you're interested in contributing your model architecture to Ultralytics, check out our [Contributing Guide](../help/contributing.md).

![Ultralytics YOLO11 Comparison Plots](https://raw.githubusercontent.com/ultralytics/assets/refs/heads/main/yolo/performance-comparison.png)

## Featured Models

Start with [YOLO26](yolo26.md) for a new project: it is the latest Ultralytics release and the only one covering all seven tasks. [YOLO11](yolo11.md) is the mature alternative, with pretrained checkpoints for each of its five tasks. Pick a specialized family only when you need promptable segmentation ([SAM 3](sam-3.md)), open-vocabulary detection ([YOLOE](yoloe.md), [YOLO-World](yolo-world.md)), or a transformer detector ([RT-DETR](rtdetr.md)).

The table lists every documented model with the tasks it covers, which of the [train, val, predict and export](../modes/index.md) modes Ultralytics supports for it, and when to choose it. [Track](../modes/track.md) is not listed separately: it runs on top of predict for Detect, Segment, Pose and OBB models, while SAM 2 and SAM 3 track through their own video predictors. [Benchmark](../modes/benchmark.md) is not listed either, because it wraps export and val across formats rather than adding support of its own.

| Model                           | Tasks                                                 | Modes                       | Choose it for                                                                                                                                                                                    |
| ------------------------------- | ----------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **[YOLO26](yolo26.md) 🚀 NEW**  | Detect, Segment, Semantic, Depth, Classify, Pose, OBB | Train, Val, Predict, Export | New projects: end-to-end NMS-free inference, edge-optimized deployment, and the widest task coverage                                                                                             |
| **[YOLO12](yolo12.md)**         | Detect, Segment, Classify, Pose, OBB                  | Train, Val, Predict, Export | Benchmarking the attention-centric community release; pretrained weights cover detection only                                                                                                    |
| **[YOLO11](yolo11.md)**         | Detect, Segment, Classify, Pose, OBB                  | Train, Val, Predict, Export | Stable production workloads, with pretrained checkpoints for every task it supports                                                                                                              |
| **[YOLOv10](yolov10.md)**       | Detect                                                | Train, Val, Predict, Export | NMS-free detection research from Tsinghua University                                                                                                                                             |
| **[YOLOv9](yolov9.md)**         | Detect, Segment                                       | Train, Val, Predict, Export | Programmable Gradient Information (PGI), implemented on the Ultralytics [YOLOv5](yolov5.md) codebase                                                                                             |
| **[YOLOv8](yolov8.md)**         | Detect, Segment, Classify, Pose, OBB                  | Train, Val, Predict, Export | Existing YOLOv8 pipelines and established third-party integrations                                                                                                                               |
| **[YOLOv7](yolov7.md)**         | Detect                                                | Predict                     | Running an upstream-trained YOLOv7 export: the package predicts from a compatible ONNX or TensorRT model, never from the native checkpoint                                                       |
| **[YOLOv6](yolov6.md)**         | Detect                                                | Train, Val, Predict, Export | Training [Meituan](https://www.meituan.com/)'s architecture from its YAML; Ultralytics hosts no YOLOv6 `.pt` checkpoints                                                                         |
| **[YOLOv5](yolov5.md)**         | Detect                                                | Train, Val, Predict, Export | Legacy Ultralytics projects; current releases load the updated YOLOv5u checkpoints                                                                                                               |
| **[YOLOv4](yolov4.md)**         | None                                                  | None                        | Architecture reference only: Alexey Bochkovskiy's Darknet-native model is not supported by the package                                                                                           |
| **[YOLOv3](yolov3.md)**         | Detect                                                | Train, Val, Predict, Export | Legacy projects on Joseph Redmon's original architecture; current releases load the updated YOLOv3u checkpoints                                                                                  |
| **[SAM 3](sam-3.md) 🚀 NEW**    | Segment                                               | Predict                     | Meta's promptable concept segmentation in images and video, from text or image exemplars; `sam3.pt` requires access approval on Hugging Face                                                     |
| **[SAM 2](sam-2.md)**           | Segment                                               | Predict                     | Meta's promptable segmentation, tracking objects across video frames                                                                                                                             |
| **[SAM](sam.md)**               | Segment                                               | Predict                     | Meta's original promptable segmentation, including auto-annotation                                                                                                                               |
| **[MobileSAM](mobile-sam.md)**  | Segment                                               | Predict                     | Promptable segmentation on mobile and other resource-constrained devices (Kyung Hee University)                                                                                                  |
| **[FastSAM](fast-sam.md)**      | Segment                                               | Val, Predict, Export        | CNN-based promptable segmentation when SAM latency is the bottleneck (Chinese Academy of Sciences)                                                                                               |
| **[YOLO-NAS](yolo-nas.md)**     | Detect                                                | Val, Predict, Export        | Deci's [NAS](https://www.ultralytics.com/glossary/neural-architecture-search-nas)-optimized detectors, kept for inference and export; Deci no longer maintains them after the NVIDIA acquisition |
| **[RT-DETR](rtdetr.md)**        | Detect                                                | Train, Val, Predict, Export | Baidu's real-time DETR detector: a convolutional backbone with a hybrid [transformer](https://www.ultralytics.com/glossary/transformer) encoder                                                  |
| **[YOLO-World](yolo-world.md)** | Detect                                                | Train, Val, Predict, Export | Open-vocabulary detection from text prompts (Tencent AI Lab); export requires the `-worldv2` checkpoints                                                                                         |
| **[YOLOE](yoloe.md)**           | Detect, Segment                                       | Train, Val, Predict, Export | Open-vocabulary detection and segmentation with text, visual, or prompt-free inference                                                                                                           |

!!! tip "Large Language Models"

    Ultralytics also ships [LLM](llm.md), an OpenAI-compatible interface to large language and vision models for text and image understanding. It has no tasks or modes of its own, so it is not in the table above, but it pairs with any YOLO pipeline and runs against OpenAI and other cloud providers or a fully local on-prem server.

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

Note the below example spotlights YOLO26 [Detect](../tasks/detect.md) models for [object detection](https://www.ultralytics.com/glossary/object-detection). For additional supported tasks see the [Segment](../tasks/segment.md), [Semantic](../tasks/semantic.md), [Depth](../tasks/depth.md), [Classify](../tasks/classify.md), [Pose](../tasks/pose.md) and [OBB](../tasks/obb.md) docs.

!!! example

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) pretrained `*.pt` models as well as configuration `*.yaml` files can be passed to the `YOLO()`, `SAM()`, `NAS()` and `RTDETR()` classes to create a model instance in Python:

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLO26n model
        model = YOLO("yolo26n.pt")

        # Display model information (optional)
        model.info()

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # Run inference with the YOLO26n model on the 'bus.jpg' image
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        CLI commands are available to directly run the models:

        ```bash
        # Load a COCO-pretrained YOLO26n model and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolo26n.pt data=coco8.yaml epochs=100 imgsz=640

        # Load a COCO-pretrained YOLO26n model and run inference on the 'bus.jpg' image
        yolo predict model=yolo26n.pt source=path/to/bus.jpg
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

## FAQ

### What is the latest Ultralytics YOLO model?

The latest Ultralytics YOLO model is [YOLO26](yolo26.md), released in January 2026. YOLO26 features end-to-end NMS-free inference, optimized edge deployment, and supports detection, instance segmentation, [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), classification, pose estimation, and OBB plus open-vocabulary versions. For stable production workloads, both YOLO26 and [YOLO11](yolo11.md) are recommended choices.

### How can I train a YOLO model on custom data?

Training a YOLO model on custom data can be easily accomplished using Ultralytics' libraries. Here's a quick example:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO model
        model = YOLO("yolo26n.pt")  # or any other YOLO model

        # Train the model on custom dataset
        results = model.train(data="custom_data.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo train model=yolo26n.pt data='custom_data.yaml' epochs=100 imgsz=640
        ```

For more detailed instructions, visit the [Train](../modes/train.md) documentation page.

### Which YOLO versions are supported by Ultralytics?

Ultralytics natively supports YOLOv3, YOLOv5, YOLOv6, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO12, and YOLO26, along with the SAM family (SAM 3, SAM 2, SAM, MobileSAM and FastSAM), YOLO-NAS, RT-DETR, YOLO-World, and YOLOE. The package publishes no weights or YAMLs for YOLOv4 or YOLOv7: YOLOv4 is documented as an architecture reference only, while YOLOv7 runs as an exported ONNX or TensorRT model. See [Featured Models](#featured-models) for the tasks and modes available for each.

### Why should I use Ultralytics Platform for [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) projects?

[Ultralytics Platform](../platform/index.md) provides a no-code, end-to-end platform for training, deploying, and managing YOLO models. It simplifies complex workflows, enabling users to focus on model performance and application. It also offers [cloud training capabilities](../platform/train/cloud-training.md), comprehensive dataset management, and user-friendly interfaces for both beginners and experienced developers.

### What types of tasks can Ultralytics YOLO models perform?

Ultralytics YOLO models are versatile and can perform tasks including object detection, instance segmentation, [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), classification, pose estimation, and oriented object detection (OBB). The latest model, [YOLO26](yolo26.md), supports all seven tasks plus open-vocabulary detection. For details on specific tasks, refer to the [Task pages](../tasks/index.md).
