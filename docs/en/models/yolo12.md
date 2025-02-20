---
comments: true
description: Discover YOLO12, featuring groundbreaking attention-centric architecture for state-of-the-art object detection with unmatched accuracy and efficiency.
keywords: YOLO12, attention-centric object detection, YOLO series, Ultralytics, computer vision, AI, machine learning, deep learning
---

# YOLO12: Attention-Centric Real-Time Object Detector

## Overview

YOLO12 introduces an attention-centric architecture that departs from the traditional CNN-based approaches used in previous YOLO models, yet retains the real-time inference speed essential for many applications. This model achieves state-of-the-art object detection accuracy through novel methodological innovations in attention mechanisms and overall network architecture, while maintaining real-time performance.

## Key Features

- **Area Attention Mechanism**: A new self-attention approach that processes large receptive fields efficiently. It divides feature maps into *l* equal-sized regions (defaulting to 4), either horizontally or vertically, avoiding complex operations and maintaining a large effective receptive field. This significantly reduces computational cost compared to standard self-attention.
- **Residual Efficient Layer Aggregation Networks (R-ELAN)**: An improved feature aggregation module based on ELAN [57], designed to address optimization challenges, especially in larger-scale attention-centric models. R-ELAN introduces:
    - Block-level residual connections with scaling (similar to layer scaling [52]).
    - A redesigned feature aggregation method creating a bottleneck-like structure.
- **Optimized Attention Architecture**: YOLO12 streamlines the standard attention mechanism for greater efficiency and compatibility with the YOLO framework. This includes:
    - Using FlashAttention [13, 14] to minimize memory access overhead.
    - Removing positional encoding for a cleaner and faster model.
    - Adjusting the MLP ratio (from the typical 4 to 1.2 or 2) to better balance computation between attention and feed-forward layers.
    - Reducing the depth of stacked blocks for improved optimization.
    - Leveraging convolution operations (where appropriate) for their computational efficiency.
    - Adding a 7x7 separable convolution (the "position perceiver") to the attention mechanism to implicitly encode positional information.
- **Comprehensive Task Support**: YOLO12 supports a range of core computer vision tasks: object detection, instance segmentation, image classification, pose estimation, and oriented object detection (OBB).
- **Enhanced Efficiency**: Achieves higher accuracy with fewer parameters compared to many prior models, demonstrating an improved balance between speed and accuracy.
- **Flexible Deployment**: Designed for deployment across diverse platforms, from edge devices to cloud infrastructure.

## Supported Tasks and Modes

YOLO12 supports a variety of computer vision tasks. The table below shows task support and the operational modes (Inference, Validation, Training, and Export) enabled for each:

| Model Type  | Task           | Inference | Validation | Training | Export |
|-------------|----------------|-----------|------------|----------|--------|
| YOLO12      | Detection      | ✅         | ✅          | ✅        | ✅      |
| YOLO12-seg  | Segmentation   | ✅         | ✅          | ✅        | ✅      |
| YOLO12-pose | Pose           | ✅         | ✅          | ✅        | ✅      |
| YOLO12-cls  | Classification | ✅         | ✅          | ✅        | ✅      |
| YOLO12-obb  | OBB            | ✅         | ✅          | ✅        | ✅      |

## Performance Metrics

YOLO12 demonstrates significant accuracy improvements across all model scales, with some trade-offs in speed compared to the *fastest* prior YOLO models. Below are quantitative results for object detection on the COCO validation dataset:

### Detection Performance (COCO val2017)

| Model  | size<br><sup>(pixels) | mAP<sup>val</sup><br>50-95 | Speed (ms)<sup>1</sup> | params<br><sup>(M) | FLOPs<br><sup>(B) | Comparison (mAP / Speed)<sup>2</sup> |
| ------ | ----------------------- | ---------------------------- | ----------------------- | ----------------- | ---------------- |-------------------------------------|
| YOLO12n | 640                     | 40.6                         | 1.64                    | 2.6               | 6.5              | +2.1% / -9%  (vs. YOLOv10n)         |
|        |                         |                              |                         |                   |                  | +1.2% / +4%   (vs. YOLOv11n)        |
| YOLO12s | 640                     | 48.0                         | 2.61                    | 9.3               | 21.4             | +1.7% / -5%   (vs. YOLOv10s)        |
|        |                         |                              |                         |                   |                  | +1.1% / -4%   (vs. YOLOv11s)        |
|        |                         |                              |                         |      9.3             |     21.4             | +1.5% / +42%  (vs. RT-DETR-R18)     |
|      |        |           |                       |      9.3   | 21.4 | +0.1% / +42% (vs. RT-DETRv2-R18)    |
| YOLO12m | 640                     | 52.5                         | 4.86                    | 20.2              | 67.5             | +1.4% / +2%   (vs. YOLOv10m)        |
|        |                         |                              |                         |                   |                  | +1.0% / +3%   (vs. YOLOv11m)        |
| YOLO12l | 640                     | 53.7                         | 6.77                    | 26.4              | 88.9             | +0.5% / +7%  (vs. YOLOv10l)         |
|        |                         |                              |                         |                   |                  | +0.4% / -8%  (vs. YOLOv11l)         |
| YOLO12x | 640                     | 55.2                         | 11.79                   | 59.1              | 199.0            | +0.8% / -9% (vs. YOLOv10x)          |
|        |                         |                              |                         |                   |                  | +0.6% / -4% (vs. YOLOv11x)          |


<sup>1</sup> Inference speed measured on an NVIDIA T4 GPU with TensorRT FP16 precision.
<sup>2</sup> Comparisons show the relative improvement in mAP and the percentage change in speed (positive indicates faster; negative indicates slower). Comparisons are made against published results for YOLOv10, YOLOv11, and RT-DETR where available.

## Usage Examples

This section provides examples for training and inference with YOLO12. For more comprehensive documentation on these and other modes (including [Validation](../modes/val.md) and [Export](../modes/export.md)), consult the dedicated [Predict](../modes/predict.md) and [Train](../modes/train.md) pages.

The examples below focus on YOLO12 [Detect](../tasks/detect.md) models (for object detection). For other supported tasks (segmentation, classification, oriented object detection, and pose estimation), refer to the respective task-specific documentation: [Segment](../tasks/segment.md), [Classify](../tasks/classify.md), [OBB](../tasks/obb.md), and [Pose](../tasks/pose.md).

!!! example

    === "Python"

        Pretrained `*.pt` models (using [PyTorch](https://pytorch.org/)) and configuration `*.yaml` files can be passed to the `YOLO()` class to create a model instance in Python:

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLO12n model
        model = YOLO("yolo12n.pt")

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # Run inference with the YOLO12n model on the 'bus.jpg' image
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        Command Line Interface (CLI) commands are also available:

        ```bash
        # Load a COCO-pretrained YOLO12n model and train on the COCO8 example dataset for 100 epochs
        yolo train model=yolo12n.pt data=coco8.yaml epochs=100 imgsz=640

        # Load a COCO-pretrained YOLO12n model and run inference on the 'bus.jpg' image
        yolo predict model=yolo12n.pt source=path/to/bus.jpg
        ```

## Key Improvements

1. **Enhanced Feature Extraction**:

    - **Area Attention**: Efficiently handles large receptive fields, reducing computational cost.
    - **Optimized Balance**: Improved balance between attention and feed-forward network computations.
    - **R-ELAN**: Enhances feature aggregation using the R-ELAN architecture.

2. **Optimization Innovations**:

    - **Residual Connections**: Introduces residual connections with scaling to stabilize training, especially in larger models.
    - **Refined Feature Integration**: Implements an improved method for feature integration within R-ELAN.
    - **FlashAttention**: Incorporates FlashAttention to reduce memory access overhead.

3. **Architectural Efficiency**:

    - **Reduced Parameters**: Achieves a lower parameter count while maintaining or improving accuracy compared to many previous models.
    - **Streamlined Attention**: Uses a simplified attention implementation, avoiding positional encoding.
    - **Optimized MLP Ratios**: Adjusts MLP ratios to more effectively allocate computational resources.

## Requirements

The Ultralytics YOLO12 implementation, by default, *does not require* FlashAttention. However, FlashAttention can be optionally compiled and used with YOLO12. To compile FlashAttention, one of the following NVIDIA GPUs is needed:

- Turing GPUs (e.g., T4, Quadro RTX series)
- Ampere GPUs (e.g., RTX30 series, A30/40/100)
- Ada Lovelace GPUs (e.g., RTX40 series)
- Hopper GPUs (e.g., H100/H200)

## Citations and Acknowledgements

If you use YOLO12 in your research, please cite the original paper and software citations included as reference implementations.

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{tian2025yolov12,
          title={YOLOv12: Attention-Centric Real-Time Object Detectors},
          author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
          journal={arXiv preprint arXiv:2502.12524},
          year={2025}
        }

        @software{yolo12,
          author = {Tian, Yunjie and Ye, Qixiang and Doermann, David},
          title = {YOLOv12: Attention-Centric Real-Time Object Detectors},
          year = {2025},
          url = {https://github.com/sunsmarterjie/yolov12},
          license = {AGPL-3.0}
        }
        ```
