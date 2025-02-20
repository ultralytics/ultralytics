---
comments: true
description: Discover YOLO12, featuring groundbreaking attention-centric architecture for state-of-the-art object detection with unmatched accuracy and efficiency.
keywords: YOLO12, attention-centric object detection, YOLO series, Ultralytics, computer vision, AI, machine learning, deep learning
---

# YOLO12: Attention-Centric Real-Time Object Detector

## Overview

YOLO12 introduces an attention-centric architecture that challenges the traditional CNN-based approach, while maintaining the real-time inference capabilities for which YOLO is known. This model achieves state-of-the-art accuracy while maintaining real-time performance through methodological innovations in both attention mechanisms and overall architectural design.

## Key Features

- **Area Attention Mechanism**: A novel approach that maintains large receptive fields while reducing computational complexity. This is achieved through an efficient partitioning of the feature maps.
- **Residual Efficient Layer Aggregation Networks (R-ELAN)**: Improved feature aggregation that enhances optimization, especially for large-scale models. R-ELAN incorporates residual connections and a refined feature integration approach.
- **Optimized Attention Architecture**: A streamlined implementation of attention, including adjusted MLP ratios and the integration of FlashAttention for efficient memory access.
- **Comprehensive Task Support**: YOLO12 supports a wide range of computer vision tasks, including object detection, instance segmentation, classification, pose estimation, and oriented object detection (OBB).
- **Enhanced Efficiency**: Achieves higher accuracy with a reduced number of parameters compared to many previous models. YOLO12 maintains a balance between speed and accuracy.
- **Flexible Deployment**: Designed for seamless deployment across various platforms, ranging from edge devices to cloud infrastructure.

## Supported Tasks and Modes

YOLO12 provides comprehensive support for various computer vision tasks. The table below indicates which tasks are supported, along with the operational modes (Inference, Validation, Training, and Export) supported for each task:

| Model Type  | Task           | Inference | Validation | Training | Export |
| ----------- | -------------- | --------- | ---------- | -------- | ------ |
| YOLO12      | Detection      | ✅        | ✅         | ✅       | ✅     |
| YOLO12-seg  | Segmentation   | ✅        | ✅         | ✅       | ✅     |
| YOLO12-pose | Pose           | ✅        | ✅         | ✅       | ✅     |
| YOLO12-cls  | Classification | ✅        | ✅         | ✅       | ✅     |
| YOLO12-obb  | OBB            | ✅        | ✅         | ✅       | ✅     |

## Performance Metrics

YOLO12 demonstrates accuracy improvements across all model scales, though at the cost of some speed relative to other YOLO models. Below are quantitative results for object detection on the COCO dataset:

### Detection Performance (COCO)

| Model   | Size | mAP@50-95 | Speed (ms)<sup>1</sup> | Parameters (M) | FLOPs (G) | Comparison (mAP / Speed)<sup>2</sup> |
| ------- | ---- | --------- | ---------------------- | -------------- | --------- | ------------------------------------ |
| YOLO12n | 640  | 40.6      | 1.64                   | 2.6            | 6.5       | +2.1% / -9% (vs. YOLOv10-n)          |
|         |      |           |                        |                |           | +1.2% / +4% (vs. YOLOv11-n)          |
| YOLO12s | 640  | 48.0      | 2.61                   | 9.3            | 21.4      | +1.7% / -5% (vs. YOLOv10-s)          |
|         |      |           |                        |                |           | +1.1% / -4% (vs. YOLOv11-s)          |
|         |      |           |                        |                |           | +1.5% / +42% (vs. RT-DETR-R18)       |
| YOLO12m | 640  | 52.5      | 4.86                   | 20.2           | 67.5      | +1.4% / +2% (vs. YOLOv10-m)          |
|         |      |           |                        |                |           | +1.0% / +3% (vs. YOLOv11-m)          |
| YOLO12l | 640  | 53.7      | 6.77                   | 26.4           | 88.9      | +0.5% / +7% (vs. YOLOv10-l)          |
|         |      |           |                        |                |           | +0.4% / -8% (vs. YOLOv11-l)          |
| YOLO12x | 640  | 55.2      | 11.79                  | 59.1           | 199.0     | +0.8% / -9% (vs. YOLOv10-x)          |
|         |      |           |                        |                |           | +0.6% / -4% (vs. YOLOv11-x)          |

<sup>1</sup> Inference speed measured on a T4 GPU with TensorRT FP16.
<sup>2</sup> Comparisons show relative improvement in mAP and percentage change in speed (positive indicates faster, negative indicates slower). Comparisons are provided against published results for YOLOv10, YOLOv11, and RT-DETR-R18 where available.

## Usage Examples

This section provides basic examples for training and inference with YOLO12. For comprehensive documentation on these and other modes (including [Validation](../modes/val.md) and [Export](../modes/export.md)), please refer to the dedicated [Predict](../modes/predict.md), [Train](../modes/train.md) documentation pages.

The examples below focus on YOLO12 [Detect](../tasks/detect.md) models, which are used for [object detection](https://www.ultralytics.com/glossary/object-detection). For other supported tasks (segmentation, classification, oriented object detection, and pose estimation), consult the corresponding task-specific documentation: [Segment](../tasks/segment.md), [Classify](../tasks/classify.md), [OBB](../tasks/obb.md), and [Pose](../tasks/pose.md).

!!! example

    === "Python"

        Pretrained `*.pt` models (using [PyTorch](https://www.ultralytics.com/glossary/pytorch)) and configuration `*.yaml` files can be passed to the `YOLO()` class to instantiate a model in Python:

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

        Command Line Interface (CLI) commands are also available for direct model execution:

        ```bash
        # Load a COCO-pretrained YOLO12n model and train on the COCO8 example dataset for 100 epochs
        yolo train model=yolo12n.pt data=coco8.yaml epochs=100 imgsz=640

        # Load a COCO-pretrained YOLO12n model and run inference on the 'bus.jpg' image
        yolo predict model=yolo12n.pt source=path/to/bus.jpg
        ```

## Key Improvements

1.  **Enhanced Feature Extraction**:

    - **Area Attention**: Efficiently processes large receptive fields.
    - **Optimized Balance**: Improved balance between attention and feed-forward networks.
    - **R-ELAN**: Enhances feature aggregation through the R-ELAN architecture.

2.  **Optimization Innovations**:

    - **Residual Connections**: Introduces residual connections with scaling techniques to ensure stable training, particularly for larger models.
    - **Refined Feature Integration**: Implements improved methods for feature integration.
    - **FlashAttention**: Integrates FlashAttention for efficient memory access.

3.  **Architectural Efficiency**:

    - **Reduced Parameters**: Achieves a lower parameter count while maintaining or improving accuracy.
    - **Streamlined Attention**: Uses a simplified attention implementation that avoids positional encoding.
    - **Optimized MLP Ratios**: Adjusts MLP ratios to allocate computational resources more effectively.

## Requirements

The Ultralytics YOLO12 implementation _does not require_ FlashAttention. However, to optionally compile and use FlashAttention with YOLO12, one of the following NVIDIA GPUs is needed:

- Turing GPUs (e.g., T4, Quadro RTX series)
- Ampere GPUs (e.g., RTX30 series, A30/40/100)
- Ada Lovelace GPUs (e.g., RTX40 series)
- Hopper GPUs (e.g., H100/H200)

## Citations and Acknowledgements

If you utilize YOLO12 in your research, please cite the following:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @software{yolo12,
          author = {Tian, Yunjie and Ye, Qixiang and Doermann, David},
          title = {YOLOv12: Attention-Centric Real-Time Object Detectors},
          year = {2025},
          url = {https://github.com/sunsmarterjie/yolov12},
          license = {AGPL-3.0}
        }
        ```
