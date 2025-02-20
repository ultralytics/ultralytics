---
comments: true
description: Discover YOLO12, featuring groundbreaking attention-centric architecture for state-of-the-art object detection with unmatched accuracy and efficiency.
keywords: YOLO12, attention-centric object detection, YOLO series, Ultralytics, computer vision, AI, machine learning, deep learning
---

# YOLO12 Attention-Centric Real-Time Object Detector

## Overview

YOLO12 introduces an attention-centric architecture that challenges the traditional CNN-based approach while maintaining the high-speed performance that YOLO is known for. This innovative model achieves state-of-the-art accuracy while preserving real-time inference capabilities through methodological innovations in attention mechanisms and architectural design.

## Key Features

- **Area Attention Mechanism**: A novel approach that maintains large receptive fields while reducing computational complexity through efficient partitioning of feature maps
- **Residual Efficient Layer Aggregation Networks (R-ELAN)**: Enhanced feature aggregation with improved optimization for large-scale models through residual connections and refined feature integration
- **Optimized Attention Architecture**: Refined attention implementation with adjusted MLP ratios and efficient memory access through FlashAttention integration
- **Comprehensive Task Support**: Robust performance across object detection, instance segmentation, classification, pose estimation, and oriented object detection
- **Enhanced Efficiency**: Achieves higher accuracy with fewer parameters compared to previous models
- **Flexible Deployment**: Seamless deployment across various platforms, from edge devices to cloud infrastructure

## Supported Tasks and Modes

YOLO12 provides comprehensive support for various computer vision tasks:

| Model Type  | Task           | Inference | Validation | Training | Export |
| ----------- | -------------- | --------- | ---------- | -------- | ------ |
| YOLO12      | Detection      | ✅        | ✅         | ✅       | ✅     |
| YOLO12-seg  | Segmentation   | ✅        | ✅         | ✅       | ✅     |
| YOLO12-pose | Pose           | ✅        | ✅         | ✅       | ✅     |
| YOLO12-cls  | Classification | ✅        | ✅         | ✅       | ✅     |
| YOLO12-obb  | OBB            | ✅        | ✅         | ✅       | ✅     |

## Performance Metrics

YOLO12 demonstrates significant accuracy improvements across all model scales at the expense of slower speeds vs. other YOLO models:

### Detection Performance (COCO)

| Model    | Size | mAP@50-95 | Speed (ms) | Parameters (M) | FLOPs (G) |
| -------- | ---- | --------- | ---------- | -------------- | --------- |
| YOLO12-n | 640  | 40.6      | 1.64       | 2.6            | 6.5       |
| YOLO12-s | 640  | 48.0      | 2.61       | 9.3            | 21.4      |
| YOLO12-m | 640  | 52.5      | 4.86       | 20.2           | 67.5      |
| YOLO12-l | 640  | 53.7      | 6.77       | 26.4           | 88.9      |
| YOLO12-x | 640  | 55.2      | 11.79      | 59.1           | 199.0     |

## Usage Examples

This section provides simple YOLO12 training and inference examples. For full documentation on these and other [modes](../modes/index.md), see the [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md), and [Export](../modes/export.md) docs pages.

Note that the example below is for YOLO12 [Detect](../tasks/detect.md) models for [object detection](https://www.ultralytics.com/glossary/object-detection). For additional supported tasks, see the [Segment](../tasks/segment.md), [Classify](../tasks/classify.md), [OBB](../tasks/obb.md), and [Pose](../tasks/pose.md) docs.

!!! example

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) pretrained `*.pt` models as well as configuration `*.yaml` files can be passed to the `YOLO()` class to create a model instance in Python:

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

        CLI commands are available to directly run the models:

        ```bash
        # Load a COCO-pretrained YOLO12n model and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolo12n.pt data=coco8.yaml epochs=100 imgsz=640

        # Load a COCO-pretrained YOLO12n model and run inference on the 'bus.jpg' image
        yolo predict model=yolo12n.pt source=path/to/bus.jpg
        ```

## Key Improvements

1. **Enhanced Feature Extraction**:

    - Area Attention mechanism for efficient large receptive field processing
    - Optimized balance between attention and feed-forward networks
    - Improved feature aggregation through R-ELAN architecture

2. **Optimization Innovations**:

    - Residual connections with scaling techniques for stable training
    - Refined feature integration methods
    - Efficient memory access through FlashAttention integration

3. **Architectural Efficiency**:
    - Reduced parameter count while maintaining or improving accuracy
    - Streamlined attention implementation without positional encoding
    - Optimized MLP ratios for better computational resource allocation

## Requirements

Ultralytics YOLO12 implementation does not require FlashAttention support. To compile FlashAttention support for YOLO12 the available NVIDIA GPUs are available:

- Turing GPUs (T4, Quadro RTX series)
- Ampere GPUs (RTX30 series, A30/40/100)
- Ada Lovelace GPUs (RTX40 series)
- Hopper GPUs (H100/H200)

## Citations and Acknowledgements

If you use YOLO12 in your research, please cite:

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
