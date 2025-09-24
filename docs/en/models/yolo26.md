---
comments: true
description: Discover YOLO26, the next evolution in real-time object detection with simplified architecture, superior performance, and remarkable efficiency for diverse computer vision tasks.
keywords: YOLO26, object detection, simplified architecture, YOLO series, Ultralytics, computer vision, AI, machine learning, edge deployment, quantization
---

# Ultralytics YOLO26

## Overview

YOLO26 is the next evolution in the [Ultralytics](https://www.ultralytics.com/) YOLO series of real-time object detectors, featuring a streamlined architecture that delivers superior performance with remarkable efficiency. By removing architectural complexity while introducing targeted enhancements, YOLO26 achieves the optimal balance of speed, [accuracy](https://www.ultralytics.com/glossary/accuracy), and deployability for a wide range of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks.

![Ultralytics YOLO26 Comparison Plots](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo-comparison-plot.png)

## Key Features

### Core Innovation: Simplified Excellence

- **DFL Removal**: Unlocks higher performance with simpler inference
- **End-to-End NMS-Free Inference**: Streamlined deployments
- **ProgLoss + STAL**: Improved accuracy, especially for small objects
- **MuSGD Optimizer**: Faster, smarter training
- **43% Faster CPU Inference**: Edge-ready efficiency

## Supported Tasks and Modes

YOLO26 continues the versatile model range tradition, offering enhanced support across various computer vision tasks:

| Model        | Task                                         | Inference | Validation | Training | Export |
| ------------ | -------------------------------------------- | --------- | ---------- | -------- | ------ |
| YOLO26       | [Detection](../tasks/detect.md)              | ✅        | ✅         | ✅       | ✅     |
| YOLO26-seg   | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLO26-pose  | [Pose/Keypoints](../tasks/pose.md)           | ✅        | ✅         | ✅       | ✅     |
| YOLO26-obb   | [Oriented Detection](../tasks/obb.md)        | ✅        | ✅         | ✅       | ✅     |
| YOLO26-cls   | [Classification](../tasks/classify.md)       | ✅        | ✅         | ✅       | ✅     |

This table provides an overview of the YOLO26 model variants, showcasing their applicability in specific tasks and compatibility with operational modes such as Inference, Validation, Training, and Export. This flexibility makes YOLO26 suitable for a wide range of applications in computer vision, from real-time detection to complex segmentation tasks.

## Performance Metrics

!!! tip "Performance"

    === "Detection (COCO)"

        See [Detection Docs](../tasks/detect.md) for usage examples with these models trained on [COCO](../datasets/detect/coco.md), which include 80 pre-trained classes.

        | Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95(e2e) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms)  | Speed<br><sup>T4 TensorRT10<br>(ms)  | params<br><sup>(M) | FLOPs<br><sup>(B) |
        | -----   | --------------------- | --------------------      | -------------------- | ------------------------------- | ------------------------------------ | ------------------ | ----------------- |
        | YOLO26n | 640                   | 39.8                      | 40.3                 | 38.90 ± 0.7                     | 1.7 ± 0.0                            | 2.4                | 5.4               |
        | YOLO26s | 640                   | 47.2                      | 47.6                 | 87.16 ± 0.9                     | 2.7 ± 0.0                            | 9.5                | 20.7                 |
        | YOLO26m | 640                   | 51.5                      | 51.7                 | 220.0 ± 1.4                     | 4.9 ± 0.1                            | 20.4               | 68.2                 |
        | YOLO26l | 640                   | 53.0                      | 53.4                 | 286.17 ± 2.0                    | 6.5 ± 0.2                            | 24.8               | 86.4                 |
        | YOLO26x | 640                   | -                         | -                    | -                               | -                                    | -                  | -                 |

    === "Segmentation (COCO)"

        See [Segmentation Docs](../tasks/segment.md) for usage examples with these models trained on [COCO](../datasets/segment/coco.md), which include 80 pre-trained classes.

        Performance metrics for segmentation models coming soon.

    === "Classification (ImageNet)"

        See [Classification Docs](../tasks/classify.md) for usage examples with these models trained on [ImageNet](../datasets/classify/imagenet.md), which include 1000 pre-trained classes.

        Performance metrics for classification models coming soon.

    === "Pose (COCO)"

        See [Pose Estimation Docs](../tasks/pose.md) for usage examples with these models trained on [COCO](../datasets/pose/coco.md), which include 1 pre-trained class, 'person'.

        Performance metrics for pose models coming soon.

    === "OBB (DOTAv1)"

        See [Oriented Detection Docs](../tasks/obb.md) for usage examples with these models trained on [DOTAv1](../datasets/obb/dota-v2.md#dota-v10), which include 15 pre-trained classes.

        Performance metrics for OBB models coming soon.


## Citations and Acknowledgements

!!! tip "Ultralytics YOLO26 Publication"

    Ultralytics has not published a formal research paper for YOLO26 due to the rapidly evolving nature of the models. We focus on advancing the technology and making it easier to use, rather than producing static documentation. For the most up-to-date information on YOLO architecture, features, and usage, please refer to our [GitHub repository](https://github.com/ultralytics/ultralytics) and [documentation](https://docs.ultralytics.com/).

If you use YOLO26 or any other software from this repository in your work, please cite it using the following format:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @software{yolo26_ultralytics,
          author = {Glenn Jocher and Jing Qiu},
          title = {Ultralytics YOLO26},
          version = {26.0.0},
          year = {2025},
          url = {https://github.com/ultralytics/ultralytics},
          orcid = {0000-0001-5950-6979, 0000-0003-3783-7069},
          license = {AGPL-3.0}
        }
        ```

Please note that the DOI is pending and will be added to the citation once it is available. YOLO26 models are provided under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) and [Enterprise](https://www.ultralytics.com/license) licenses.

## FAQ

### What are the key improvements in Ultralytics YOLO26 compared to YOLO11?

Ultralytics YOLO26 introduces several breakthrough improvements:

- **DFL Removal**: Unlocks higher performance with simpler inference
- **End-to-End NMS-Free Inference**: Streamlined deployments
- **ProgLoss + STAL**: Improved accuracy, especially for small objects
- **MuSGD Optimizer**: Faster, smarter training
- **43% Faster CPU Inference**: Edge-ready efficiency

### What tasks can YOLO26 models perform?

YOLO26 models are versatile and support multiple computer vision tasks:

- **Object Detection:** Identify and locate objects within images
- **Instance Segmentation:** Detect objects with pixel-perfect boundaries
- **Image Classification:** Categorize entire images into predefined classes
- **Pose Estimation:** Detect human keypoints and poses
- **Oriented Object Detection (OBB):** Detect rotated objects for aerial/satellite imagery

Each YOLO26 model size (n, s, m, l, x) supports all these tasks out-of-the-box, making it a unified solution for diverse vision applications.

### Why is YOLO26 better for edge deployment?

YOLO26 is specifically optimized for edge deployment through several key advantages:

- **Faster CPU Performance:** Up to 43% improvement in CPU inference speed, crucial for devices without GPUs
- **Smaller Model Size:** 11.5% fewer parameters mean reduced memory footprint
- **Simplified Architecture:** Removal of complex components like DFL module means fewer deployment issues
- **Flexible Export Options:** Full support for TensorRT, ONNX, CoreML, TFLite, and OpenVINO

These improvements make YOLO26 ideal for deployment on resource-constrained devices while maintaining high accuracy.

Note that while YOLO26n metrics are final, other model sizes currently show estimated performance values that will be updated with final benchmarks.
