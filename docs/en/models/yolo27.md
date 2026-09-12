---
title: Ultralytics YOLO27 - Coming Soon
comments: true
description: Preview Ultralytics YOLO27, an upcoming real-time vision model family undergoing final R&D. Models are not yet available, and no launch date is set.
keywords: YOLO27, Ultralytics YOLO, coming soon, object detection, NMS-free, computer vision, real-time inference
---

!!! info "YOLO27 is coming soon"

    Ultralytics YOLO27 models are undergoing final R&D, with a launch anticipated later this year. **The models are not yet available, and no launch date has been set.** This page is a preview; model weights, configurations, and implementation code are not being released at this time. Features and performance may change before launch.

# Ultralytics YOLO27

## Overview

[Ultralytics](https://www.ultralytics.com) YOLO27 is an upcoming family of real-time vision models. Its detection models use two complementary designs: a streamlined CNN architecture for the compact N and S models, and a query-based, NMS-free architecture for the larger M and L models.

The planned family includes four sizes — N, S, M, and L — and seven tasks: [object detection](../tasks/detect.md), [instance segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [oriented object detection](../tasks/obb.md). The two-design split applies to detection; the other tasks use the CNN architecture in the current research design.

## Preview Features

The current research design focuses on the following improvements. These are preview details, not a final release specification.

- **Efficient compact models:** YOLO27 N and S use dual-scale detection to reduce detection-head computation, with an emphasis on small-object accuracy.
- **Training improvements:** Foreground alignment supervision aims to narrow the accuracy gap between training-time supervision and the inference-time detection head without adding inference overhead.
- **Query-based NMS-free detection:** YOLO27 M and L use a transformer decoder to produce final detections without separate non-maximum suppression.
- **A shared interface:** The family is being developed for the familiar Ultralytics training, validation, prediction, and export workflow. Public package support will accompany the release.

## Planned Model Sizes

| Model   | Research focus                                         |
| ------- | ------------------------------------------------------ |
| YOLO27n | Compact detection for edge devices and real-time video |
| YOLO27s | Small-model accuracy with efficient inference          |
| YOLO27m | A balance of accuracy and GPU inference speed          |
| YOLO27l | Higher accuracy for demanding vision applications      |

All sizes are **unreleased**. Final task coverage and deployment support will be documented at launch.

## Preliminary Performance Results

Internal research results currently span **42.3–60.4 mAP on COCO** across the four detection sizes at a 640-pixel input, with **0.62–2.32 ms TensorRT latency on an NVIDIA RTX PRO 6000**. YOLO27l has reached **61.2 mAP** at an 800-pixel input, exceeding 60 mAP in these preliminary evaluations.

Accuracy is measured on the COCO validation set, and GPU latency uses TensorRT 11 with FP16 precision. These results are preliminary and may change during final R&D. They cannot yet be reproduced with publicly available YOLO27 weights; final benchmarks and reproduction instructions will accompany the release.

CPU latency is measured on an AMD EPYC 9655 using ONNX Runtime with FP32 precision. Parameters and FLOPs describe fused inference models after Conv/BatchNorm folding and removal of unused detection branches. Detection speed uses the NMS-free head. All tables below describe unreleased research models.

### Detection (COCO)

| Model   | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>RTX PRO 6000 TensorRT11<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ------- | --------------------------- | -------------------------- | ------------------------------------ | --------------------------------------------------- | ------------------------ | ----------------------- |
| YOLO27n | 640                         | 42.3                       | 16.1 ± 1.6                           | 0.62 ± 0.00                                         | 3.0                      | 7.2                     |
| YOLO27s | 640                         | 49.6                       | 33.2 ± 0.1                           | 0.79 ± 0.00                                         | 11.8                     | 28.2                    |
| YOLO27m | 640                         | 55.8                       | 67.2 ± 0.3                           | 1.39 ± 0.00                                         | 22.8                     | 65.0                    |
| YOLO27l | 640                         | **60.4**                   | 149.6 ± 1.4                          | 2.32 ± 0.00                                         | 72.3                     | 165.3                   |

### Instance Segmentation (COCO)

| Model       | size<br><sup>(pixels)</sup> | mAP<sup>box<br>50-95</sup> | mAP<sup>mask<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>RTX PRO 6000 TensorRT11<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ----------- | --------------------------- | -------------------------- | --------------------------- | ------------------------------------ | --------------------------------------------------- | ------------------------ | ----------------------- |
| YOLO27n-seg | 640                         | 41.8                       | 35.4                        | 22.0 ± 2.2                           | 0.73 ± 0.00                                         | 3.0                      | 11.2                    |
| YOLO27s-seg | 640                         | 50.0                       | 42.5                        | 45.8 ± 0.1                           | 0.96 ± 0.00                                         | 11.6                     | 43.1                    |
| YOLO27m-seg | 640                         | 53.2                       | 45.0                        | 109.3 ± 0.9                          | 1.46 ± 0.00                                         | 25.4                     | 139.6                   |
| YOLO27l-seg | 640                         | 57.7                       | 47.9                        | 248.3 ± 1.3                          | 2.86 ± 0.00                                         | 67.5                     | 361.9                   |

### Semantic Segmentation (Cityscapes)

| Model       | size<br><sup>(pixels)</sup> | mIoU<sup>val</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>RTX PRO 6000 TensorRT11<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ----------- | --------------------------- | ------------------ | ------------------------------------ | --------------------------------------------------- | ------------------------ | ----------------------- |
| YOLO27n-sem | 1024 &times; 2048           | 78.8               | 95.6 ± 3.0                           | 0.89 ± 0.00                                         | 2.0                      | 34.0                    |
| YOLO27s-sem | 1024 &times; 2048           | 81.2               | 169.6 ± 0.9                          | 1.43 ± 0.00                                         | 7.8                      | 131.5                   |
| YOLO27m-sem | 1024 &times; 2048           | 82.4               | 336.9 ± 2.8                          | 2.71 ± 0.00                                         | 16.1                     | 385.4                   |
| YOLO27l-sem | 1024 &times; 2048           | 83.8               | 787.1 ± 2.2                          | 6.43 ± 0.00                                         | 44.8                     | 1085.8                  |

### Depth Estimation

| Model         | size<br><sup>(pixels)</sup> | delta1<sup>NYU</sup> | delta1<sup>KITTI-580</sup> | delta1<sup>bench mean</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>RTX PRO 6000 TensorRT11<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ------------- | --------------------------- | -------------------- | -------------------------- | --------------------------- | ------------------------------------ | --------------------------------------------------- | ------------------------ | ----------------------- |
| YOLO27n-depth | 768                         | 0.8314               | 0.8256                     | 0.7238                      | 47.2 ± 4.0                           | 0.79 ± 0.00                                         | 5.4                      | 49.3                    |
| YOLO27s-depth | 768                         | 0.8682               | 0.7835                     | 0.7454                      | 72.1 ± 0.5                           | 0.97 ± 0.00                                         | 13.0                     | 77.3                    |
| YOLO27m-depth | 768                         | 0.8652               | 0.7746                     | 0.7476                      | 120.2 ± 0.4                          | 1.34 ± 0.00                                         | 23.4                     | 143.6                   |
| YOLO27l-depth | 768                         | 0.8711               | 0.8041                     | 0.7527                      | 253.7 ± 0.5                          | 2.65 ± 0.00                                         | 59.3                     | 341.8                   |

### Classification

| Model       | size<br><sup>(pixels)</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>RTX PRO 6000 TensorRT11<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B) at 224</sup> |
| ----------- | --------------------------- | ------------------------------------ | --------------------------------------------------- | ------------------------ | ------------------------------ |
| YOLO27n-cls | 224                         | 1.9 ± 0.0                            | 0.28 ± 0.00                                         | 2.9                      | 0.5                            |
| YOLO27s-cls | 224                         | 3.1 ± 0.1                            | 0.31 ± 0.00                                         | 6.9                      | 1.7                            |
| YOLO27m-cls | 224                         | 6.0 ± 0.6                            | 0.39 ± 0.00                                         | 12.4                     | 6.1                            |
| YOLO27l-cls | 224                         | 16.7 ± 0.1                           | 0.69 ± 0.00                                         | 32.8                     | 18.5                           |

### Pose (COCO)

| Model        | size<br><sup>(pixels)</sup> | mAP<sup>pose<br>50-95</sup> | mAP<sup>pose<br>50</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>RTX PRO 6000 TensorRT11<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ------------ | --------------------------- | --------------------------- | ------------------------ | ------------------------------------ | --------------------------------------------------- | ------------------------ | ----------------------- |
| YOLO27n-pose | 640                         | 58.0                        | 84.1                     | 19.6 ± 0.1                           | 0.67 ± 0.00                                         | 3.1                      | 9.2                     |
| YOLO27s-pose | 640                         | 64.3                        | 87.0                     | 36.6 ± 0.2                           | 0.80 ± 0.00                                         | 11.2                     | 31.1                    |
| YOLO27m-pose | 640                         | 69.0                        | 89.9                     | 76.6 ± 0.2                           | 1.29 ± 0.00                                         | 22.8                     | 86.2                    |
| YOLO27l-pose | 640                         | 72.2                        | 91.3                     | 181.1 ± 0.7                          | 2.45 ± 0.00                                         | 61.7                     | 242.3                   |

### Oriented Detection (DOTAv1)

| Model       | size<br><sup>(pixels)</sup> | mAP<sup>test<br>50-95</sup> | mAP<sup>test<br>50</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>RTX PRO 6000 TensorRT11<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ----------- | --------------------------- | --------------------------- | ------------------------ | ------------------------------------ | --------------------------------------------------- | ------------------------ | ----------------------- |
| YOLO27n-obb | 1024                        | 54.0                        | 80.3                     | 42.6 ± 0.2                           | 0.83 ± 0.00                                         | 3.0                      | 19.7                    |
| YOLO27s-obb | 1024                        | 55.8                        | 81.6                     | 88.2 ± 0.6                           | 1.30 ± 0.00                                         | 10.9                     | 76.6                    |
| YOLO27m-obb | 1024                        | 55.9                        | 82.6                     | 188.0 ± 0.7                          | 1.92 ± 0.00                                         | 22.8                     | 221.7                   |
| YOLO27l-obb | 1024                        | 57.2                        | 82.7                     | 455.8 ± 1.1                          | 4.33 ± 0.00                                         | 68.7                     | 622.6                   |

Classification accuracy is not yet reported in this preview.

## Availability

YOLO27 weights and model implementations are **not publicly available**. Installing or updating the `ultralytics` package does not provide access to YOLO27, and there are no public YOLO27 download or inference instructions yet.

For training and deployment today, use [YOLO26](yolo26.md). This page will be updated with supported tasks, benchmarks, and usage instructions when YOLO27 is released.

## FAQ

### When will YOLO27 launch?

YOLO27 is undergoing final R&D, with a launch anticipated later this year. No specific date has been set, and timing remains subject to research and validation progress.

### Can I download or run YOLO27 now?

No. YOLO27 models are not yet available. This documentation preview does not release model weights, configurations, or implementation code.

### Should I upgrade from YOLO26 now?

Continue using [YOLO26](yolo26.md) for current projects. YOLO27 migration guidance and usage examples will be published when the models become available.

### Are the preview features and benchmarks final?

No. The architecture, task coverage, performance, and deployment support are still being finalized. Treat the information on this page as a research preview.
