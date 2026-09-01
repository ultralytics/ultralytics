---
comments: true
description: YOLO27 is an in-development Ultralytics research family pairing a streamlined two-scale CNN design for small models with query-based NMS-free detection for large models.
keywords: YOLO27, Ultralytics YOLO, object detection, NMS-free, end-to-end detection, small object detection, computer vision, AI
---

# Ultralytics YOLO27

## Overview

YOLO27 is an in-development Ultralytics object detection research family. It explores two complementary designs:
a streamlined CNN architecture for the small N and S models, and a query-based, NMS-free architecture for the larger
M, L, and X models. The results below are preliminary COCO validation measurements, not released model checkpoints.

Across the five scales, YOLO27 reaches **41.6-60.5 mAP on COCO** at **1.8-12.3 ms latency on an NVIDIA T4**. The
small models are built for speed, while the larger models trade compute for accuracy — YOLO27x sets the family
record at **60.5 mAP**, the strongest result in the current research line.

!!! warning "Research preview"

    YOLO27 is under active development. Architectures, training recipes, supported tasks, weights, and deployment
    behavior may change before release. This page documents the current detection research only.

## Key Features

- **Streamlined two-scale detection (N/S)**
  Standard detectors predict objects on three feature maps — fine, medium, and coarse. YOLO27 N and S drop the
  medium one and predict only on a fine map (for small objects) and a coarse map (for large objects). This removes a
  large chunk of detection-head computation, making the models faster while keeping accuracy competitive.

- **Stronger small-object detection (N/S)**
  The early, high-resolution feature stage is widened so it can capture more fine-grained detail. Combined with the
  surviving fine prediction map, this improves localization and regression for small objects — the hardest category
  for compact models.

- **Stable two-scale training (N/S)**
  A fixed scaling on the fused features keeps the two prediction scales balanced during training, so the simplified
  architecture trains as reliably as the full three-scale design.

- **Auxiliary foreground supervision (N/S)**
  During training, an extra lightweight branch learns to tell "object" from "background" at every location, giving
  the backbone and neck a stronger learning signal early on and gradually aligning with the final detection head.
  The branch is used only during training and is removed for inference and export, so it costs nothing at deployment.

- **Query-based detection without NMS (M/L/X)**
  The larger models replace dense prediction with a transformer decoder that refines a fixed set of object queries
  and directly outputs the final detections — no non-maximum suppression post-processing needed. YOLO27m and YOLO27l
  pair this decoder with the proven YOLO26-style convolutional backbone, while YOLO27x adds an UltraViT backbone that
  uses self-attention in its deepest stage to capture global context, plus a hybrid encoder that fuses features
  across scales.

- **Purpose-built training recipes**
  The small models train with MuSGD (the hybrid Muon + SGD optimizer introduced in YOLO26) with a tunable
  classification-head learning rate. The larger models train with a flat-then-cosine learning-rate schedule, a
  gentler learning rate for the pretrained backbone, and an augmentation schedule that fades out mosaic-style
  augmentations at the end of training for a cleaner final convergence.

- **One simple interface**
  Both architectures are used through the same `YOLO` class. The right training, validation, prediction, and export
  pipeline is selected automatically from the model, so code written for one YOLO27 scale works unchanged for the
  others.

## Performance Metrics

The following are preliminary research measurements. Detection accuracy is reported on the COCO validation set with
latency on an NVIDIA T4; segmentation and classification tables list model size and inference speed on CPU (ONNX) and
GPU (TensorRT).

=== "Detection (COCO)"

    YOLO27m is evaluated at a 512-pixel input; all other detection scales use 640 pixels.

    | Model   | Size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Latency<br><sup>T4 (ms)</sup> | Params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
    | ------- | --------------------------- | -------------------------- | ----------------------------- | ------------------------ | ----------------------- |
    | YOLO27n | 640                         | 41.6                       | 1.8                           | 3.0                      | 7.2                     |
    | YOLO27s | 640                         | 49.2                       | 2.7                           | 11.8                     | 28.2                    |
    | YOLO27m | 512                         | 55.7                       | 4.4                           | 27.2                     | 83.4                    |
    | YOLO27l | 640                         | 57.7                       | 6.5                           | 30.4                     | 85.4                    |
    | YOLO27x | 640                         | **60.5**                   | **12.3**                      | 65.6                     | 173.1                   |

=== "Segmentation (COCO)"

    Measured at a 640-pixel input.

    | Model       | Size<br><sup>(pixels)</sup> | CPU ONNX<br><sup>(ms)</sup> | TensorRT<br><sup>(ms)</sup> | Params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
    | ----------- | --------------------------- | --------------------------- | --------------------------- | ------------------------ | ----------------------- |
    | YOLO27n-seg | 640                         | 63.1 ± 3.1                  | **2.092 ± 0.051**           | 3.0                      | 11.1                    |
    | YOLO27s-seg | 640                         | 119.0 ± 5.0                 | **3.562 ± 0.046**           | 11.6                     | 42.9                    |
    | YOLO27m-seg | 640                         | 280.7 ± 15.1                | **7.409 ± 0.193**           | 25.4                     | 139.4                   |
    | YOLO27l-seg | 640                         | 343.0 ± 7.7                 | **9.198 ± 0.194**           | 30.0                     | 161.0                   |
    | YOLO27x-seg | 640                         | 672.7 ± 54.8                | **18.123 ± 0.545**          | 67.5                     | 361.3                   |

=== "Classification (ImageNet)"

    Measured at a 224-pixel input.

    | Model       | Size<br><sup>(pixels)</sup> | CPU ONNX<br><sup>(ms)</sup> | TensorRT<br><sup>(ms)</sup> | Params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
    | ----------- | --------------------------- | --------------------------- | --------------------------- | ------------------------ | ----------------------- |
    | YOLO27n-cls | 224                         | 4.2 ± 0.6                   | **1.131 ± 0.046**           | 2.9                      | 0.6                     |
    | YOLO27s-cls | 224                         | 7.3 ± 0.7                   | **1.412 ± 0.025**           | 6.9                      | 1.9                     |
    | YOLO27m-cls | 224                         | 16.8 ± 1.1                  | **1.905 ± 0.053**           | 12.4                     | 6.2                     |
    | YOLO27l-cls | 224                         | 24.8 ± 1.6                  | **1.935 ± 0.075**           | 15.5                     | 8.4                     |
    | YOLO27x-cls | 224                         | 44.1 ± 3.4                  | **1.951 ± 0.038**           | 32.8                     | 18.6                    |

## Current Scope

YOLO27 research currently covers object detection, instance segmentation, and classification. Support for additional
tasks will be documented when the corresponding models are finalized.

---

## FAQ

### Is YOLO27 released?

No. YOLO27 is an active research effort, and no released weights or stable production interface are documented here.

### Why do YOLO27 N and S predict on only two scales?

Most detectors predict on three feature maps at different resolutions. YOLO27 N and S keep the fine map that small
objects depend on and the coarse map that large objects need, and skip the medium one. This cuts a significant share
of detection-head computation, and the training improvements above keep the accuracy-latency tradeoff competitive.

### What makes the YOLO27x result notable?

YOLO27x is the reported accuracy leader, reaching 60.5 mAP on COCO validation at 12.3 ms T4 latency. It combines the
UltraViT backbone, multi-scale feature fusion, and a query-based detector that produces final detections directly,
without NMS.
