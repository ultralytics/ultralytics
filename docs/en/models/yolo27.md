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

The following preliminary measurements were reported on the COCO validation set at 640-pixel input. Latency was
measured on an NVIDIA T4; parameters and FLOPs are measured on the fused deployment graphs.

| Model   | mAP<sup>val<br>50-95</sup> | Latency<br><sup>T4 (ms)</sup> | Params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ------- | -------------------------- | ----------------------------- | ------------------------ | ----------------------- |
| YOLO27n | 41.6                       | 1.8                           | 3.0                      | 7.2                     |
| YOLO27s | 49.2                       | 2.7                           | 11.8                     | 28.2                    |
| YOLO27m | 55.7                       | 4.4                           | 27.2                     | 83.4                    |
| YOLO27l | 57.7                       | 6.5                           | 30.4                     | 85.4                    |
| YOLO27x | **60.5**                   | **12.3**                      | 65.6                     | 173.1                   |

## Current Scope

YOLO27 results currently cover object detection. Segmentation and classification configurations derived from the
compact CNN architecture exist as research configs, and support for additional tasks will be documented when the
corresponding models are finalized.

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
