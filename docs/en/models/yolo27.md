---
comments: true
description: YOLO27 is an in-development Ultralytics research family combining compact P4/P5 detection, auxiliary foreground supervision, UltraViT backbones, and DETR-style detection.
keywords: YOLO27, Ultralytics YOLO, object detection, UltraViT, DETR, P4 P5 detection, computer vision, AI
---

# Ultralytics YOLO27

## Overview

YOLO27 is an in-development Ultralytics object detection research family. It combines two complementary directions:
compact P4/P5 dense detection for the N, S, and M scales, and a query-based DETR design for the L and X scales. The
reported results are preliminary COCO validation measurements, not released model checkpoints.

The compact models remove the P3/8 detection output and predict only from P4/16 and P5/32 features. At a 640-pixel
input, this changes the dense detection grid from 8,400 locations (80 × 80, 40 × 40, and 20 × 20) to 2,000 locations
(40 × 40 and 20 × 20). The design therefore concentrates detection work on semantically stronger features while
substantially reducing head computation.

The larger-model path uses multi-scale P3/P4/P5 features with a DETR decoder. Its current X design combines an
UltraViT backbone, a hybrid multi-scale encoder, and a fixed set of object queries to produce end-to-end detections.
**YOLO27x reaches 60.5 mAP on COCO validation at 12.8 ms on an NVIDIA T4**, the strongest reported accuracy in the
current research results.

!!! warning "Research preview"

    YOLO27 is under active development. Architectures, training recipes, supported tasks, weights, and deployment
    behavior may change before release. This page documents the current detection research only.

## Key Features

### Compact P4/P5 Detection for N, S, and M

YOLO27 N, S, and M remove the P3 detection head. At 640 pixels, P4 contributes a 40 × 40 grid and P5 contributes a
20 × 20 grid, giving 2,000 candidate locations instead of the 8,400 locations of a P3/P4/P5 head. This simplifies
the prediction head and lowers the amount of dense classification and box-regression work. The reported N, S, and M
models retain strong COCO validation accuracy at 42.0, 48.8, and 52.8 mAP, respectively.

### Auxiliary Foreground Supervision

The compact detection head can attach a training-only auxiliary branch that predicts one class-agnostic foreground
logit at every head input location. The branch is a small 3 × 3 convolution followed by a 1 × 1 prediction layer and
receives the same non-detached features as the detection head. It therefore sends direct deep-supervision gradients
into the backbone and neck rather than only learning its own output.

Its target is deliberately aligned with the model's end-to-end training schedule. Early in training, the target gives
foreground credit to the denser one-to-many assignment. As training progresses, ambiguous dense positives are reduced
until the auxiliary target emphasizes the one-to-one positives used by the deployed head. This helps the shared
features transition toward end-to-end detection without asking them to treat one-to-many positives as background too
early. The auxiliary branch and loss are absent from inference and export, so they add no deployment cost.

### UltraViT Backbone

UltraViT is the backbone direction explored across the YOLO27 research. Its early and intermediate stages use
reparameterized convolutional token mixers and convolutional feed-forward blocks, which preserve efficient spatial
processing at high resolution. The coarsest stage uses multi-head self-attention, where global context is less
expensive. This division gives the model local convolutional efficiency at fine scales and global feature interaction
at the deepest scale.

### DETR-Style L and X Designs

The L and X research path replaces dense prediction with a query-based DETR decoder. A hybrid encoder first aligns
and fuses P3, P4, and P5 features through top-down and bottom-up multi-scale paths. The decoder then refines a fixed
set of object queries against those fused features and directly returns the final detections, avoiding a separate
non-maximum-suppression stage.

The X configuration pairs this detector with UltraViT and uses 300 object queries. It is the current accuracy leader
of the reported family at **60.5 mAP**, while retaining **12.8 ms** T4 latency.

## Performance Metrics

The following preliminary measurements were reported on the COCO validation set at 640-pixel input. Latency was
measured on an NVIDIA T4; the exact runtime and precision configuration are not yet reported.

| Model   | mAP<sup>val<br>50-95</sup> | Latency<br><sup>T4 (ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ------- | -------------------------- | ----------------------------- | ------------------------ | ----------------------- |
| YOLO27n | 42.0                       | 1.8                           | TBD                      | TBD                     |
| YOLO27s | 48.8                       | 2.8                           | TBD                      | TBD                     |
| YOLO27m | 52.8                       | 5.3                           | TBD                      | TBD                     |
| YOLO27l | 57.6                       | 6.8                           | TBD                      | TBD                     |
| YOLO27x | **60.5**                   | **12.8**                      | TBD                      | TBD                     |

Parameters and FLOPs will be added after the final model configurations and fused deployment graphs are available.

## Current Scope

YOLO27 results currently cover object detection. Pretrained weights, complete parameter and FLOPs measurements, and
support for additional tasks will be documented when the corresponding models are finalized.

---

## FAQ

### Is YOLO27 released?

No. YOLO27 is an active research effort, and no released weights or stable production interface are documented here.

### Why do YOLO27 N, S, and M use only P4 and P5 outputs?

Using only P4 and P5 reduces dense prediction locations from 8,400 to 2,000 at a 640-pixel input. The architecture
then uses stronger P4/P5 features and auxiliary foreground supervision to maintain the accuracy-latency tradeoff.

### What makes the YOLO27x result notable?

YOLO27x is the reported accuracy leader, reaching 60.5 mAP on COCO validation at 12.8 ms T4 latency. It combines the
UltraViT backbone with multi-scale feature fusion and a query-based DETR detector.
