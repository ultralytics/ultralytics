---
comments: true
description: YOLO27 is an in-development Ultralytics research family combining compact P3/P5 CNN detection with auxiliary foreground supervision, and DEIM-decoder models with YOLO26 CSP or UltraViT backbones.
keywords: YOLO27, Ultralytics YOLO, object detection, UltraViT, DETR, DEIM, P3 P5 detection, computer vision, AI
---

# Ultralytics YOLO27

## Overview

YOLO27 is an in-development Ultralytics object detection research family. It combines two complementary directions:
compact CNN detection for the N and S scales, and a query-based DEIM-decoder design for the M, L, and X scales. The
reported results are preliminary COCO validation measurements, not released model checkpoints.

The compact models remove the P4/16 detection level and predict only from P3/8 and P5/32 features. At a 640-pixel
input, this changes the dense detection grid from 8,400 locations (80 × 80, 40 × 40, and 20 × 20) to 6,800 locations
(80 × 80 and 20 × 20), and the neck dams upsampled features with a fixed 0.5 scaling (SNI) before top-down fusion.
The design keeps the fine-resolution P3 grid for small objects while cutting the mid-level head computation.

The larger-model path uses multi-scale P3/P4/P5 features with a DEIM decoder. Its X design combines an UltraViT
backbone, a hybrid multi-scale encoder, and a fixed set of object queries to produce end-to-end detections.
**YOLO27x reaches 60.5 mAP on COCO validation at 12.3 ms on an NVIDIA T4**, the strongest reported accuracy in the
current research results.

!!! warning "Research preview"

    YOLO27 is under active development. Architectures, training recipes, supported tasks, weights, and deployment
    behavior may change before release. This page documents the current detection research only.

## Key Features

### Compact P3/P5 CNN Detection for N and S

YOLO27 N and S remove the P4 detection level from an otherwise YOLO26-style CNN. At 640 pixels the head runs on an
80 × 80 P3 grid and a 20 × 20 P5 grid — 6,800 candidate locations instead of the 8,400 of a P3/P4/P5 head — keeping
the fine grid that matters for small objects while dropping the mid-level head entirely. Upsampled features in the
neck are damped by a constant 0.5 scale (SNI) before fusion, which stabilizes the two-level topology, and the
backbone P3 stage width expands per scale through a named `bbp3e` scale parameter. Both sizes train with the MuSGD
optimizer and a tunable classification-head learning-rate multiplier.

### Auxiliary Foreground Supervision

The compact detection head attaches a training-only auxiliary branch that predicts one class-agnostic foreground
logit at every head input location. The branch is a small 3 × 3 convolution followed by a 1 × 1 prediction layer and
receives the same non-detached features as the detection head. It therefore sends direct deep-supervision gradients
into the backbone and neck rather than only learning its own output.

Its target follows the model's end-to-end training schedule (o2m): early in training, the target gives foreground
credit to the denser one-to-many assignment; as training progresses, the auxiliary emphasis shifts toward the
one-to-one positives used by the deployed head. This helps the shared features transition toward end-to-end
detection without treating one-to-many positives as background too early. The branch is declared in the model
configuration (`aux_fg: True`), and both the branch and its loss are absent from inference and export, so they add
no deployment cost.

### DEIM-Decoder M, L, and X Designs

The M, L, and X scales replace dense prediction with a query-based DEIM decoder — a D-FINE-style transformer decoder
with iterative bounding-box refinement, denoising queries, and one-to-many auxiliary matching during training. The
decoder refines a fixed set of 300 object queries against fused multi-scale features and directly returns the final
detections, avoiding a separate non-maximum-suppression stage. M and L pair a YOLO26-style CSP backbone and FPN/PAN
neck with 2-layer and 4-layer decoders; the layer count is the main capacity knob between them. Training uses a
flat-cosine learning-rate schedule, a separate backbone learning rate (`backbone_lr_ratio`), and an augmentation
schedule that decays mosaic/mixup/copy-paste to zero for a final no-augmentation stage.

### UltraViT Backbone for X

The X configuration pairs the DEIM decoder with UltraViT and a hybrid encoder neck. UltraViT's early and
intermediate stages use reparameterizable convolutional token mixers and convolutional feed-forward blocks, which
preserve efficient spatial processing at high resolution; the coarsest stage uses multi-head self-attention, where
global context is least expensive. The hybrid encoder then fuses P3, P4, and P5 features through top-down and
bottom-up paths before the decoder. It is the current accuracy leader of the reported family at **60.5 mAP** with
**12.3 ms** T4 latency.

### One Model Facade

Both architectures are served by the single `YOLO` class: the detect task routes to the standard CNN pipeline or the
DEIM pipeline automatically based on the model head, so training, validation, prediction, and export use the same
interface across all five scales.

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

### Why do YOLO27 N and S use only P3 and P5 outputs?

Dropping the P4 level removes the mid-sized 40 × 40 prediction grid at a 640-pixel input, leaving the fine 80 × 80
P3 grid for small objects and the coarse 20 × 20 P5 grid for large ones. SNI feature damping and auxiliary
foreground supervision maintain the accuracy-latency tradeoff of the two-level head.

### What makes the YOLO27x result notable?

YOLO27x is the reported accuracy leader, reaching 60.5 mAP on COCO validation at 12.3 ms T4 latency. It combines the
UltraViT backbone, a hybrid multi-scale encoder, and a 300-query DEIM decoder that produces end-to-end detections
without NMS.
