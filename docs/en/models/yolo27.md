---
title: YOLO27
comments: true
description: Learn about Ultralytics YOLO27 object detection models with YOLO26-style CSP and UltraViT backbones, DeimDecoder heads, training, inference, validation, and export.
keywords: YOLO27, Ultralytics, object detection, DeimDecoder, UltraViT, HybridEncoder, YOLO26, NMS-free, real-time detection
---

# Ultralytics YOLO27

## Overview

Ultralytics YOLO27 is loaded through the standard `YOLO` Python class and supports [Train](../modes/train.md),
[Val](../modes/val.md), [Predict](../modes/predict.md), and [Export](../modes/export.md) modes.

The medium, large, and extra-large configurations pair a YOLO feature extractor with a transformer decoder, so they
produce query-based predictions instead of dense grid predictions:

- **YOLO27m** uses a YOLO26-style CSP backbone, an FPN/PAN neck, and a 2-layer `DeimDecoder`.
- **YOLO27l** uses a YOLO26-style CSP backbone, an FPN/PAN neck, and a 4-layer `DeimDecoder`.
- **YOLO27x** uses an UltraViT backbone, a HybridEncoder neck, and a 6-layer `DeimDecoder`.

All three models produce query-based predictions for NMS-free object detection with 300 decoder queries by default.

## Key Features

- **NMS-Free Query-Based Detection**
  Predictions come from a fixed set of 300 object queries matched one-to-one with ground truth during training, so the standard inference path returns final boxes directly without non-maximum suppression. Latency does not vary with the number of objects in the image.

- **`DeimDecoder` Head with Distribution-Based Box Regression**
  Box edges are predicted as discrete distributions over `reg_max + 1` bins (33 in all three shipped configs) and integrated back into coordinates, rather than regressed as single values. Training therefore adds a Distribution Focal Loss term (`fgl_loss`) alongside the classification, L1, and GIoU losses, plus a distillation term (`ddf_loss`) that transfers each decoder layer's distribution to the next.

- **UltraViT Backbone (YOLO27x)**
  An efficient hybrid backbone that keeps FastViT-style reparameterized convolution blocks at the three high-resolution stages and places global self-attention only at the coarsest P5/32 stage, where the token count is smallest. The convolutional blocks fold their training-time branches, residuals, and normalization into single convolutions for deployment, and the attention stage uses scaled dot-product attention directly to keep exported graphs compact.

## Model Variants

| Model   | Config         | Backbone         | Neck          | Decoder       | Decoder Layers |
| ------- | -------------- | ---------------- | ------------- | ------------- | -------------- |
| YOLO27m | `yolo27m.yaml` | YOLO26-style CSP | FPN/PAN       | `DeimDecoder` | 2              |
| YOLO27l | `yolo27l.yaml` | YOLO26-style CSP | FPN/PAN       | `DeimDecoder` | 4              |
| YOLO27x | `yolo27x.yaml` | UltraViT         | HybridEncoder | `DeimDecoder` | 6              |

## Supported Tasks and Modes

| Model Family | Task                                      | Inference | Validation | Training | Export |
| ------------ | ----------------------------------------- | --------- | ---------- | -------- | ------ |
| YOLO27       | [Object Detection](../tasks/detect.md)    | Yes       | Yes        | Yes      | Yes    |

## Performance Metrics

!!! tip "Performance"

    === "Detection (COCO)"

        | Model             | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | ----------------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
        | yolo27m           | 512                         | -                          | 158.9 ± 7.5                          | 4.4 ± 0.1                                 | 27.2                     | 54.1                    |
        | yolo27l           | 640                         | -                          | 315.6 ± 25.7                         | 6.5 ± 0.1                                 | 30.4                     | 85.4                    |
        | yolo27x           | 640                         | -                          | 416.5 ± 15.8                         | 12.3 ± 0.3                                | 65.6                     | 173.1                   |

_Parameters and FLOPs are reported from the export-mode models after deployment fusion and reparameterization, at the
image size shown for each model. FLOPs count one multiply-add as two floating-point operations._

## Usage Examples

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Build a YOLO27l model from YAML
        model = YOLO("yolo27l.yaml")

        # Train and run inference
        model.train(data="coco8.yaml", epochs=100, imgsz=640)
        results = model("path/to/image.jpg")
        ```

    === "CLI"

        ```bash
        yolo train model=yolo27l.yaml data=coco8.yaml epochs=100 imgsz=640
        yolo predict model=yolo27l.yaml source=path/to/image.jpg
        ```

## Training Notes

YOLO27 accepts standard Ultralytics training arguments such as `data`, `epochs`, `imgsz`, `batch`, `optimizer`,
`lr0`, `lrf`, and augmentation probabilities. Augmentation is disabled for the final four epochs of every run, which
follows DEIM. The backbone learning rate is fixed at `0.1` times the head learning rate.

YOLO27 models can be trained with the standard defaults in the repository without supplying these overrides. However,
the following settings produced the best fine-tuning performance for pretrained YOLO27m, YOLO27l, and YOLO27x models
in our experiments. Use `imgsz=512` for YOLO27m and `imgsz=640` for YOLO27l and YOLO27x.

```python
from ultralytics import YOLO

model = YOLO("yolo27l.pt")
model.train(
    data="path/to/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer="AdamW",
    lr0=0.0005,
    lrf=0.5,
    weight_decay=0.000125,
    warmup_epochs=0.27,
    momentum=0.9,
    warmup_momentum=0.9,
    warmup_bias_lr=0.0,
    nbs=16,
    deterministic=False,
    mosaic=0.5,
    mixup=0.5,
    copy_paste=0.5,
    scale=0.9,
)
```

Set `deterministic=False` for faster training. DETR deformable attention uses `grid_sample`, whose CUDA backward path
does not support deterministic execution, so fully deterministic CUDA training is not available for this architecture.

## Inference and Export Notes

YOLO27m, YOLO27l, and YOLO27x use 300 decoder queries by default. Increasing `max_det` does not create additional
queries; change the query count in the model YAML and retrain if the dataset can contain more than 300 objects per
image.

Decoder depth is part of each architecture: YOLO27m uses 2 layers, YOLO27l uses 4, and YOLO27x uses 6. Export preserves
the selected architecture and decoder behavior.

## FAQ

### How are YOLO27m, YOLO27l, and YOLO27x different from other YOLO models?

Standard YOLO models predict on dense feature grids. YOLO27m, YOLO27l, and YOLO27x instead use a transformer decoder
with a fixed set of object queries, producing NMS-free predictions in their standard inference path.
