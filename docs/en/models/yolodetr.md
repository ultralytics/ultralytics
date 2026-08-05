---
title: YOLO-DETR
comments: true
description: Learn about Ultralytics YOLO-DETR object detection models with YOLO26-style CSP and UltraViT backbones, DeimDecoder heads, training, inference, validation, and export.
keywords: YOLO-DETR, YOLODETR, YOLO27-DETR, Ultralytics, object detection, DETR, DeimDecoder, UltraViT, HybridEncoder, YOLO26, NMS-free, real-time detection
---

# Ultralytics YOLO-DETR

## Overview

Ultralytics YOLO-DETR combines YOLO feature extractors with a transformer decoder in the standard Ultralytics
workflow. It uses the dedicated `YOLODETR` Python class and supports [Train](../modes/train.md),
[Val](../modes/val.md), [Predict](../modes/predict.md), and [Export](../modes/export.md) modes.

Two model configurations are available:

- **YOLO27l** uses a YOLO26-style CSP backbone, an FPN/PAN neck, and a 4-layer `DeimDecoder`.
- **YOLO27x** uses an UltraViT backbone, a HybridEncoder neck, and a 6-layer `DeimDecoder`.

Both models produce query-based predictions for NMS-free object detection with 300 decoder queries by default.

## Key Features

- **NMS-Free Query-Based Detection**
  Predictions come from a fixed set of 300 object queries matched one-to-one with ground truth during training, so the standard inference path returns final boxes directly without non-maximum suppression. Latency does not vary with the number of objects in the image.

- **`DeimDecoder` Head with Distribution-Based Box Regression**
  Box edges are predicted as discrete distributions over `reg_max + 1` bins (33 in both shipped configs) and integrated back into coordinates, rather than regressed as single values. Training therefore adds a Distribution Focal Loss term (`fgl_loss`) alongside the classification, L1, and GIoU losses, plus a distillation term (`ddf_loss`) that transfers each decoder layer's distribution to the next.

- **UltraViT Backbone (YOLO27x)**
  An efficient hybrid backbone that keeps FastViT-style reparameterized convolution blocks at the three high-resolution stages and places global self-attention only at the coarsest P5/32 stage, where the token count is smallest. The convolutional blocks fold their training-time branches, residuals, and normalization into single convolutions for deployment, and the attention stage uses scaled dot-product attention directly to keep exported graphs compact.

## Model Variants

| Model   | Config         | Backbone         | Neck          | Decoder       | Decoder Layers |
| ------- | -------------- | ---------------- | ------------- | ------------- | -------------- |
| YOLO27l | `yolo27l.yaml` | YOLO26-style CSP | FPN/PAN       | `DeimDecoder` | 4              |
| YOLO27x | `yolo27x.yaml` | UltraViT         | HybridEncoder | `DeimDecoder` | 6              |

## Supported Tasks and Modes

| Model Family | Task                                      | Inference | Validation | Training | Export |
| ------------ | ----------------------------------------- | --------- | ---------- | -------- | ------ |
| YOLO-DETR    | [Object Detection](../tasks/detect.md)    | Yes       | Yes        | Yes      | Yes    |

## Performance Metrics

!!! tip "Performance"

    === "Detection (COCO)"

        | Model   | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | ------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
        | YOLO27l | 640                         |                            | 335.6 ± 2.6                          | 6.8 ± 0.1                                 | 30.3                     | 86.0                    |
        | YOLO27x | 640                         |                            | 649.7 ± 6.1                          | 12.8 ± 0.2                                | 65.5                     | 176.0                   |

_Parameters and FLOPs are measured from the exported ONNX graphs at `imgsz=640`. FLOPs count one multiply-add as two
floating-point operations._

## Usage Examples

!!! example

    === "Python"

        ```python
        from ultralytics import YOLODETR

        # Build a YOLO27l model from YAML
        model = YOLODETR("yolo27l.yaml")

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

YOLO-DETR accepts standard Ultralytics training arguments such as `data`, `epochs`, `imgsz`, `batch`, `optimizer`,
`lr0`, `lrf`, and augmentation probabilities. It also supports these trainer-specific arguments through
`model.train(...)`:

| Argument            | Default | Description                                                                |
| ------------------- | ------- | -------------------------------------------------------------------------- |
| `no_aug_epoch`      | `4`     | Number of final epochs in which augmentation is disabled                   |
| `backbone_lr_ratio` | `0.02`  | Multiplier applied to backbone parameter-group learning rates              |
| `base_size_repeat`  | `3`     | Extra weight given to the base image size during multi-scale sampling      |

Recommended starting settings are:

| Model   | Config         | `lr0`    | `backbone_lr_ratio` | `weight_decay` |
| ------- | -------------- | -------- | ------------------- | -------------- |
| YOLO27l | `yolo27l.yaml` | `0.0005` | `0.025`             | `0.000125`     |
| YOLO27x | `yolo27x.yaml` | `0.0005` | `0.02`              | `0.000125`     |

```python
from ultralytics import YOLODETR

model = YOLODETR("yolo27x.yaml")
model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    optimizer="AdamW",
    lr0=0.0005,
    weight_decay=0.000125,
    backbone_lr_ratio=0.02,
)
```

## Inference and Export Notes

YOLO-DETR uses 300 decoder queries by default. Increasing `max_det` does not create additional queries; change the
query count in the model YAML and retrain if the dataset can contain more than 300 objects per image.

Decoder depth is part of each architecture: YOLO27l uses 4 layers and YOLO27x uses 6. Export preserves the
selected architecture and decoder behavior.

## FAQ

### How is YOLO-DETR different from other YOLO models?

Standard YOLO models predict on dense feature grids. YOLO-DETR instead uses a transformer decoder with a fixed set of
object queries, producing NMS-free predictions in its standard inference path.
