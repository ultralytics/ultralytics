---
title: YOLO-DETR
comments: true
description: Learn about Ultralytics YOLO-DETR object detection models with YOLO26-style CSP and DINOv3-ViT + STA backbones, DETR decoders, training, inference, validation, and export.
keywords: YOLO-DETR, YOLODETR, YOLO27-DETR, Ultralytics, object detection, DETR, RTDETRDecoderV2, DeimDecoder, DINOv3, DINOv3-ViT, Vision Transformer, YOLO26, NMS-free, real-time detection
---

# Ultralytics YOLO-DETR

## Overview

Ultralytics YOLO-DETR is a DETR-style object detection family that combines YOLO feature extractors and transformer
decoder heads inside the standard Ultralytics workflow. It uses the dedicated `YOLODETR` Python class while keeping
the familiar [Train](../modes/train.md), [Val](../modes/val.md), [Predict](../modes/predict.md), and
[Export](../modes/export.md) modes.

The family has two architecture groups:

- **YOLO26-style CSP variants (`n/s/m/l`)** use a YOLO26-style convolutional backbone and `RTDETRDecoderV2` head.
- **DINOv3-ViT + STA variants (`x/xxl`)** use a DINOv3 Vision Transformer backbone with spatial token aggregation
  (STA) and a `DeimDecoder` head. The `x` scale uses DINOv3-ViT-S/16-plus; the `xxl` scale uses DINOv3-ViT-B/16.

Both groups produce DETR-style query predictions, making YOLO-DETR suitable for NMS-free object detection pipelines
with fixed decoder queries and end-to-end export support.

## Key Features

- **DETR-style object queries:** YOLO-DETR predicts a fixed set of object queries, avoiding traditional dense anchor
  grids and non-maximum suppression in the standard inference path.
- **Ultralytics workflow integration:** Models work with the same training, validation, prediction, and export modes
  used across other Ultralytics model families.
- **Two backbone groups:** Compact and standard variants use YOLO26-style CSP backbones, while the larger variants use
  DINOv3-ViT + STA backbones for stronger feature extraction.
- **Scale-specific decoder depth:** Each variant selects its decoder-layer count from the model config, balancing
  latency and accuracy for the target deployment point.
- **Efficient nano decoder:** YOLO27n-DETR enables `efficient_ms=True`, scheduling single-level cross-attention across
  feature levels in round-robin order for a lighter multi-scale decoder path.
- **YOLODETR training recipe:** The trainer adds augmentation decay, a flat-cosine learning-rate schedule, and separate
  learning rates for backbone and head parameter groups.

## Model Variants

The main YOLO-DETR family currently contains six release-facing variants. The `n/s/m/l` models are scale-resolved from
`yolo27-detr.yaml`; the `x` and `xxl` models use separate DINOv3-ViT-backed configs.

| Model          | Config                | Backbone Source              | Decoder           | Decoder Layers | Notes                                                      |
| -------------- | --------------------- | ---------------------------- | ----------------- | -------------- | ---------------------------------------------------------- |
| YOLO27n-DETR   | `yolo27n-detr.yaml`   | YOLO26n-style CSP            | `RTDETRDecoderV2` | 3              | Uses `efficient_ms=True` round-robin multi-scale attention |
| YOLO27s-DETR   | `yolo27s-detr.yaml`   | YOLO26s-style CSP            | `RTDETRDecoderV2` | 3              | Standard multi-scale decoder                               |
| YOLO27m-DETR   | `yolo27m-detr.yaml`   | YOLO26l-style CSP            | `RTDETRDecoderV2` | 2              | Shorter decoder for a middle deployment point              |
| YOLO27l-DETR   | `yolo27l-detr.yaml`   | YOLO26l-style CSP            | `RTDETRDecoderV2` | 4              | Larger CSP variant with a deeper decoder                   |
| YOLO27x-DETR   | `yolo27x-detr.yaml`   | DINOv3-ViT-S/16-plus + STA   | `DeimDecoder`     | 6              | DINOv3-dependent high-capacity variant                     |
| YOLO27xxl-DETR | `yolo27xxl-detr.yaml` | DINOv3-ViT-B/16 + STA        | `DeimDecoder`     | 4              | DINOv3-dependent extra-large variant                       |

!!! note "Backbone labels"

    `YOLO27m-DETR` uses the YOLO26l-style CSP backbone with a shorter 2-layer decoder, so its `m` label describes the
    deployment point rather than a separate YOLO26m backbone. The `n/s/m/l` configs also resolve scale-specific
    `RTDETRDecoderV2` settings such as decoder depth and `efficient_ms` from `scale_args`.

## Supported Tasks and Modes

YOLO-DETR is currently an object detection family.

| Model Family | Task                                | Inference | Validation | Training | Export |
| ------------ | ----------------------------------- | --------- | ---------- | -------- | ------ |
| YOLO-DETR    | [Object Detection](../tasks/detect.md) | Yes       | Yes        | Yes      | Yes    |

## Performance Metrics

!!! tip "Performance"

    === "Detection (COCO)"

        | Model          | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> |
        | -------------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ |
        | YOLO27n-DETR   | 480                         | -                          | 49.8 ± 1.0                           | 1.7 ± 0.0                                 | 6.04                     |
        | YOLO27s-DETR   | 512                         | -                          | 114.2 ± 2.4                          | 2.7 ± 0.0                                 | 13.05                    |
        | YOLO27m-DETR   | 512                         | -                          | 224.7 ± 5.0                          | 4.6 ± 0.1                                 | 26.35                    |
        | YOLO27l-DETR   | 640                         | -                          | 357.1 ± 2.5                          | 7.2 ± 0.1                                 | 28.65                    |
        | YOLO27x-DETR   | 640                         | -                          | 965.3 ± 5.1                          | 13.8 ± 0.2                                | 50.37                    |
        | YOLO27xxl-DETR | 640                         | -                          | 1775.3 ± 12.2                        | 25.3 ± 0.8                                | 106.22                   |

_Latency measured on NVIDIA T4 with TensorRT v10.11 FP16 (batch=1) and CPU ONNX at each scale's recommended `imgsz`
(shown in the `size` column) to preserve the latency-performance balance. mAP values will be filled in once
final COCO benchmarks are complete._

## Usage Examples

This example shows simple YOLO-DETR training and inference. For complete mode-specific options, see the
[Train](../modes/train.md), [Val](../modes/val.md), [Predict](../modes/predict.md), and [Export](../modes/export.md)
documentation.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLODETR

        # Load a YOLO27n-DETR model from YAML
        model = YOLODETR("yolo27n-detr.yaml")

        # Train the model on the COCO8 example dataset (imgsz=480 is the recommended nano size)
        results = model.train(data="coco8.yaml", epochs=100, imgsz=480)

        # Run inference on an image
        results = model("path/to/image.jpg")
        ```

    === "CLI"

        ```bash
        # Train a YOLO27n-DETR model on the COCO8 example dataset (imgsz=480 is the recommended nano size)
        yolo train model=yolo27n-detr.yaml data=coco8.yaml epochs=100 imgsz=480

        # Run inference with a YOLO27n-DETR model
        yolo predict model=yolo27n-detr.yaml source=path/to/image.jpg
        ```

## Training Notes

YOLO-DETR accepts standard Ultralytics training arguments such as `data`, `epochs`, `imgsz`, `batch`, `optimizer`,
`lr0`, `lrf`, and augmentation probabilities. It also supports YOLO-DETR-specific training options through
`model.train(...)`:

| Argument            | Default | Description                                                                 |
| ------------------- | ------- | --------------------------------------------------------------------------- |
| `no_aug_epoch`      | `4`     | Number of trailing epochs where augmentation is disabled                    |
| `backbone_lr_ratio` | `0.02`  | Multiplier applied to backbone parameter-group learning rates               |
| `base_size_repeat`  | `3`     | Extra weight given to the base image size during multi-scale size sampling  |

These options are handled by `YOLODETRTrainer` and are not added to `default.yaml`. Pass them directly to
`model.train(...)` when you need to override the defaults.

### Backbone Learning Rate

`backbone_lr_ratio` multiplies the main learning rate for backbone parameter groups:

```text
backbone_lr = lr0 * backbone_lr_ratio
```

Use a smaller ratio when the backbone is a pretrained transformer that should adapt slowly, and a larger ratio when
fine-tuning YOLO26-style CSP backbones. The following examples show practical starting points:

| Model group                     | Example model       | `lr0`    | `backbone_lr_ratio` | Effective backbone LR | Notes                                      |
| ------------------------------- | ------------------- | -------- | ------------------- | --------------------- | ------------------------------------------ |
| DINOv3-ViT + STA backbone       | `yolo27x-detr.yaml` | `0.0005` | `0.02`              | `0.00001`             | Protects the pretrained ViT backbone       |
| YOLO26L-style CSP backbone      | `yolo27l-detr.yaml` | `0.0001` | `0.1`               | `0.00001`             | Fine-tunes the CSP backbone more directly  |

The effective backbone learning rate can be similar across recipes even when `lr0` and `backbone_lr_ratio` differ. For
example, the DINOv3-ViT recipe uses a higher head LR with a lower backbone ratio, while the YOLO26L-style CSP recipe
uses a lower head LR with a higher backbone ratio.

```python
from ultralytics import YOLODETR

model = YOLODETR("yolo27x-detr.yaml")
model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    no_aug_epoch=4,
    backbone_lr_ratio=0.02,
    base_size_repeat=3,
)
```

## Inference and Export Notes

YOLO-DETR models use a fixed number of decoder queries, 300 in the default configs. Increasing `max_det` does not
create additional decoder queries; it only changes how many predictions can be returned after inference. If a dataset
can contain more objects than the configured query count, adjust the query count in the model YAML and retrain so the
decoder learns the additional queries.

Decoder depth is part of the selected architecture. The YOLO26-style CSP variants use 3, 3, 2, and 4 decoder layers
for `n`, `s`, `m`, and `l`, respectively. The DINOv3-ViT + STA variants use 6 decoder layers for `x` and 4 for `xxl`.
Export keeps the selected architecture and decoder behavior, including `efficient_ms=True` for YOLO27n-DETR.

## FAQ

### How is YOLO-DETR different from other YOLO models?

Standard YOLO models use CNN-based detection heads that predict outputs on dense feature grids. YOLO-DETR replaces
that with a Transformer-based DETR-style decoder head, adopting the object-query design and the training and inference
practices established in the DETR literature, which yields NMS-free predictions with a fixed set of decoder queries
per image.




## License

YOLO27-DETR-X and YOLO27-DETR-XXL use DINOv3-derived weights and are Built with DINOv3. These variants are subject to
the [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/) in addition to the applicable
Ultralytics license terms. The `n/s/m/l` variants use YOLO26-style CSP backbones and are covered by the Ultralytics
license terms only.
