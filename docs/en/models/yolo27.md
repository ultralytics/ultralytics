---
comments: true
description: YOLO27 from Ultralytics pairs a streamlined two-scale CNN design for compact models with query-based NMS-free detection for large models, delivering state-of-the-art real-time object detection.
keywords: YOLO27, Ultralytics YOLO, object detection, NMS-free, end-to-end detection, small object detection, computer vision, AI, real-time inference
---

# Ultralytics YOLO27

## Overview

[Ultralytics](https://www.ultralytics.com) YOLO27 is a family of real-time vision models built around two
complementary designs: a streamlined CNN architecture for the compact N and S models, and a query-based, NMS-free
architecture for the larger M, L, and X models. Both designs are end-to-end and deploy through the same interface.

Across its five detection scales, YOLO27 reaches **41.6-60.5 mAP on COCO** at **1.8-12.3 ms latency on an NVIDIA
T4**. The compact models are built for speed, while the larger models trade compute for accuracy — YOLO27x sets the
family record at **60.5 mAP**.

!!! example "Quickstart"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo27n.pt")  # load a pretrained YOLO27n model
        results = model("path/to/bus.jpg")  # run inference
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo27n.pt source=path/to/bus.jpg
        ```

## Key Features

- **Streamlined two-scale detection**
  Standard detectors predict objects on three feature maps — fine, medium, and coarse. YOLO27 N and S drop the
  medium one and predict only on a fine map (for small objects) and a coarse map (for large objects), with a fixed
  scaling on the fused features keeping the two scales balanced. This removes a large chunk of detection-head
  computation, making the models faster while training as reliably as the full three-scale design.

- **Stronger small-object detection**
  The early, high-resolution feature stage is widened so it can capture more fine-grained detail. Combined with the
  surviving fine prediction map, this improves localization and regression for small objects — the hardest category
  for compact models.

- **Auxiliary foreground supervision**
  During training, an extra lightweight branch learns to tell "object" from "background" at every location. It is
  designed to close the gap between the denser one-to-many supervision used during training and the one-to-one head
  that produces the final predictions — cutting that accuracy gap from 0.9/0.8 mAP on YOLO26n/s to just 0.4 mAP on
  YOLO27n/s, so the deployed one-to-one head keeps nearly all of the training-time accuracy. The branch is used only
  during training and is removed for inference and export, so it costs nothing at deployment.

- **Query-based detection without NMS**
  The larger models replace dense prediction with a transformer decoder that refines a fixed set of object queries
  and directly outputs the final detections — no non-maximum suppression post-processing needed. YOLO27m and YOLO27l
  pair this decoder with the proven YOLO26-style convolutional backbone, while YOLO27x adds an UltraViT backbone that
  uses self-attention in its deepest stage to capture global context, plus a hybrid encoder that fuses features
  across scales.

- **One simple interface**
  Both architectures are used through the same `YOLO` class. The right training, validation, prediction, and export
  pipeline is selected automatically from the model, so code written for one YOLO27 scale works unchanged for the
  others.

---

## Supported Tasks and Modes

YOLO27 supports the following tasks across its five model scales, with training, validation, inference, and export
support:

| Model      | Filenames                                                                            | Task                                         | Training | Validation | Inference | Export |
| ---------- | ------------------------------------------------------------------------------------ | -------------------------------------------- | -------- | ---------- | --------- | ------ |
| YOLO27     | `yolo27n.pt` `yolo27s.pt` `yolo27m.pt` `yolo27l.pt` `yolo27x.pt`                     | [Detection](../tasks/detect.md)              | ✅       | ✅         | ✅        | ✅     |
| YOLO27-seg | `yolo27n-seg.pt` `yolo27s-seg.pt` `yolo27m-seg.pt` `yolo27l-seg.pt` `yolo27x-seg.pt` | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLO27-cls | `yolo27n-cls.pt` `yolo27s-cls.pt` `yolo27m-cls.pt` `yolo27l-cls.pt` `yolo27x-cls.pt` | [Classification](../tasks/classify.md)       | ✅       | ✅         | ✅        | ✅     |

!!! note "Two architecture paths"

    YOLO27 detection uses two designs under one interface: the N and S scales use the streamlined CNN architecture,
    while the M, L, and X scales use the query-based NMS-free architecture. Segmentation and classification models
    all use the CNN architecture. Support for additional tasks is under development.

---

## Performance Metrics

Detection accuracy is reported on the COCO validation set with latency on an NVIDIA T4; the segmentation and
classification tables list model size and inference speed on CPU (ONNX) and GPU (TensorRT).

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

---

## Usage Examples

This section provides simple YOLO27 training and inference examples. For full documentation on these and other
[modes](../modes/index.md), see the [Predict](../modes/predict.md), [Train](../modes/train.md),
[Val](../modes/val.md), and [Export](../modes/export.md) docs pages.

Note that the example below is for YOLO27 [Detect](../tasks/detect.md) models for [object
detection](https://www.ultralytics.com/glossary/object-detection). For additional supported tasks, see the
[Segment](../tasks/segment.md) and [Classify](../tasks/classify.md) docs.

!!! example

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) pretrained `*.pt` models as well as configuration
        `*.yaml` files can be passed to the `YOLO()` class to create a model instance in Python:

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLO27n model
        model = YOLO("yolo27n.pt")

        # Run inference with the YOLO27n model on the 'bus.jpg' image
        results = model("path/to/bus.jpg")

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        CLI commands are available to directly run the models:

        ```bash
        # Load a COCO-pretrained YOLO27n model and run inference on the 'bus.jpg' image
        yolo predict model=yolo27n.pt source=path/to/bus.jpg

        # Load a COCO-pretrained YOLO27n model and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolo27n.pt data=coco8.yaml epochs=100 imgsz=640
        ```

YOLO27 code, models, and documentation are available in the [Ultralytics GitHub
repository](https://github.com/ultralytics/ultralytics) and [Ultralytics Docs](../index.md) under
[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) and [Enterprise](https://www.ultralytics.com/license)
licenses.

---

## FAQ

### What are the key improvements in YOLO27?

- **Streamlined two-scale detection (N/S)**: drops the medium prediction map for a faster head with competitive accuracy
- **Stronger small-object detection (N/S)**: a widened early feature stage improves small-object localization
- **Auxiliary foreground supervision (N/S)**: cuts the one-to-many vs one-to-one accuracy gap from 0.9/0.8 mAP on
  YOLO26n/s to 0.4 mAP on YOLO27n/s, at zero inference cost
- **Query-based NMS-free detection (M/L/X)**: a transformer decoder outputs final detections directly
- **One simple interface**: both architectures run through the same `YOLO` class

### What tasks does YOLO27 support?

YOLO27 supports three tasks across its five scales (n, s, m, l, x):

- [Object Detection](../tasks/detect.md)
- [Instance Segmentation](../tasks/segment.md)
- [Image Classification](../tasks/classify.md)

Support for additional tasks is under development.

### Why do YOLO27 N and S predict on only two scales?

Most detectors predict on three feature maps at different resolutions. YOLO27 N and S keep the fine map that small
objects depend on and the coarse map that large objects need, and skip the medium one. This cuts a significant share
of detection-head computation, and the training improvements above keep the accuracy-latency tradeoff competitive.

### What makes the YOLO27x result notable?

YOLO27x is the family accuracy leader, reaching 60.5 mAP on COCO validation at 12.3 ms T4 latency. It combines the
UltraViT backbone, multi-scale feature fusion, and a query-based detector that produces final detections directly,
without NMS.

### How do I get started with YOLO27?

YOLO27 models are available through the `ultralytics` package. Install or update the package and load a model:

```python
from ultralytics import YOLO

# Load a pretrained YOLO27 nano model
model = YOLO("yolo27n.pt")

# Run inference on an image
results = model("image.jpg")
```

See the [Usage Examples](#usage-examples) section for training, validation, and export instructions.
