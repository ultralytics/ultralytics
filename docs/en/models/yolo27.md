---
comments: true
description: YOLO27 from Ultralytics pairs a streamlined dual-scale CNN design for compact models with query-based NMS-free detection for large models, and is the first Ultralytics family to surpass 60 mAP on COCO.
keywords: YOLO27, Ultralytics YOLO, object detection, NMS-free, end-to-end detection, small object detection, computer vision, AI, real-time inference
---

# Ultralytics YOLO27

## Overview

[Ultralytics](https://www.ultralytics.com) YOLO27 is a family of real-time vision models built around two
complementary designs: a streamlined CNN architecture for the compact N and S models, and a query-based, NMS-free
architecture for the larger M, L, and X models. Both designs are end-to-end and deploy through the same interface.

Across its five detection scales, YOLO27 reaches **41.6-60.5 mAP on COCO** at **1.8-12.1 ms latency on an NVIDIA
T4** — and up to **61.0 mAP** with YOLO27x at a larger 800-pixel input. YOLO27x is the **first Ultralytics model to
surpass 60 mAP on COCO**, while the compact YOLO27n/s improve on YOLO26n/s accuracy at essentially the same speed.

### YOLO27 vs YOLO26

YOLO26 is compared using its end-to-end (one-to-one head) numbers, matching YOLO27's NMS-free evaluation.

| Scale | YOLO26 mAP<sup>val<br>50-95 (e2e)</sup> | YOLO27 mAP<sup>val<br>50-95</sup> | Δ mAP    | YOLO26 T4 (ms) | YOLO27 T4 (ms) |
| ----- | --------------------------------------- | --------------------------------- | -------- | -------------- | -------------- |
| n     | 40.1                                    | 41.6                              | +1.5     | 1.7            | 1.8            |
| s     | 47.8                                    | 49.2                              | +1.4     | 2.5            | 2.7            |
| m     | 52.5                                    | 55.7                              | **+3.2** | 4.7            | **4.6**        |
| l     | 54.4                                    | 57.7                              | +3.3     | 6.2            | 6.5            |
| x     | 56.9                                    | 60.5                              | **+3.6** | 11.8           | 12.1           |

<sup>YOLO27m is evaluated at a 512-pixel input; YOLO26m and all other scales use 640 pixels.</sup>

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

- **Dual-scale detection**
  Standard detectors predict objects on three feature maps — fine, medium, and coarse. YOLO27 N and S drop the
  medium one and predict only on a fine map (for small objects) and a coarse map (for large objects), with a fixed
  scaling on the fused features keeping the two scales balanced. This removes a large chunk of detection-head
  computation, making the models faster while training as reliably as the full three-scale design.

- **Stronger small-object detection**
  The early, high-resolution feature stage is widened so it can capture more fine-grained detail. Combined with the
  surviving fine prediction map, this improves localization and regression for small objects — the hardest category
  for compact models.

- **Foreground alignment supervision**
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

## Which YOLO27 Should I Use?

- **YOLO27n / YOLO27s** — edge devices, drones, and real-time video: the fastest models in the family, with improved
  small-object detection from the dual-scale design.
- **YOLO27m / YOLO27l** — the accuracy-speed sweet spot on GPUs: YOLO27m alone improves on YOLO26m by 3.2 mAP while
  running faster, making it the default choice for production GPU deployment.
- **YOLO27x** — accuracy-critical applications: the first Ultralytics model above 60 mAP on COCO, reaching 61.0 mAP
  at a larger input size while staying real-time on GPU.

---

## Supported Tasks and Modes

YOLO27 supports the following tasks across its five model scales. Detection, instance segmentation, and
classification are available today with training, validation, inference, and export support; the remaining tasks are
training now and will be released as they finalize:

| Model        | Filenames                                                                                      | Task                                          | Training | Validation | Inference | Export |
| ------------ | ---------------------------------------------------------------------------------------------- | --------------------------------------------- | -------- | ---------- | --------- | ------ |
| YOLO27       | `yolo27n.pt` `yolo27s.pt` `yolo27m.pt` `yolo27l.pt` `yolo27x.pt`                               | [Detection](../tasks/detect.md)               | ✅       | ✅         | ✅        | ✅     |
| YOLO27-seg   | `yolo27n-seg.pt` `yolo27s-seg.pt` `yolo27m-seg.pt` `yolo27l-seg.pt` `yolo27x-seg.pt`           | [Instance Segmentation](../tasks/segment.md)  | ✅       | ✅         | ✅        | ✅     |
| YOLO27-sem   | `yolo27n-sem.pt` `yolo27s-sem.pt` `yolo27m-sem.pt` `yolo27l-sem.pt` `yolo27x-sem.pt`           | [Semantic Segmentation](../tasks/semantic.md) | 🚧       | 🚧         | 🚧        | 🚧     |
| YOLO27-depth | `yolo27n-depth.pt` `yolo27s-depth.pt` `yolo27m-depth.pt` `yolo27l-depth.pt` `yolo27x-depth.pt` | [Depth Estimation](../tasks/depth.md)         | 🚧       | 🚧         | 🚧        | 🚧     |
| YOLO27-cls   | `yolo27n-cls.pt` `yolo27s-cls.pt` `yolo27m-cls.pt` `yolo27l-cls.pt` `yolo27x-cls.pt`           | [Classification](../tasks/classify.md)        | ✅       | ✅         | ✅        | ✅     |
| YOLO27-pose  | `yolo27n-pose.pt` `yolo27s-pose.pt` `yolo27m-pose.pt` `yolo27l-pose.pt` `yolo27x-pose.pt`      | [Pose/Keypoints](../tasks/pose.md)            | 🚧       | 🚧         | 🚧        | 🚧     |
| YOLO27-obb   | `yolo27n-obb.pt` `yolo27s-obb.pt` `yolo27m-obb.pt` `yolo27l-obb.pt` `yolo27x-obb.pt`           | [Oriented Detection](../tasks/obb.md)         | 🚧       | 🚧         | 🚧        | 🚧     |

🚧 Models are currently training and will be released as they finalize.

!!! note "Two architecture paths"

    YOLO27 detection uses two designs under one interface: the N and S scales use the streamlined CNN architecture,
    while the M, L, and X scales use the query-based NMS-free architecture. All other tasks use the CNN architecture.

---

## Performance Metrics

Detection accuracy is reported on the COCO validation set with latency on an NVIDIA T4 (TensorRT); the segmentation
and classification tables list model size and inference speed on CPU (ONNX) and GPU (TensorRT). Accuracy numbers can
be reproduced with `yolo val model=yolo27n.pt data=coco.yaml`.

=== "Detection (COCO)"

    YOLO27m is evaluated at a 512-pixel input; all other detection scales use 640 pixels. YOLO27x is additionally
    reported at an 800-pixel input.

    | Model   | Size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | CPU ONNX<br><sup>(ms)</sup> | T4 TensorRT<br><sup>(ms)</sup> | Params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
    | ------- | --------------------------- | -------------------------- | --------------------------- | ------------------------------ | ------------------------ | ----------------------- |
    | YOLO27n | 640                         | 41.6                       | —                           | 1.8                            | 3.0                      | 7.2                     |
    | YOLO27s | 640                         | 49.2                       | —                           | 2.7                            | 11.8                     | 28.2                    |
    | YOLO27m | 512                         | 55.7                       | 161.5 ± 7.0                 | 4.6                            | 27.2                     | 83.4                    |
    | YOLO27l | 640                         | 57.7                       | 258.7 ± 11.6                | 6.5                            | 30.4                     | 85.4                    |
    | YOLO27x | 640                         | 60.5                       | 414.4 ± 19.5                | 12.1                           | 65.6                     | 173.1                   |
    | YOLO27x | 800                         | **61.0**                   | 622.5 ± 30.2                | 17.1                           | 65.6                     | 266.8                   |

=== "Segmentation (COCO)"

    Measured at a 640-pixel input.

    | Model       | Size<br><sup>(pixels)</sup> | CPU ONNX<br><sup>(ms)</sup> | TensorRT<br><sup>(ms)</sup> | Params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
    | ----------- | --------------------------- | --------------------------- | --------------------------- | ------------------------ | ----------------------- |
    | YOLO27n-seg | 640                         | 63.1 ± 3.1                  | **2.092 ± 0.051**           | 3.0                      | 11.1                    |
    | YOLO27s-seg | 640                         | 119.0 ± 5.0                 | **3.562 ± 0.046**           | 11.6                     | 42.9                    |
    | YOLO27m-seg | 640                         | 280.7 ± 15.1                | **7.409 ± 0.193**           | 25.4                     | 139.4                   |
    | YOLO27l-seg | 640                         | 343.0 ± 7.7                 | **9.198 ± 0.194**           | 30.0                     | 161.0                   |
    | YOLO27x-seg | 640                         | 672.7 ± 54.8                | **18.123 ± 0.545**          | 67.5                     | 361.3                   |

=== "Semantic Segmentation (Cityscapes)"

    YOLO27 semantic segmentation models are currently training — results will be added once the models are finalized.

=== "Depth Estimation (NYU Depth V2)"

    YOLO27 depth estimation models are currently training — results will be added once the models are finalized.

=== "Classification (ImageNet)"

    Measured at a 224-pixel input.

    | Model       | Size<br><sup>(pixels)</sup> | CPU ONNX<br><sup>(ms)</sup> | TensorRT<br><sup>(ms)</sup> | Params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
    | ----------- | --------------------------- | --------------------------- | --------------------------- | ------------------------ | ----------------------- |
    | YOLO27n-cls | 224                         | 4.2 ± 0.6                   | **1.131 ± 0.046**           | 2.9                      | 0.6                     |
    | YOLO27s-cls | 224                         | 7.3 ± 0.7                   | **1.412 ± 0.025**           | 6.9                      | 1.9                     |
    | YOLO27m-cls | 224                         | 16.8 ± 1.1                  | **1.905 ± 0.053**           | 12.4                     | 6.2                     |
    | YOLO27l-cls | 224                         | 24.8 ± 1.6                  | **1.935 ± 0.075**           | 15.5                     | 8.4                     |
    | YOLO27x-cls | 224                         | 44.1 ± 3.4                  | **1.951 ± 0.038**           | 32.8                     | 18.6                    |

=== "Pose (COCO)"

    YOLO27 pose estimation models are currently training — results will be added once the models are finalized.

=== "OBB (DOTAv1)"

    YOLO27 oriented detection models are currently training — results will be added once the models are finalized.

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
[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) and
[Enterprise](https://www.ultralytics.com/license) licenses.

---

## FAQ

### What are the key improvements in YOLO27?

- **Dual-scale detection (N/S)**: drops the medium prediction map for a faster head with competitive accuracy
- **Stronger small-object detection (N/S)**: a widened early feature stage improves small-object localization
- **Foreground alignment supervision (N/S)**: cuts the one-to-many vs one-to-one accuracy gap from 0.9/0.8 mAP on
  YOLO26n/s to 0.4 mAP on YOLO27n/s, at zero inference cost
- **Query-based NMS-free detection (M/L/X)**: a transformer decoder outputs final detections directly
- **One simple interface**: both architectures run through the same `YOLO` class

### Should I upgrade from YOLO26?

Yes, for most use cases. YOLO27 improves end-to-end accuracy at every scale: +1.5/+1.4 mAP for the compact n/s
models at essentially the same speed, and +3.2/+3.3/+3.6 mAP for m/l/x. YOLO27x is the first Ultralytics model to
surpass 60 mAP on COCO. Note that YOLO27m is evaluated at a 512-pixel input (versus 640 for YOLO26m), which is part
of its speed advantage.

### Is YOLO27 a drop-in replacement for YOLO26?

Yes. All YOLO27 models use the same `YOLO` class and the same train/val/predict/export API as YOLO26 — the correct
pipeline (CNN or query-based) is selected automatically from the model. Swapping `yolo26n.pt` for `yolo27n.pt` is
the only change required.

### Why do YOLO27 N and S predict on only two scales?

Most detectors predict on three feature maps at different resolutions. YOLO27 N and S keep the fine map that small
objects depend on and the coarse map that large objects need, and skip the medium one. This cuts a significant share
of detection-head computation, and the training improvements above keep the accuracy-latency tradeoff competitive.

### What makes the YOLO27x result notable?

YOLO27x is the first Ultralytics model to surpass 60 mAP on COCO, reaching 60.5 mAP at a 640-pixel input (12.1 ms on
an NVIDIA T4) and 61.0 mAP at an 800-pixel input (17.1 ms). It combines the UltraViT backbone, multi-scale feature
fusion, and a query-based detector that produces final detections directly, without NMS.

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
