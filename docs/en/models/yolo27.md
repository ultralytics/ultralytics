---
title: Ultralytics YOLO27 Real-Time Object Detection
comments: true
description: Preview Ultralytics YOLO27 - four sizes, seven tasks, and NMS-free detection. Models are undergoing final R&D and are not yet available. No launch date is set.
keywords: YOLO27, Ultralytics YOLO, object detection, NMS-free, end-to-end detection, small object detection, computer vision, AI, real-time inference
---

!!! info "YOLO27 is coming soon"

    YOLO27 models are undergoing final R&D, with a launch anticipated later this year. **The models are not yet available, and no launch date has been set.** This page previews the upcoming models; features and benchmarks may change before release. Code examples below are intended for use once the models and package support are released and will not work with the current public package. Model weights, configurations, and implementation code are not being released at this time.

# Ultralytics YOLO27

## Overview

[Ultralytics](https://www.ultralytics.com) YOLO27 is a family of real-time vision models whose detection models
use two complementary designs: a streamlined CNN architecture for the compact N and S models, and a query-based,
NMS-free architecture for the larger M and L models. Both designs are end-to-end and deploy through the same
interface.

YOLO27 is the upcoming model family in the Ultralytics YOLO series, succeeding [YOLO26](yolo26.md). It comes in four
sizes — N, S, M and L — and supports [object detection](../tasks/detect.md),
[instance segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md),
[depth estimation](../tasks/depth.md), [classification](../tasks/classify.md),
[pose estimation](../tasks/pose.md) and [oriented object detection](../tasks/obb.md). The two-design split applies
to detection; all other tasks use the CNN architecture.

Across its four detection scales, YOLO27 reaches **42.3-60.4 mAP on COCO** at **0.62-2.32 ms latency on an NVIDIA
RTX PRO 6000** — and up to **61.2 mAP** with YOLO27l at a larger 800-pixel input. YOLO27l is the **first Ultralytics
model to surpass 60 mAP on COCO**, while the compact YOLO27n/s improve on YOLO26n/s accuracy at essentially the same
speed.

!!! example "Quickstart (after release)"

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
  The larger models replace dense prediction with a [transformer](https://www.ultralytics.com/glossary/transformer)
  decoder that refines a fixed set of object queries
  and directly outputs the final detections — no
  [non-maximum suppression](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) post-processing needed. YOLO27m pairs this
  decoder with the proven YOLO26-style convolutional backbone, while YOLO27l keeps the same FPN/PAN neck and swaps
  in an UltraViT backbone that uses self-attention in its deepest stage to capture global context.

- **One simple interface**
  Both architectures are used through the same [`YOLO` Python interface](../usage/python.md). The right training, validation, prediction, and export
  pipeline is selected automatically from the model, so code written for one YOLO27 scale works unchanged for the
  others.

## Which YOLO27 Should I Use?

The following guidance is for choosing a model once YOLO27 is released. For projects today, use [YOLO26](yolo26.md).

- **YOLO27n / YOLO27s** — edge devices, drones, and real-time video: the fastest models in the family, with improved
  small-object detection from the dual-scale design. See [NVIDIA Jetson](../guides/nvidia-jetson.md) and
  [Raspberry Pi](../guides/raspberry-pi.md) for device-specific deployment.
- **YOLO27m** — the accuracy-speed sweet spot on GPUs: improves on YOLO26m by 3.2 mAP at the same latency,
  making it the default choice for production [GPU deployment](../guides/model-deployment-options.md).
- **YOLO27l** — accuracy-critical applications: the first Ultralytics model above 60 mAP on COCO, reaching 61.2 mAP
  at a larger input size while staying real-time on GPU.

To compare against other families, see [all Ultralytics models](index.md). To measure the accuracy-speed trade-off
on your own hardware rather than ours, see [Benchmark mode](../modes/benchmark.md).

---

## Supported Tasks and Modes

The following table previews planned task and mode support across the four YOLO27 model scales. Checkmarks indicate intended support at release, not current public availability; all listed model files are unreleased.

| Model        | Filenames                                                                   | Task                                          | Training | Validation | Inference | Export |
| ------------ | --------------------------------------------------------------------------- | --------------------------------------------- | -------- | ---------- | --------- | ------ |
| YOLO27       | `yolo27n.pt` `yolo27s.pt` `yolo27m.pt` `yolo27l.pt`                         | [Detection](../tasks/detect.md)               | ✅       | ✅         | ✅        | ✅     |
| YOLO27-seg   | `yolo27n-seg.pt` `yolo27s-seg.pt` `yolo27m-seg.pt` `yolo27l-seg.pt`         | [Instance Segmentation](../tasks/segment.md)  | ✅       | ✅         | ✅        | ✅     |
| YOLO27-sem   | `yolo27n-sem.pt` `yolo27s-sem.pt` `yolo27m-sem.pt` `yolo27l-sem.pt`         | [Semantic Segmentation](../tasks/semantic.md) | ✅       | ✅         | ✅        | ✅     |
| YOLO27-depth | `yolo27n-depth.pt` `yolo27s-depth.pt` `yolo27m-depth.pt` `yolo27l-depth.pt` | [Depth Estimation](../tasks/depth.md)         | ✅       | ✅         | ✅        | ✅     |
| YOLO27-cls   | `yolo27n-cls.pt` `yolo27s-cls.pt` `yolo27m-cls.pt` `yolo27l-cls.pt`         | [Classification](../tasks/classify.md)        | ✅       | ✅         | ✅        | ✅     |
| YOLO27-pose  | `yolo27n-pose.pt` `yolo27s-pose.pt` `yolo27m-pose.pt` `yolo27l-pose.pt`     | [Pose/Keypoints](../tasks/pose.md)            | ✅       | ✅         | ✅        | ✅     |
| YOLO27-obb   | `yolo27n-obb.pt` `yolo27s-obb.pt` `yolo27m-obb.pt` `yolo27l-obb.pt`         | [Oriented Detection](../tasks/obb.md)         | ✅       | ✅         | ✅        | ✅     |

YOLO27 Detect, Segment, Pose, and OBB models also work with [Track mode](../modes/track.md), which runs on top of
predict for multi-object tracking across video frames.

!!! note "Two architecture paths"

    YOLO27 detection uses two designs under one interface: the N and S scales use the streamlined CNN architecture,
    while the M and L scales use the query-based NMS-free architecture. All other tasks use the CNN architecture.

---

## Performance Metrics

Detection accuracy is reported on the COCO validation set. See [YOLO Performance Metrics](../guides/yolo-performance-metrics.md) for explanations of mAP, precision, and recall. Inference speed is measured on an NVIDIA RTX PRO 6000
([TensorRT](../integrations/tensorrt.md) 11, FP16) for GPU and an AMD EPYC 9655 ([ONNX Runtime](../integrations/onnx.md), FP32) for CPU. After release, accuracy numbers can be reproduced
with `yolo val model=yolo27n.pt data=coco.yaml`.

!!! note "Preliminary results"

    These research results may change during final R&D. Public weights and reproduction instructions will accompany the release.

!!! tip "Performance"

    === "Detection (COCO)"

        See [Detection Docs](../tasks/detect.md) for usage examples with these models trained on [COCO](../datasets/detect/coco.md), which include 80 pretrained classes. The table uses a 640-pixel input; the overview also reports YOLO27l at 800 pixels.

        | Model   | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>RTX PRO 6000 TensorRT11<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | ------- | --------------------------- | -------------------------- | ------------------------------------ | --------------------------------------------------- | ------------------------ | ----------------------- |
        | YOLO27n | 640                         | 42.3                       | **16.1 ± 1.6**                       | **0.62 ± 0.00**                                     | **3.0**                  | **7.2**                 |
        | YOLO27s | 640                         | 49.6                       | 33.2 ± 0.1                           | 0.79 ± 0.00                                         | 11.8                     | 28.2                    |
        | YOLO27m | 640                         | 55.8                       | 67.2 ± 0.3                           | 1.39 ± 0.00                                         | 22.8                     | 65.0                    |
        | YOLO27l | 640                         | **60.4**                   | 149.6 ± 1.4                          | 2.32 ± 0.00                                         | 72.3                     | 165.3                   |

    === "Segmentation (COCO)"

        See [Segmentation Docs](../tasks/segment.md) for usage examples with these models trained on [COCO](../datasets/segment/coco.md), which include 80 pretrained classes.

        | Model       | size<br><sup>(pixels)</sup> | mAP<sup>box<br>50-95</sup> | mAP<sup>mask<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>RTX PRO 6000 TensorRT11<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | ----------- | --------------------------- | -------------------------- | --------------------------- | ------------------------------------ | --------------------------------------------------- | ------------------------ | ----------------------- |
        | YOLO27n-seg | 640                         | 41.8                       | 35.4                        | **22.0 ± 2.2**                       | **0.73 ± 0.00**                                     | **3.0**                  | **11.2**                |
        | YOLO27s-seg | 640                         | 50.0                       | 42.5                        | 45.8 ± 0.1                           | 0.96 ± 0.00                                         | 11.6                     | 43.1                    |
        | YOLO27m-seg | 640                         | 53.2                       | 45.0                        | 109.3 ± 0.9                          | 1.46 ± 0.00                                         | 25.4                     | 139.6                   |
        | YOLO27l-seg | 640                         | **57.7**                   | **47.9**                    | 248.3 ± 1.3                          | 2.86 ± 0.00                                         | 67.5                     | 361.9                   |

    === "Semantic Segmentation (Cityscapes)"

        See [Semantic Segmentation Docs](../tasks/semantic.md) for usage examples with these models trained on [Cityscapes](../datasets/semantic/cityscapes.md), which include 19 pretrained classes.

        | Model       | size<br><sup>(pixels)</sup> | mIoU<sup>val</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>RTX PRO 6000 TensorRT11<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | ----------- | --------------------------- | ------------------ | ------------------------------------ | --------------------------------------------------- | ------------------------ | ----------------------- |
        | YOLO27n-sem | 1024 &times; 2048           | 78.8               | **95.6 ± 3.0**                       | **0.89 ± 0.00**                                     | **2.0**                  | **34.0**                |
        | YOLO27s-sem | 1024 &times; 2048           | 81.2               | 169.6 ± 0.9                          | 1.43 ± 0.00                                         | 7.8                      | 131.5                   |
        | YOLO27m-sem | 1024 &times; 2048           | 82.4               | 336.9 ± 2.8                          | 2.71 ± 0.00                                         | 16.1                     | 385.4                   |
        | YOLO27l-sem | 1024 &times; 2048           | **83.8**           | 787.1 ± 2.2                          | 6.43 ± 0.00                                         | 44.8                     | 1085.8                  |

    === "Depth Estimation (NYU Depth V2)"

        See [Depth Estimation Docs](../tasks/depth.md) for usage examples with these models pretrained on a broad multi-dataset mix and evaluated on [NYU Depth V2](../datasets/depth/nyu-depth-v2.md).

        | Model         | size<br><sup>(pixels)</sup> | delta1<sup>NYU</sup> | delta1<sup>KITTI-580</sup> | delta1<sup>bench mean</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>RTX PRO 6000 TensorRT11<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | ------------- | --------------------------- | -------------------- | -------------------------- | --------------------------- | ------------------------------------ | --------------------------------------------------- | ------------------------ | ----------------------- |
        | YOLO27n-depth | 768                         | 0.8314               | **0.8256**                 | 0.7238                      | **47.2 ± 4.0**                       | **0.79 ± 0.00**                                     | **5.4**                  | **49.3**                |
        | YOLO27s-depth | 768                         | 0.8682               | 0.7835                     | 0.7454                      | 72.1 ± 0.5                           | 0.97 ± 0.00                                         | 13.0                     | 77.3                    |
        | YOLO27m-depth | 768                         | 0.8652               | 0.7746                     | 0.7476                      | 120.2 ± 0.4                          | 1.34 ± 0.00                                         | 23.4                     | 143.6                   |
        | YOLO27l-depth | 768                         | **0.8711**           | 0.8041                     | **0.7527**                  | 253.7 ± 0.5                          | 2.65 ± 0.00                                         | 59.3                     | 341.8                   |

    === "Classification (ImageNet)"

        See [Classification Docs](../tasks/classify.md) for usage examples with these models trained on [ImageNet](../datasets/classify/imagenet.md), which include 1000 pretrained classes.

        | Model       | size<br><sup>(pixels)</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>RTX PRO 6000 TensorRT11<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B) at 224</sup> |
        | ----------- | --------------------------- | ------------------------------------ | --------------------------------------------------- | ------------------------ | ------------------------------ |
        | YOLO27n-cls | 224                         | **1.9 ± 0.0**                        | **0.28 ± 0.00**                                     | **2.9**                  | **0.5**                        |
        | YOLO27s-cls | 224                         | 3.1 ± 0.1                            | 0.31 ± 0.00                                         | 6.9                      | 1.7                            |
        | YOLO27m-cls | 224                         | 6.0 ± 0.6                            | 0.39 ± 0.00                                         | 12.4                     | 6.1                            |
        | YOLO27l-cls | 224                         | 16.7 ± 0.1                           | 0.69 ± 0.00                                         | 32.8                     | 18.5                           |

    === "Pose (COCO)"

        See [Pose Estimation Docs](../tasks/pose.md) for usage examples with these models trained on [COCO](../datasets/pose/coco.md), which include 1 pretrained class, 'person'.

        | Model        | size<br><sup>(pixels)</sup> | mAP<sup>pose<br>50-95</sup> | mAP<sup>pose<br>50</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>RTX PRO 6000 TensorRT11<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | ------------ | --------------------------- | --------------------------- | ------------------------ | ------------------------------------ | --------------------------------------------------- | ------------------------ | ----------------------- |
        | YOLO27n-pose | 640                         | 58.0                        | 84.1                     | **19.6 ± 0.1**                       | **0.67 ± 0.00**                                     | **3.1**                  | **9.2**                 |
        | YOLO27s-pose | 640                         | 64.3                        | 87.0                     | 36.6 ± 0.2                           | 0.80 ± 0.00                                         | 11.2                     | 31.1                    |
        | YOLO27m-pose | 640                         | 69.0                        | 89.9                     | 76.6 ± 0.2                           | 1.29 ± 0.00                                         | 22.8                     | 86.2                    |
        | YOLO27l-pose | 640                         | **72.2**                    | **91.3**                 | 181.1 ± 0.7                          | 2.45 ± 0.00                                         | 61.7                     | 242.3                   |

    === "OBB (DOTAv1)"

        See [Oriented Detection Docs](../tasks/obb.md) for usage examples with these models trained on [DOTAv1](../datasets/obb/dota-v2.md#dota-v10), which include 15 pretrained classes.

        | Model       | size<br><sup>(pixels)</sup> | mAP<sup>test<br>50-95</sup> | mAP<sup>test<br>50</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>RTX PRO 6000 TensorRT11<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | ----------- | --------------------------- | --------------------------- | ------------------------ | ------------------------------------ | --------------------------------------------------- | ------------------------ | ----------------------- |
        | YOLO27n-obb | 1024                        | 54.0                        | 80.3                     | **42.6 ± 0.2**                       | **0.83 ± 0.00**                                     | **3.0**                  | **19.7**                |
        | YOLO27s-obb | 1024                        | 55.8                        | 81.6                     | 88.2 ± 0.6                           | 1.30 ± 0.00                                         | 10.9                     | 76.6                    |
        | YOLO27m-obb | 1024                        | 55.9                        | 82.6                     | 188.0 ± 0.7                          | 1.92 ± 0.00                                         | 22.8                     | 221.7                   |
        | YOLO27l-obb | 1024                        | **57.2**                    | **82.7**                 | 455.8 ± 1.1                          | 4.33 ± 0.00                                         | 68.7                     | 622.6                   |

_Params and FLOPs values are for the fused model after Conv/BatchNorm folding and removal of the unused detection branch. Speed measurements select the NMS-free head with `nms=False`. Pretrained checkpoints retain the full training architecture and may show higher counts._

---

## Usage Examples

!!! info "For use after release"

    These examples show the intended YOLO27 API once model weights and package support are released. They cannot be run with the current public `ultralytics` package.

This section provides simple YOLO27 training and inference examples. For full documentation on these and other
[modes](../modes/index.md), see the [Predict](../modes/predict.md), [Train](../modes/train.md),
[Val](../modes/val.md), and [Export](../modes/export.md) docs pages. For custom datasets, see [Tips for Best Training Results](../guides/model-training-tips.md) and the [Hyperparameter Tuning Guide](../guides/hyperparameter-tuning.md).

Note that the example below is for YOLO27 [Detect](../tasks/detect.md) models for [object
detection](https://www.ultralytics.com/glossary/object-detection). For additional supported tasks, see the
[Segment](../tasks/segment.md) and [Classify](../tasks/classify.md) docs.

!!! example "Training and inference (after release)"

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

        After release, use [CLI commands](../usage/cli.md) to directly run the models:

        ```bash
        # Load a COCO-pretrained YOLO27n model and run inference on the 'bus.jpg' image
        yolo predict model=yolo27n.pt source=path/to/bus.jpg

        # Load a COCO-pretrained YOLO27n model and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolo27n.pt data=coco8.yaml epochs=100 imgsz=640
        ```

Only this documentation preview is available now. YOLO27 model weights, configurations, and implementation code remain unreleased. Release instructions will be added to this page when the models become available.

---

## FAQ

### What are the key improvements in YOLO27?

- **Dual-scale detection (N/S)**: drops the medium prediction map for a faster head with competitive accuracy
- **Stronger small-object detection (N/S)**: a widened early feature stage improves small-object localization
- **Foreground alignment supervision (N/S)**: cuts the one-to-many vs one-to-one accuracy gap from 0.9/0.8 mAP on
  YOLO26n/s to 0.4 mAP on YOLO27n/s, at zero inference cost
- **Query-based NMS-free detection (M/L)**: a transformer decoder outputs final detections directly
- **One simple interface**: both architectures run through the same `YOLO` class

### Should I upgrade from YOLO26?

Not yet — YOLO27 is unreleased. Continue using [YOLO26](yolo26.md) for current projects. Preliminary results show improved accuracy across model sizes, with YOLO27l surpassing 60 mAP on COCO. Evaluate the final released models for your workload before migrating.

### Is YOLO27 a drop-in replacement for YOLO26?

After release, YOLO27 is intended to use the same `YOLO` class and the same train/val/predict/export API as YOLO26 — the correct
pipeline (CNN or query-based) is selected automatically from the model. Swapping `yolo26n.pt` for `yolo27n.pt` is
the intended model-selection change once YOLO27 support and weights are available.

### Why do YOLO27 N and S predict on only two scales?

Most detectors predict on three feature maps at different resolutions. YOLO27 N and S keep the fine map that small
objects depend on and the coarse map that large objects need, and skip the medium one. This cuts a significant share
of detection-head computation, and the training improvements above keep the accuracy-latency tradeoff competitive.

### What accuracy and inference speed does YOLO27l achieve on COCO?

YOLO27l is the first Ultralytics model to surpass 60 mAP on COCO, reaching 60.4 mAP at a 640-pixel input (2.3 ms on
an NVIDIA RTX PRO 6000) and 61.2 mAP at an 800-pixel input (2.9 ms). It combines the UltraViT backbone, multi-scale
feature fusion, and a query-based detector that produces final detections directly, without NMS.

### When will YOLO27 be available?

YOLO27 is undergoing final R&D, with a launch anticipated later this year. No launch date has been set. Model weights and implementation code are not publicly available yet.

### How do I get started with YOLO27?

Once YOLO27 is released, see [Quickstart](../quickstart.md) to install or update the `ultralytics` package, then load a model as shown below. This example is for use after release:

```python
from ultralytics import YOLO

# Load a pretrained YOLO27 nano model
model = YOLO("yolo27n.pt")

# Run inference on an image
results = model("image.jpg")
```

See the [Usage Examples](#usage-examples) section for training, validation, and export instructions.
