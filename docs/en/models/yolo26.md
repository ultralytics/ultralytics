---
comments: true
description: YOLO26 from Ultralytics delivers unified, real-time, end-to-end vision models optimized for accurate and efficient deployment.
keywords: YOLO26, Ultralytics YOLO, object detection, end-to-end NMS-free, YOLOE-26, open-vocabulary detection, computer vision, AI, machine learning, edge AI, real-time inference
---

# Ultralytics YOLO26

## Overview

[Ultralytics](https://www.ultralytics.com) YOLO26 is a unified family of real-time vision models described in the [Ultralytics YOLO26 paper](https://arxiv.org/abs/2606.03748). It introduces native end-to-end inference, a lighter detection head, an updated training recipe, and task-specific heads for detection, segmentation, pose estimation, classification, and oriented detection.

Across its five detection scales, YOLO26 reaches **40.9-57.5 mAP on COCO** at **1.7-11.8 ms T4 TensorRT latency**. The paper also reports **up to 43% faster CPU ONNX inference** for YOLO26n compared with YOLO11n on an Intel Xeon CPU @ 2.00 GHz.

See the [unreleased YOLO27 preview](yolo27.md) for planned architecture, tasks, and preliminary benchmarks.

![Ultralytics YOLO26 Comparison Plots](https://cdn.ul.run/i/1b042c1a3e984e8d2eb58c2af3c53965.avif)

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/7lZa3Yi2kbo"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train a YOLO26 model on Your Custom Dataset in <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb" target="_blank">Google Colab</a>.
</p>

!!! example "Quickstart"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")  # load a pretrained YOLO26n model
        results = model("path/to/bus.jpg")  # run inference
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo26n.pt source=path/to/bus.jpg
        ```

!!! tip "Try on Ultralytics Platform"

    Explore and run YOLO26 models directly on [Ultralytics Platform](https://platform.ultralytics.com/ultralytics/yolo26).

The YOLO26 model family is built around four design areas:

- **Native end-to-end inference:** The optional one-to-one detection head produces predictions without non-maximum suppression (NMS), simplifying deployment and reducing post-processing.
- **Lighter box regression:** YOLO26 removes Distribution Focal Loss (DFL), reducing detection-head complexity while preserving an unconstrained regression range.
- **Training recipe updates:** The training pipeline combines **MuSGD** (a hybrid Muon + SGD optimizer), **Progressive Loss**, and **STAL** (Small-Target-Aware Label Assignment) to improve optimization, shift supervision toward the inference-time head, and maintain positive label coverage for small objects. The full hyperparameters behind the released checkpoints are documented in the [YOLO26 Training Recipe guide](../guides/yolo26-training-recipe.md).
- **Task-specific heads and losses:** YOLO26 adds targeted designs for instance segmentation, semantic segmentation variants, pose estimation, and oriented detection while keeping a single model pipeline across tasks.

Together, these updates improve the accuracy-latency tradeoff across model scales and deployment targets.

## Key Features

- **DFL-Free Regression**
  YOLO26 removes Distribution Focal Loss (DFL), reducing detection-head complexity and simplifying export.

- **End-to-End NMS-Free Inference**
  YOLO26 supports **native end-to-end** inference with `nms=False`, producing predictions without a separate NMS pass. The default one-to-many path uses NMS for accuracy.

- **Progressive Loss + STAL**
  Progressive Loss shifts training emphasis toward the inference-time head, while STAL improves positive label coverage for small objects.

- **MuSGD Optimizer**
  A hybrid optimizer that combines [SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html) with [Muon](https://arxiv.org/abs/2502.16982), adapting optimization ideas from large language model training to computer vision.

- **Efficient Deployment**
  The simplified head and optional NMS-free path reduce inference overhead across export targets and hardware profiles, including the paper's reported CPU ONNX speedup for YOLO26n versus YOLO11n.

- **Instance Segmentation Enhancements**
  Introduces semantic segmentation loss to improve model convergence and an upgraded proto module that leverages multi-scale information for superior mask quality. The paper reports gains over YOLO11 of up to +2.5 box AP and +3.7 mask AP on COCO instance segmentation.

- **Precision Pose Estimation**
  Integrates [Residual Log-Likelihood Estimation](https://arxiv.org/abs/2107.11291) (RLE) for more accurate keypoint localization and optimizes the decoding process for increased inference speed. The paper reports up to +7.2 AP over YOLO11 on COCO pose estimation.

- **Refined OBB Decoding**
  Introduces a specialized angle loss to improve detection accuracy for square-shaped objects and optimizes OBB decoding to resolve boundary discontinuity issues. The paper reports up to +3.4 mAP over YOLO11 on DOTA-v1.0 oriented detection.

![Ultralytics YOLO26 End-to-End Comparison Plots](https://cdn.ul.run/i/93c237f74ef9032861f5c821943a2f7b.avif)

## Supported Tasks and Modes

YOLO26 supports the standard Ultralytics task set across five model scales:

| Model        | Filenames                                                                                      | Task                                          | Training | Validation | Inference | Export |
| ------------ | ---------------------------------------------------------------------------------------------- | --------------------------------------------- | -------- | ---------- | --------- | ------ |
| YOLO26       | `yolo26n.pt` `yolo26s.pt` `yolo26m.pt` `yolo26l.pt` `yolo26x.pt`                               | [Detection](../tasks/detect.md)               | ✅       | ✅         | ✅        | ✅     |
| YOLO26-seg   | `yolo26n-seg.pt` `yolo26s-seg.pt` `yolo26m-seg.pt` `yolo26l-seg.pt` `yolo26x-seg.pt`           | [Instance Segmentation](../tasks/segment.md)  | ✅       | ✅         | ✅        | ✅     |
| YOLO26-sem   | `yolo26n-sem.pt` `yolo26s-sem.pt` `yolo26m-sem.pt` `yolo26l-sem.pt` `yolo26x-sem.pt`           | [Semantic Segmentation](../tasks/semantic.md) | ✅       | ✅         | ✅        | ✅     |
| YOLO26-depth | `yolo26n-depth.pt` `yolo26s-depth.pt` `yolo26m-depth.pt` `yolo26l-depth.pt` `yolo26x-depth.pt` | [Depth Estimation](../tasks/depth.md)         | ✅       | ✅         | ✅        | ✅     |
| YOLO26-cls   | `yolo26n-cls.pt` `yolo26s-cls.pt` `yolo26m-cls.pt` `yolo26l-cls.pt` `yolo26x-cls.pt`           | [Classification](../tasks/classify.md)        | ✅       | ✅         | ✅        | ✅     |
| YOLO26-pose  | `yolo26n-pose.pt` `yolo26s-pose.pt` `yolo26m-pose.pt` `yolo26l-pose.pt` `yolo26x-pose.pt`      | [Pose/Keypoints](../tasks/pose.md)            | ✅       | ✅         | ✅        | ✅     |
| YOLO26-obb   | `yolo26n-obb.pt` `yolo26s-obb.pt` `yolo26m-obb.pt` `yolo26l-obb.pt` `yolo26x-obb.pt`           | [Oriented Detection](../tasks/obb.md)         | ✅       | ✅         | ✅        | ✅     |

This unified framework covers real-time detection, instance segmentation, semantic segmentation, monocular depth estimation, classification, pose estimation, and oriented object detection with training, validation, inference, and export support.

!!! note "Architecture-only variants"

    [`yolo26-p2.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26-p2.yaml) and [`yolo26-p6.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26-p6.yaml) add a P2 (small-object) or P6 (large-input) detection head and are shipped as YAML architectures only. No scale-specific `yolo26*-p2.pt` or `yolo26*-p6.pt` weights are released. Instantiate a scaled config from YAML (for example, `YOLO("yolo26n-p6.yaml")`) and train or fine-tune it as needed.

## Performance Metrics

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["YOLO26"]'></canvas>

!!! tip "Performance"

    === "Detection (COCO)"

        See [Detection Docs](../tasks/detect.md) for usage examples with these models trained on [COCO](../datasets/detect/coco.md), which include 80 pretrained classes.

        | Model                                                                  | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | mAP<sup>val<br>50-95(e2e)</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | ---------------------------------------------------------------------- | --------------------------- | -------------------------- | ------------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
        | [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n) | 640                         | 40.9                       | 40.1                            | 38.9 ± 0.7                           | 1.7 ± 0.0                                 | 2.4                      | 5.5                     |
        | [YOLO26s](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s) | 640                         | 48.6                       | 47.8                            | 87.2 ± 0.9                           | 2.5 ± 0.0                                 | 9.5                      | 20.9                    |
        | [YOLO26m](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m) | 640                         | 53.1                       | 52.5                            | 220.0 ± 1.4                          | 4.7 ± 0.1                                 | 20.4                     | 68.4                    |
        | [YOLO26l](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l) | 640                         | 55.0                       | 54.4                            | 286.2 ± 2.0                          | 6.2 ± 0.2                                 | 24.8                     | 86.8                    |
        | [YOLO26x](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x) | 640                         | 57.5                       | 56.9                            | 525.8 ± 4.0                          | 11.8 ± 0.2                                | 55.7                     | 194.4                   |

    === "Segmentation (COCO)"

        See [Segmentation Docs](../tasks/segment.md) for usage examples with these models trained on [COCO](../datasets/segment/coco.md), which include 80 pretrained classes.

        | Model                                                                          | size<br><sup>(pixels)</sup> | mAP<sup>box<br>50-95(e2e)</sup> | mAP<sup>mask<br>50-95(e2e)</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | ------------------------------------------------------------------------------ | --------------------------- | ------------------------------- | -------------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
        | [YOLO26n-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-seg) | 640                         | 39.6                            | 33.9                             | 53.3 ± 0.5                           | 2.1 ± 0.0                                 | 2.7                      | 9.3                     |
        | [YOLO26s-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s-seg) | 640                         | 47.3                            | 40.0                             | 118.4 ± 0.9                          | 3.3 ± 0.0                                 | 10.4                     | 34.5                    |
        | [YOLO26m-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m-seg) | 640                         | 52.5                            | 44.1                             | 328.2 ± 2.4                          | 6.7 ± 0.1                                 | 23.6                     | 121.7                   |
        | [YOLO26l-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l-seg) | 640                         | 54.4                            | 45.5                             | 387.0 ± 3.7                          | 8.0 ± 0.1                                 | 28.0                     | 140.1                   |
        | [YOLO26x-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x-seg) | 640                         | 56.5                            | 47.0                             | 787.0 ± 6.8                          | 16.4 ± 0.1                                | 62.8                     | 314.0                   |

    === "Semantic Segmentation (Cityscapes)"

        See [Semantic Segmentation Docs](../tasks/semantic.md) for usage examples with these models trained on [Cityscapes](../datasets/semantic/cityscapes.md), which include 19 pretrained classes.

        | Model                                                                          | size<br><sup>(pixels)</sup> | mIoU<sup>val</sup> | Speed<br><sup>RTX3090 PyTorch<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | ------------------------------------------------------------------------------ | --------------------------- | ------------------ | ------------------------------------------- | ------------------------ | ----------------------- |
        | [YOLO26n-sem](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-sem) | 1024 &times; 2048           | 78.3               | 4.4 ± 0.0                                   | 1.6                      | 23.8                    |
        | [YOLO26s-sem](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s-sem) | 1024 &times; 2048           | 80.8               | 8.4 ± 0.0                                   | 6.5                      | 91.0                    |
        | [YOLO26m-sem](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m-sem) | 1024 &times; 2048           | 82.0               | 19.9 ± 0.1                                  | 14.3                     | 305.5                   |
        | [YOLO26l-sem](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l-sem) | 1024 &times; 2048           | 82.9               | 26.5 ± 0.1                                  | 17.8                     | 388.2                   |
        | [YOLO26x-sem](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x-sem) | 1024 &times; 2048           | 83.6               | 48.9 ± 0.2                                  | 40.1                     | 866.9                   |

    === "Depth Estimation (NYU Depth V2)"

        See [Depth Estimation Docs](../tasks/depth.md) for usage examples with these models pretrained on a broad multi-dataset mix and evaluated on [NYU Depth V2](../datasets/depth/nyu-depth-v2.md).

        | Model                                                                              | size<br><sup>(pixels)</sup> | delta1<sup>NYU</sup> | abs_rel<sup>NYU</sup> | rmse<sup>NYU</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | ---------------------------------------------------------------------------------- | --------------------------- | -------------------- | --------------------- | ------------------ | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
        | [YOLO26n-depth](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-depth) | 768                         | 0.882                | 0.109                 | 0.414              | 272.0 ± 27.2                         | 2.7 ± 0.1                                 | 6.3                      | 46.9                    |
        | [YOLO26s-depth](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s-depth) | 768                         | 0.896                | 0.104                 | 0.399              | 393.7 ± 13.1                         | 3.8 ± 0.0                                 | 13.2                     | 68.0                    |
        | [YOLO26m-depth](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m-depth) | 768                         | 0.921                | 0.089                 | 0.364              | 621.5 ± 49.7                         | 6.0 ± 0.1                                 | 23.3                     | 130.4                   |
        | [YOLO26l-depth](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l-depth) | 768                         | 0.930                | 0.083                 | 0.351              | 821.9 ± 50.7                         | 7.7 ± 0.1                                 | 27.7                     | 157.0                   |
        | [YOLO26x-depth](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x-depth) | 768                         | 0.933                | 0.080                 | 0.344              | 1240.9 ± 73.3                        | 13.6 ± 0.2                                | 57.0                     | 301.7                   |

    === "Classification (ImageNet)"

        See [Classification Docs](../tasks/classify.md) for usage examples with these models trained on [ImageNet](../datasets/classify/imagenet.md), which include 1000 pretrained classes.

        | Model                                                                          | size<br><sup>(pixels)</sup> | acc<br><sup>top1</sup> | acc<br><sup>top5</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B) at 224</sup> |
        | ------------------------------------------------------------------------------ | --------------------------- | ---------------------- | ---------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ------------------------------ |
        | [YOLO26n-cls](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-cls) | 224                         | 71.4                   | 90.1                   | 5.0 ± 0.3                            | 1.1 ± 0.0                                 | 2.8                      | 0.4                            |
        | [YOLO26s-cls](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s-cls) | 224                         | 76.0                   | 92.9                   | 7.9 ± 0.2                            | 1.3 ± 0.0                                 | 6.7                      | 1.5                            |
        | [YOLO26m-cls](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m-cls) | 224                         | 78.1                   | 94.2                   | 17.2 ± 0.4                           | 2.0 ± 0.0                                 | 11.6                     | 4.8                            |
        | [YOLO26l-cls](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l-cls) | 224                         | 79.0                   | 94.6                   | 23.2 ± 0.3                           | 2.8 ± 0.0                                 | 14.1                     | 6.0                            |
        | [YOLO26x-cls](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x-cls) | 224                         | 79.9                   | 95.0                   | 41.4 ± 0.9                           | 3.8 ± 0.0                                 | 29.6                     | 13.5                           |

    === "Pose (COCO)"

        See [Pose Estimation Docs](../tasks/pose.md) for usage examples with these models trained on [COCO](../datasets/pose/coco.md), which include 1 pretrained class, 'person'.

        | Model                                                                            | size<br><sup>(pixels)</sup> | mAP<sup>pose<br>50-95(e2e)</sup> | mAP<sup>pose<br>50(e2e)</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | -------------------------------------------------------------------------------- | --------------------------- | -------------------------------- | ----------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
        | [YOLO26n-pose](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-pose) | 640                         | 57.2                             | 83.3                          | 40.3 ± 0.5                           | 1.8 ± 0.0                                 | 2.9                      | 7.6                     |
        | [YOLO26s-pose](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s-pose) | 640                         | 63.0                             | 86.6                          | 85.3 ± 0.9                           | 2.7 ± 0.0                                 | 10.4                     | 24.1                    |
        | [YOLO26m-pose](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m-pose) | 640                         | 68.8                             | 89.6                          | 218.0 ± 1.5                          | 5.0 ± 0.1                                 | 21.5                     | 73.3                    |
        | [YOLO26l-pose](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l-pose) | 640                         | 70.4                             | 90.5                          | 275.4 ± 2.4                          | 6.5 ± 0.1                                 | 25.9                     | 91.7                    |
        | [YOLO26x-pose](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x-pose) | 640                         | 71.6                             | 91.6                          | 565.4 ± 3.0                          | 12.2 ± 0.2                                | 57.6                     | 202.3                   |

    === "OBB (DOTAv1)"

        See [Oriented Detection Docs](../tasks/obb.md) for usage examples with these models trained on [DOTAv1](../datasets/obb/dota-v2.md#dota-v10), which include 15 pretrained classes.

        | Model                                                                          | size<br><sup>(pixels)</sup> | mAP<sup>test<br>50-95(e2e)</sup> | mAP<sup>test<br>50(e2e)</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>T4 TensorRT10<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        | ------------------------------------------------------------------------------ | --------------------------- | -------------------------------- | ----------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
        | [YOLO26n-obb](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-obb) | 1024                        | 52.4                             | 78.9                          | 97.7 ± 0.9                           | 2.8 ± 0.0                                 | 2.4                      | 14.8                    |
        | [YOLO26s-obb](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s-obb) | 1024                        | 54.8                             | 80.9                          | 218.0 ± 1.4                          | 4.9 ± 0.1                                 | 9.8                      | 56.7                    |
        | [YOLO26m-obb](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m-obb) | 1024                        | 55.3                             | 81.0                          | 579.2 ± 3.8                          | 10.2 ± 0.3                                | 21.2                     | 184.9                   |
        | [YOLO26l-obb](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l-obb) | 1024                        | 56.2                             | 81.6                          | 735.6 ± 3.1                          | 13.0 ± 0.2                                | 25.6                     | 232.4                   |
        | [YOLO26x-obb](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x-obb) | 1024                        | 56.7                             | 81.7                          | 1485.7 ± 11.5                        | 30.5 ± 0.9                                | 57.6                     | 520.1                   |

_Params and FLOPs values are for the fused model after Conv/BatchNorm folding and removal of the unused detection branch. Speed measurements select the NMS-free head with `nms=False`. Pretrained checkpoints retain the full training architecture and may show higher counts._

## Usage Examples

This section provides simple YOLO26 training and inference examples. For full documentation on these and other [modes](../modes/index.md), see the [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md), and [Export](../modes/export.md) docs pages.

Note that the example below is for YOLO26 [Detect](../tasks/detect.md) models for [object detection](https://www.ultralytics.com/glossary/object-detection). For additional supported tasks, see the [Segment](../tasks/segment.md), [Semantic Segmentation](../tasks/semantic.md), [Depth](../tasks/depth.md), [Classify](../tasks/classify.md), [Pose](../tasks/pose.md), and [OBB](../tasks/obb.md) docs.

!!! example

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) pretrained `*.pt` models as well as configuration `*.yaml` files can be passed to the `YOLO()` class to create a model instance in Python:

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLO26n model
        model = YOLO("yolo26n.pt")

        # Run inference with the YOLO26n model on the 'bus.jpg' image
        results = model("path/to/bus.jpg")

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        CLI commands are available to directly run the models:

        ```bash
        # Load a COCO-pretrained YOLO26n model and run inference on the 'bus.jpg' image
        yolo predict model=yolo26n.pt source=path/to/bus.jpg

        # Load a COCO-pretrained YOLO26n model and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolo26n.pt data=coco8.yaml epochs=100 imgsz=640
        ```

!!! note "Dual-Head Architecture"

    YOLO26 detection models use a **dual-head architecture** that provides flexibility for different deployment scenarios:

    - **One-to-Many Head (Default)**: Generates traditional YOLO outputs requiring NMS post-processing, outputting `(N, nc + 4, 8400)` where `nc` is the number of classes. This head typically achieves slightly higher accuracy at the cost of additional processing.
    - **One-to-One Head (`nms=False`)**: Produces end-to-end predictions without NMS, outputting `(N, 300, 6)` with a maximum of 300 detections per image. This head is optimized for fast inference and simplified deployment.

    You can switch between heads during export, prediction, or validation:

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Use one-to-many head (default, Ultralytics applies NMS)
        results = model.predict("image.jpg")  # inference
        metrics = model.val(data="coco.yaml")  # validation
        model.export(format="onnx")  # export

        # Opt into the NMS-free one-to-one head
        results = model.predict("image.jpg", nms=False)  # inference
        metrics = model.val(data="coco.yaml", nms=False)  # validation
        model.export(format="onnx", nms=False)  # export
        ```

    === "CLI"

        ```bash
        # Use one-to-many head (default, Ultralytics applies NMS)
        yolo predict model=yolo26n.pt source=image.jpg
        yolo val model=yolo26n.pt data=coco.yaml
        yolo export model=yolo26n.pt format=onnx

        # Opt into the NMS-free one-to-one head
        yolo predict model=yolo26n.pt source=image.jpg nms=False
        yolo val model=yolo26n.pt data=coco.yaml nms=False
        yolo export model=yolo26n.pt format=onnx nms=False
        ```

    Prediction and validation default to the one-to-many head for accuracy. Use `nms=False` for NMS-free inference, or `nms=True` to embed NMS in a supported export. Both heads are trained regardless of this choice. See the [End-to-End Detection guide](../guides/end2end-detection.md) for output formats and export compatibility.

## YOLOE-26: Open-Vocabulary Detection and Segmentation

YOLO26 also powers [YOLOE-26](yoloe.md), an open-vocabulary variant that detects and segments object categories from **text prompts**, **visual prompts**, or a **prompt-free mode** instead of a fixed class list learned at training time. YOLOE-26 keeps YOLO26's optional NMS-free end-to-end (e2e) head, so open-vocabulary inference stays fast enough for dynamic environments where target categories change over time. YOLOE-26x reaches **40.6 AP** on LVIS minival under text prompting, **38.5 AP** under visual prompting and **31.1 AP** prompt-free — the paper's Non-E2E figures, which its end-to-end head trails by 1.1, 2.3 and 1.2 AP.

See the **[YOLOE documentation](yoloe.md)** for per-scale performance tables, prompt-free variants, and full usage examples.

## Citations and Acknowledgments

For a complete technical description of the YOLO26 architecture, training recipe, task heads, and YOLOE-26 open-vocabulary extension, read [Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models](https://arxiv.org/abs/2606.03748). If you use YOLO26 in your research, please cite:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{jocher2026ultralyticsyolo26unifiedrealtime,
          title = {Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models},
          author = {Glenn Jocher and Jing Qiu and Mengyu Liu and Shuai Lyu and Fatih Cagatay Akyon and Muhammet Esat Kalfaoglu},
          year = {2026},
          eprint = {2606.03748},
          archivePrefix = {arXiv},
          primaryClass = {cs.CV},
          doi = {10.48550/arXiv.2606.03748},
          url = {https://arxiv.org/abs/2606.03748},
        }
        ```

YOLO26 code, models, and documentation are available in the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics) and [Ultralytics Docs](../index.md) under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) and [Enterprise](https://www.ultralytics.com/license) licenses.

## FAQ

### What are the key improvements in YOLO26?

- **DFL-free regression**: Simplifies the detection head and export path
- **End-to-end NMS-free inference**: Supports NMS-free inference with `nms=False`
- **Progressive Loss + STAL**: Improves training alignment and small-object label coverage
- **MuSGD optimizer**: Combines SGD with Muon-inspired optimization for stable training
- **Task-specific heads and losses**: Improves segmentation, pose, and oriented detection support

### What tasks does YOLO26 support?

YOLO26 is a **unified model family**, providing end-to-end support for multiple computer vision tasks:

- [Object Detection](../tasks/detect.md)
- [Instance Segmentation](../tasks/segment.md)
- [Semantic Segmentation](../tasks/semantic.md)
- [Monocular Depth Estimation](../tasks/depth.md)
- [Image Classification](../tasks/classify.md)
- [Pose Estimation](../tasks/pose.md)
- [Oriented Object Detection (OBB)](../tasks/obb.md)

Each size variant (n, s, m, l, x) supports all tasks, plus open-vocabulary versions via [YOLOE-26](yoloe.md).

### Why is YOLO26 efficient for deployment?

YOLO26 improves deployment efficiency with:

- Optional native end-to-end inference without NMS (`nms=False`)
- DFL-free regression and a lighter detection head
- Fused-model export that removes training-only auxiliary components
- Up to 43% faster CPU ONNX inference for YOLO26n versus YOLO11n on an Intel Xeon CPU @ 2.00 GHz
- Flexible export formats including TensorRT, ONNX, CoreML, LiteRT, and OpenVINO

### How do I get started with YOLO26?

YOLO26 models are available for download through the `ultralytics` package. Install or update the package and load a model:

```python
from ultralytics import YOLO

# Load a pretrained YOLO26 nano model
model = YOLO("yolo26n.pt")

# Run inference on an image
results = model("image.jpg")
```

See the [Usage Examples](#usage-examples) section for training, validation, and export instructions.
