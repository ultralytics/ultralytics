---
comments: true
description: YOLO26 from Ultralytics delivers unified, real-time, end-to-end vision models optimized for accurate and efficient deployment.
keywords: YOLO26, Ultralytics YOLO, object detection, end-to-end NMS-free, YOLOE-26, open-vocabulary detection, computer vision, AI, machine learning, edge AI, real-time inference
---

# Ultralytics YOLO26

## Overview

[Ultralytics](https://www.ultralytics.com/) YOLO26 is a unified family of real-time vision models described in the [Ultralytics YOLO26 paper](https://arxiv.org/abs/2606.03748). It introduces native end-to-end inference, a lighter detection head, an updated training recipe, and task-specific heads for detection, segmentation, pose estimation, classification, and oriented detection.

Across its five detection scales, YOLO26 reaches **40.9-57.5 mAP on COCO** at **1.7-11.8 ms T4 TensorRT latency**. The paper also reports **up to 43% faster CPU ONNX inference** for YOLO26n compared with YOLO11n on an Intel Xeon CPU @ 2.00 GHz.

![Ultralytics YOLO26 Comparison Plots](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/Ultralytics-YOLO26-Benchmark.jpg)

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

- **Native end-to-end inference:** The default one-to-one detection head produces predictions without non-maximum suppression (NMS), simplifying deployment and reducing post-processing.
- **Lighter box regression:** YOLO26 removes Distribution Focal Loss (DFL), reducing detection-head complexity while preserving an unconstrained regression range.
- **Training recipe updates:** The training pipeline combines **MuSGD** (a hybrid Muon + SGD optimizer), **Progressive Loss**, and **STAL** (Small-Target-Aware Label Assignment) to improve optimization, shift supervision toward the inference-time head, and maintain positive label coverage for small objects. The full hyperparameters behind the released checkpoints are documented in the [YOLO26 Training Recipe guide](../guides/yolo26-training-recipe.md).
- **Task-specific heads and losses:** YOLO26 adds targeted designs for instance segmentation, semantic segmentation variants, pose estimation, and oriented detection while keeping a single model pipeline across tasks.

Together, these updates improve the accuracy-latency tradeoff across model scales and deployment targets.

## Key Features

- **DFL-Free Regression**
  YOLO26 removes Distribution Focal Loss (DFL), reducing detection-head complexity and simplifying export.

- **End-to-End NMS-Free Inference**
  Unlike traditional detectors that rely on NMS as a separate post-processing step, YOLO26 is **natively end-to-end** by default. Predictions are generated directly, reducing latency and making production integration simpler.

- **Progressive Loss + STAL**
  Progressive Loss shifts training emphasis toward the inference-time head, while STAL improves positive label coverage for small objects.

- **MuSGD Optimizer**
  A hybrid optimizer that combines [SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html) with [Muon](https://arxiv.org/abs/2502.16982), adapting optimization ideas from large language model training to computer vision.

- **Efficient Deployment**
  The simplified head and NMS-free default path reduce inference overhead across export targets and hardware profiles, including the paper's reported CPU ONNX speedup for YOLO26n versus YOLO11n.

- **Instance Segmentation Enhancements**
  Introduces semantic segmentation loss to improve model convergence and an upgraded proto module that leverages multi-scale information for superior mask quality. The paper reports gains over YOLO11 of up to +2.5 box AP and +3.7 mask AP on COCO instance segmentation.

- **Precision Pose Estimation**
  Integrates [Residual Log-Likelihood Estimation](https://arxiv.org/abs/2107.11291) (RLE) for more accurate keypoint localization and optimizes the decoding process for increased inference speed. The paper reports up to +7.2 AP over YOLO11 on COCO pose estimation.

- **Refined OBB Decoding**
  Introduces a specialized angle loss to improve detection accuracy for square-shaped objects and optimizes OBB decoding to resolve boundary discontinuity issues. The paper reports up to +3.4 mAP over YOLO11 on DOTA-v1.0 oriented detection.

![Ultralytics YOLO26 End-to-End Comparison Plots](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/Ultralytics-YOLO26-Benchmark-E2E.jpg)

---

## Supported Tasks and Modes

YOLO26 supports the standard Ultralytics task set across five model scales:

| Model       | Filenames                                                                                 | Task                                          | Inference | Validation | Training | Export |
| ----------- | ----------------------------------------------------------------------------------------- | --------------------------------------------- | --------- | ---------- | -------- | ------ |
| YOLO26      | `yolo26n.pt` `yolo26s.pt` `yolo26m.pt` `yolo26l.pt` `yolo26x.pt`                          | [Detection](../tasks/detect.md)               | ✅        | ✅         | ✅       | ✅     |
| YOLO26-seg  | `yolo26n-seg.pt` `yolo26s-seg.pt` `yolo26m-seg.pt` `yolo26l-seg.pt` `yolo26x-seg.pt`      | [Instance Segmentation](../tasks/segment.md)  | ✅        | ✅         | ✅       | ✅     |
| YOLO26-sem  | `yolo26n-sem.pt` `yolo26s-sem.pt` `yolo26m-sem.pt` `yolo26l-sem.pt` `yolo26x-sem.pt`      | [Semantic Segmentation](../tasks/semantic.md) | ✅        | ✅         | ✅       | ✅     |
| YOLO26-pose | `yolo26n-pose.pt` `yolo26s-pose.pt` `yolo26m-pose.pt` `yolo26l-pose.pt` `yolo26x-pose.pt` | [Pose/Keypoints](../tasks/pose.md)            | ✅        | ✅         | ✅       | ✅     |
| YOLO26-obb  | `yolo26n-obb.pt` `yolo26s-obb.pt` `yolo26m-obb.pt` `yolo26l-obb.pt` `yolo26x-obb.pt`      | [Oriented Detection](../tasks/obb.md)         | ✅        | ✅         | ✅       | ✅     |
| YOLO26-cls  | `yolo26n-cls.pt` `yolo26s-cls.pt` `yolo26m-cls.pt` `yolo26l-cls.pt` `yolo26x-cls.pt`      | [Classification](../tasks/classify.md)        | ✅        | ✅         | ✅       | ✅     |

This unified framework covers real-time detection, instance segmentation, semantic segmentation, classification, pose estimation, and oriented object detection with training, validation, inference, and export support.

!!! note "Architecture-only variants"

    [`yolo26-p2.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26-p2.yaml) and [`yolo26-p6.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26-p6.yaml) add a P2 (small-object) or P6 (large-input) detection head and are shipped as YAML architectures only. No scale-specific `yolo26*-p2.pt` or `yolo26*-p6.pt` weights are released. Instantiate a scaled config from YAML (for example, `YOLO("yolo26n-p6.yaml")`) and train or fine-tune it as needed.

---

## Performance Metrics

!!! tip "Performance"

    === "Detection (COCO)"

        See [Detection Docs](../tasks/detect.md) for usage examples with these models trained on [COCO](../datasets/detect/coco.md), which include 80 pretrained classes.

        --8<-- "docs/macros/yolo-det-perf.md"

    === "Segmentation (COCO)"

        See [Segmentation Docs](../tasks/segment.md) for usage examples with these models trained on [COCO](../datasets/segment/coco.md), which include 80 pretrained classes.

        --8<-- "docs/macros/yolo-seg-perf.md"

    === "Semantic Segmentation (Cityscapes)"

        See [Semantic Segmentation Docs](../tasks/semantic.md) for usage examples with these models trained on [Cityscapes](../datasets/semantic/cityscapes.md), which include 19 pretrained classes.

        --8<-- "docs/macros/yolo-semantic-perf.md"

    === "Classification (ImageNet)"

        See [Classification Docs](../tasks/classify.md) for usage examples with these models trained on [ImageNet](../datasets/classify/imagenet.md), which include 1000 pretrained classes.

        --8<-- "docs/macros/yolo-cls-perf.md"

    === "Pose (COCO)"

        See [Pose Estimation Docs](../tasks/pose.md) for usage examples with these models trained on [COCO](../datasets/pose/coco.md), which include 1 pretrained class, 'person'.

        --8<-- "docs/macros/yolo-pose-perf.md"

    === "OBB (DOTAv1)"

        See [Oriented Detection Docs](../tasks/obb.md) for usage examples with these models trained on [DOTAv1](../datasets/obb/dota-v2.md#dota-v10), which include 15 pretrained classes.

        --8<-- "docs/macros/yolo-obb-perf.md"

_Params and FLOPs values are for the fused model after `model.fuse()`, which merges Conv and BatchNorm layers and removes the auxiliary one-to-many detection head. Pretrained checkpoints retain the full training architecture and may show higher counts._

---

## Usage Examples

This section provides simple YOLO26 training and inference examples. For full documentation on these and other [modes](../modes/index.md), see the [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md), and [Export](../modes/export.md) docs pages.

Note that the example below is for YOLO26 [Detect](../tasks/detect.md) models for [object detection](https://www.ultralytics.com/glossary/object-detection). For additional supported tasks, see the [Segment](../tasks/segment.md), [Semantic Segmentation](../tasks/semantic.md), [Classify](../tasks/classify.md), [OBB](../tasks/obb.md), and [Pose](../tasks/pose.md) docs.

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

    - **One-to-One Head (Default)**: Produces end-to-end predictions without NMS, outputting `(N, 300, 6)` with a maximum of 300 detections per image. This head is optimized for fast inference and simplified deployment.
    - **One-to-Many Head**: Generates traditional YOLO outputs requiring NMS post-processing, outputting `(N, nc + 4, 8400)` where `nc` is the number of classes. This head typically achieves slightly higher accuracy at the cost of additional processing.

    You can switch between heads during export, prediction, or validation:

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Use one-to-one head (default, no NMS required)
        results = model.predict("image.jpg")  # inference
        metrics = model.val(data="coco.yaml")  # validation
        model.export(format="onnx")  # export

        # Use one-to-many head (requires NMS)
        results = model.predict("image.jpg", end2end=False)  # inference
        metrics = model.val(data="coco.yaml", end2end=False)  # validation
        model.export(format="onnx", end2end=False)  # export
        ```

    === "CLI"

        ```bash
        # Use one-to-one head (default, no NMS required)
        yolo predict model=yolo26n.pt source=image.jpg
        yolo val model=yolo26n.pt data=coco.yaml
        yolo export model=yolo26n.pt format=onnx

        # Use one-to-many head (requires NMS)
        yolo predict model=yolo26n.pt source=image.jpg end2end=False
        yolo val model=yolo26n.pt data=coco.yaml end2end=False
        yolo export model=yolo26n.pt format=onnx end2end=False
        ```

    The choice depends on your deployment requirements: use the one-to-one head for maximum speed and simplicity, or the one-to-many head when accuracy is the top priority.

## YOLOE-26: Open-Vocabulary Detection and Segmentation

YOLO26 also powers [YOLOE-26](yoloe.md), an open-vocabulary variant that detects and segments object categories from **text prompts**, **visual prompts**, or a **prompt-free mode** instead of a fixed class list learned at training time. YOLOE-26 keeps YOLO26's NMS-free, end-to-end (e2e) design, so open-vocabulary inference stays fast enough for dynamic environments where target categories change over time. YOLOE-26x reaches **40.6 AP** on LVIS minival under text prompting, **38.5 AP** under visual prompting, and **31.1 AP** in the prompt-free Non-E2E setting.

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

---

## FAQ

### What are the key improvements in YOLO26?

- **DFL-free regression**: Simplifies the detection head and export path
- **End-to-end NMS-free inference**: Removes NMS from the default inference path
- **Progressive Loss + STAL**: Improves training alignment and small-object label coverage
- **MuSGD optimizer**: Combines SGD with Muon-inspired optimization for stable training
- **Task-specific heads and losses**: Improves segmentation, pose, and oriented detection support

### What tasks does YOLO26 support?

YOLO26 is a **unified model family**, providing end-to-end support for multiple computer vision tasks:

- [Object Detection](../tasks/detect.md)
- [Instance Segmentation](../tasks/segment.md)
- [Semantic Segmentation](../tasks/semantic.md)
- [Image Classification](../tasks/classify.md)
- [Pose Estimation](../tasks/pose.md)
- [Oriented Object Detection (OBB)](../tasks/obb.md)

Each size variant (n, s, m, l, x) supports all tasks, plus open-vocabulary versions via [YOLOE-26](yoloe.md).

### Why is YOLO26 efficient for deployment?

YOLO26 improves deployment efficiency with:

- Native end-to-end inference without NMS by default
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
