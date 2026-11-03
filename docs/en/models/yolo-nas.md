---
title: YOLO-NAS Object Detection
comments: true
description: YOLO-NAS is Deci AI's quantization-friendly object detector designed with Neural Architecture Search. Run inference, validation and export with Ultralytics.
keywords: YOLO-NAS, Deci AI, object detection, deep learning, Neural Architecture Search, Ultralytics, Python API, YOLO model, SuperGradients, pretrained models, quantization, AutoNAC
---

# YOLO-NAS

!!! note "Important Update"

    [Deci](https://www.linkedin.com/company/deciai/), the original creator of YOLO-NAS, has been acquired by NVIDIA, and these models are no longer actively maintained by Deci. Ultralytics continues to support their use, but no further updates from the original team are expected.

## Overview

Developed by Deci AI, YOLO-NAS is an [object detection](https://www.ultralytics.com/glossary/object-detection) model produced by [Neural Architecture Search](https://www.ultralytics.com/glossary/neural-architecture-search-nas) and designed to address limitations of earlier YOLO models. It improves [quantization](https://www.ultralytics.com/glossary/model-quantization) support and [accuracy](https://www.ultralytics.com/glossary/accuracy)-latency trade-offs.

![YOLO-NAS model accuracy comparison on COCO benchmark](https://cdn.ul.run/i/cbd2b1eab2314cd7b7107a5bbdcc829b.avif) **Overview of YOLO-NAS.** YOLO-NAS uses quantization-aware blocks and selective quantization to preserve accuracy after quantization. When converted to its INT8 quantized version, the model loses little accuracy, an improvement over earlier YOLO models.

### Key Features

- **Quantization-Friendly Basic Block:** YOLO-NAS introduces a new basic block that is friendly to quantization, addressing one of the significant limitations of previous YOLO models.
- **Advanced Training and Quantization:** YOLO-NAS uses advanced training schemes and [post-training quantization](https://www.ultralytics.com/blog/pruning-and-quantization-in-computer-vision-a-quick-guide) to improve performance.
- **AutoNAC Optimization and Pre-training:** YOLO-NAS uses AutoNAC optimization and is pretrained on datasets such as [COCO](../datasets/detect/coco.md), [Objects365](../datasets/detect/objects365.md), and [Roboflow 100](../datasets/detect/roboflow-100.md). This pre-training makes it well suited for downstream object detection tasks in production.

## Pretrained Models

Ultralytics provides pretrained YOLO-NAS models in three sizes, each with a different balance of speed and accuracy:

!!! tip "Performance"

    === "Detection (COCO)"

        | Model            | mAP<sub>50-95</sub> | Latency (ms) |
        | ---------------- | ------------------- | ------------ |
        | YOLO-NAS S       | 47.5                | 3.21         |
        | YOLO-NAS M       | 51.55               | 5.85         |
        | YOLO-NAS L       | **52.22**           | 7.87         |
        | YOLO-NAS S INT-8 | 47.03               | **2.36**     |
        | YOLO-NAS M INT-8 | 51.0                | 3.78         |
        | YOLO-NAS L INT-8 | 52.1                | 4.78         |

Each model variant is designed to offer a balance between [Mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP) and latency, helping you optimize your object detection tasks for both performance and speed.

## Usage Examples

The `ultralytics` Python package provides a Python API and CLI for running YOLO-NAS models.

YOLO-NAS models need the `super-gradients` package, which requires a Python 3.10 environment:

```bash
pip install ultralytics super-gradients "setuptools<81"
```

The following examples show how to use YOLO-NAS models with the `ultralytics` package for inference and validation:

### Inference and Validation Examples

In this example, we validate YOLO-NAS-s on the [COCO8](https://platform.ultralytics.com/ultralytics/datasets/coco8) dataset.

!!! example

    This example provides simple [inference](../modes/predict.md) and [validation](../modes/val.md) code for YOLO-NAS. YOLO-NAS also supports [export](../modes/export.md), but the `ultralytics` package does not support training it.

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) pretrained `*.pt` model files can be passed to the `NAS()` class to create a model instance in Python:

        ```python
        from ultralytics import NAS

        # Load a COCO-pretrained YOLO-NAS-s model
        model = NAS("yolo_nas_s.pt")

        # Display model information (optional)
        model.info()

        # Validate the model on the COCO8 example dataset
        results = model.val(data="coco8.yaml")

        # Run inference with the YOLO-NAS-s model on the 'bus.jpg' image
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        CLI commands are available to directly run the models:

        ```bash
        # Load a COCO-pretrained YOLO-NAS-s model and validate its performance on the COCO8 example dataset
        yolo val model=yolo_nas_s.pt data=coco8.yaml

        # Load a COCO-pretrained YOLO-NAS-s model and run inference on the 'bus.jpg' image
        yolo predict model=yolo_nas_s.pt source=path/to/bus.jpg
        ```

## Supported Tasks and Modes

YOLO-NAS comes in three variants: Small (s), Medium (m), and Large (l). Each variant is designed to cater to different computational and performance needs:

- **YOLO-NAS-s**: Optimized for environments where computational resources are limited but efficiency is key.
- **YOLO-NAS-m**: Offers a balanced approach, suitable for general-purpose object detection with higher accuracy.
- **YOLO-NAS-l**: Tailored for scenarios requiring the highest accuracy, where computational resources are less of a constraint.

Below is a detailed overview of each model, including links to their pretrained weights, the tasks they support, and their compatibility with different operating modes.

| Model Type | Pretrained Weights                                                                            | Tasks Supported                        | Training | Validation | Inference | Export |
| ---------- | --------------------------------------------------------------------------------------------- | -------------------------------------- | -------- | ---------- | --------- | ------ |
| YOLO-NAS-s | [yolo_nas_s.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo_nas_s.pt) | [Object Detection](../tasks/detect.md) | ❌       | ✅         | ✅        | ✅     |
| YOLO-NAS-m | [yolo_nas_m.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo_nas_m.pt) | [Object Detection](../tasks/detect.md) | ❌       | ✅         | ✅        | ✅     |
| YOLO-NAS-l | [yolo_nas_l.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo_nas_l.pt) | [Object Detection](../tasks/detect.md) | ❌       | ✅         | ✅        | ✅     |

## Citations and Acknowledgments

If you use YOLO-NAS in your research or development work, please cite SuperGradients:

!!! quote "Citation"

    === "BibTeX"

        ```bibtex
        @misc{supergradients,
              doi = {10.5281/ZENODO.7789328},
              url = {https://zenodo.org/records/7789328},
              author = {Aharon,  Shay and {Louis-Dupont} and {Ofri Masad} and Yurkova,  Kate and {Lotem Fridman} and {Lkdci} and Khvedchenya,  Eugene and Rubin,  Ran and Bagrov,  Natan and Tymchenko,  Borys and Keren,  Tomer and Zhilko,  Alexander and {Eran-Deci}},
              title = {Super-Gradients},
              publisher = {GitHub},
              journal = {GitHub repository},
              year = {2021},
        }
        ```

We express our gratitude to Deci AI's [SuperGradients](https://github.com/Deci-AI/super-gradients/) team for their efforts in creating and maintaining this valuable resource for the [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) community.

## FAQ

### What is YOLO-NAS and how does it improve over previous YOLO models?

YOLO-NAS, developed by Deci AI, is an object detection model designed with Neural Architecture Search (NAS). It addresses limitations of previous YOLO models by introducing [features](#key-features) like quantization-friendly basic blocks and advanced training schemes, which improve performance, particularly in environments with limited computational resources. YOLO-NAS also supports quantization, maintaining high accuracy even when converted to its INT8 version, enhancing its suitability for production environments.

### How can I integrate YOLO-NAS models into my Python application?

You can integrate YOLO-NAS models into your Python application using the `ultralytics` package. Here's a simple example of how to load a pretrained YOLO-NAS model and perform inference:

```python
from ultralytics import NAS

# Load a COCO-pretrained YOLO-NAS-s model
model = NAS("yolo_nas_s.pt")

# Validate the model on the COCO8 example dataset
results = model.val(data="coco8.yaml")

# Run inference with the YOLO-NAS-s model on the 'bus.jpg' image
results = model("path/to/bus.jpg")
```

For more information, refer to the [Inference and Validation Examples](#inference-and-validation-examples).

### What are the key features of YOLO-NAS and why should I consider using it?

YOLO-NAS introduces several key features that make it a strong choice for quantized object detection:

- **Quantization-Friendly Basic Block:** Enhanced architecture that preserves accuracy after quantization.
- **Advanced Training and Quantization:** Uses advanced training schemes and post-training quantization techniques.
- **AutoNAC Optimization and Pre-training:** Uses AutoNAC optimization and is pretrained on datasets like COCO, Objects365, and Roboflow 100.

These features contribute to its high accuracy, efficient performance, and suitability for deployment in production environments. Learn more in the [Key Features](#key-features) section.

### Which tasks and modes are supported by YOLO-NAS models?

YOLO-NAS models support object detection in the validation, inference, and export [modes](#supported-tasks-and-modes). They do not support training. The supported models include YOLO-NAS-s, YOLO-NAS-m, and YOLO-NAS-l, each tailored to different computational capacities and performance needs.

### Are there pretrained YOLO-NAS models available and how do I access them?

Yes, Ultralytics provides pretrained YOLO-NAS models that you can access directly. These models are pretrained on datasets like COCO. You can download them using the links in the [Supported Tasks and Modes](#supported-tasks-and-modes) section. Here are some examples:

- [YOLO-NAS-s](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo_nas_s.pt)
- [YOLO-NAS-m](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo_nas_m.pt)
- [YOLO-NAS-l](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo_nas_l.pt)
