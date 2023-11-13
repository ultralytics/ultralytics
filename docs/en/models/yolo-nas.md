---
comments: true
description: Explore detailed documentation of YOLO-NAS, a superior object detection model. Learn about its features, pre-trained models, usage with Ultralytics Python API, and more.
keywords: YOLO-NAS, Deci AI, object detection, deep learning, neural architecture search, Ultralytics Python API, YOLO model, pre-trained models, quantization, optimization, COCO, Objects365, Roboflow 100
---

# YOLO-NAS

## Overview

Developed by Deci AI, YOLO-NAS is a groundbreaking object detection foundational model. It is the product of advanced Neural Architecture Search technology, meticulously designed to address the limitations of previous YOLO models. With significant improvements in quantization support and accuracy-latency trade-offs, YOLO-NAS represents a major leap in object detection.

![Model example image](https://learnopencv.com/wp-content/uploads/2023/05/yolo-nas_COCO_map_metrics.png)
**Overview of YOLO-NAS.** YOLO-NAS employs quantization-aware blocks and selective quantization for optimal performance. The model, when converted to its INT8 quantized version, experiences a minimal precision drop, a significant improvement over other models. These advancements culminate in a superior architecture with unprecedented object detection capabilities and outstanding performance.

### Key Features

- **Quantization-Friendly Basic Block:** YOLO-NAS introduces a new basic block that is friendly to quantization, addressing one of the significant limitations of previous YOLO models.
- **Sophisticated Training and Quantization:** YOLO-NAS leverages advanced training schemes and post-training quantization to enhance performance.
- **AutoNAC Optimization and Pre-training:** YOLO-NAS utilizes AutoNAC optimization and is pre-trained on prominent datasets such as COCO, Objects365, and Roboflow 100. This pre-training makes it extremely suitable for downstream object detection tasks in production environments.

## Usage

Ultralytics has made YOLO-NAS models easy to integrate into your Python applications via our `ultralytics` python package. The package provides a user-friendly Python API to streamline the process.

The following examples show how to use YOLO-NAS models with the `ultralytics` package for inference and validation:

### Inference and Validation Examples

In this example we validate YOLO-NAS-s on the COCO8 dataset.

!!! example ""

    This example provides simple inference and validation code for YOLO-NAS. For handling inference results see [Predict](../modes/predict.md) mode. For using YOLO-NAS with additional modes see [Val](../modes/val.md) and [Export](../modes/export.md). YOLO-NAS on the `ultralytics` package does not support training.

    === "Python"

        PyTorch pretrained `*.pt` models files can be passed to the `NAS()` class to create a model instance in python:

        ```python
        from ultralytics import NAS

        # Load a COCO-pretrained YOLO-NAS-s model
        model = NAS('yolo_nas_s.pt')

        # Display model information (optional)
        model.info()

        # Validate the model on the COCO8 example dataset
        results = model.val(data='coco8.yaml')

        # Run inference with the YOLO-NAS-s model on the 'bus.jpg' image
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLI commands are available to directly run the models:

        ```bash
        # Load a COCO-pretrained YOLO-NAS-s model and validate it's performance on the COCO8 example dataset
        yolo val model=yolo_nas_s.pt data=coco8.yaml

        # Load a COCO-pretrained YOLO-NAS-s model and run inference on the 'bus.jpg' image
        yolo predict model=yolo_nas_s.pt source=path/to/bus.jpg
        ```

### Supported Tasks

The YOLO-NAS models are primarily designed for object detection tasks. Additional tasks may be supported by the official repo, the ones supported in the Ultralytics library are shown here.

| Model Type | Pre-trained Weights                               | Tasks            |
|------------|---------------------------------------------------|:----------------:|
| YOLO-NAS   | `yolo_nas_s.pt`, `yolo_nas_m.pt`, `yolo_nas_l.pt` | Object Detection |

### Supported Modes

The YOLO-NAS models support both inference and validation modes, allowing you to predict and validate results with ease. Training mode, however, is currently not supported.

| Mode       | Supported          |
|------------|:------------------:|
| Inference  | :heavy_check_mark: |
| Validation | :heavy_check_mark: |
| Training   | :x:                |

# Performance

=== "Detection (COCO)"

        | Model                                                                                      | size<sup>(pixels)<br> | mAP<sup>val<br>50 | Speed<sup>T4 FP16<br>(ms)<br> |
        | ------------------------------------------------------------------------------------------ | :-------------------: | :---------------: | :---------------------------: |
        | [YOLO-NAS S](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo_nas_s.pt) | 640                   | 47.5              | 3.21                          |
        | [YOLO-NAS M](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo_nas_m.pt) | 640                   | 55.6              | 5.85                          |
        | [YOLO-NAS L](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo_nas_l.pt) | 640                   | 52.2              | 7.87                          |


## Citations and Acknowledgements

If you employ YOLO-NAS in your research or development work, please cite SuperGradients:

!!! note ""

    === "BibTeX"

        ```bibtex
        @misc{supergradients,
              doi = {10.5281/ZENODO.7789328},
              url = {https://zenodo.org/record/7789328},
              author = {Aharon,  Shay and {Louis-Dupont} and {Ofri Masad} and Yurkova,  Kate and {Lotem Fridman} and {Lkdci} and Khvedchenya,  Eugene and Rubin,  Ran and Bagrov,  Natan and Tymchenko,  Borys and Keren,  Tomer and Zhilko,  Alexander and {Eran-Deci}},
              title = {Super-Gradients},
              publisher = {GitHub},
              journal = {GitHub repository},
              year = {2021},
        }
        ```

We express our gratitude to Deci AI's [SuperGradients](https://github.com/Deci-AI/super-gradients/) team for their efforts in creating and maintaining this valuable resource for the computer vision community. We believe YOLO-NAS, with its innovative architecture and superior object detection capabilities, will become a critical tool for developers and researchers alike.

*Keywords: YOLO-NAS, Deci AI, object detection, deep learning, neural architecture search, Ultralytics Python API, YOLO model, SuperGradients, pre-trained models, quantization-friendly basic block, advanced training schemes, post-training quantization, AutoNAC optimization, COCO, Objects365, Roboflow 100*
