---
comments: true
description: Dive into YOLO-NAS, Deci's next-generation object detection model, offering breakthroughs in speed and accuracy. Learn how to utilize pre-trained models using the Ultralytics Python API for various tasks.
---

# Deci's YOLO-NAS

## Overview

Developed by Deci AI, YOLO-NAS is a groundbreaking object detection foundational model. It is the product of advanced Neural Architecture Search technology, meticulously designed to address the limitations of previous YOLO models. With significant improvements in quantization support and accuracy-latency trade-offs, YOLO-NAS represents a major leap in object detection.

![Model example image](https://learnopencv.com/wp-content/uploads/2023/05/yolo-nas_COCO_map_metrics.png)
**Overview of YOLO-NAS.** YOLO-NAS employs quantization-aware blocks and selective quantization for optimal performance. The model, when converted to its INT8 quantized version, experiences a minimal precision drop, a significant improvement over other models. These advancements culminate in a superior architecture with unprecedented object detection capabilities and outstanding performance.

### Key Features

- **Quantization-Friendly Basic Block:** YOLO-NAS introduces a new basic block that is friendly to quantization, addressing one of the significant limitations of previous YOLO models.
- **Sophisticated Training and Quantization:** YOLO-NAS leverages advanced training schemes and post-training quantization to enhance performance. 
- **AutoNAC Optimization and Pre-training:** YOLO-NAS utilizes AutoNAC optimization and is pre-trained on prominent datasets such as COCO, Objects365, and Roboflow 100. This pre-training makes it extremely suitable for downstream object detection tasks in production environments.

## Pre-trained Models

Ultralytics provides several pre-trained YOLO-NAS models available for auto-download:

| Model            | mAP   | Latency (ms) |
|------------------|-------|--------------|
| YOLO-NAS S       | 47.5  | 3.21         |
| YOLO-NAS M       | 51.55 | 5.85         |
| YOLO-NAS L       | 52.22 | 7.87         |
| YOLO-NAS S INT-8 | 47.03 | 2.36         |
| YOLO-NAS M INT-8 | 51.0  | 3.78         |
| YOLO-NAS L INT-8 | 52.1  | 4.78         |

## Usage

### Python API

```python
from ultralytics import NAS

model = NAS("yolo_nas_l.pt")
model.info()  # display model information
model.predict("path/to/image.jpg")  # predict
```

### Supported Tasks

| Model Type | Pre-trained Weights                                                                           | Tasks Supported  |
|------------|-----------------------------------------------------------------------------------------------|------------------|
| YOLO-NAS-s | [yolo_nas_s.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo_nas_s.pt) | Object Detection |
| YOLO-NAS-m | [yolo_nas_m.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo_nas_m.pt) | Object Detection |
| YOLO-NAS-l | [yolo_nas_l.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo_nas_l.pt) | Object Detection |

### Supported Modes

| Mode       | Supported          |
|------------|--------------------|
| Inference  | :heavy_check_mark: |
| Validation | :heavy_check_mark: |
| Training   | :x:                |

## Acknowledgements and Citations

If you employ YOLO-NAS in your research or development work, please cite SuperGradients:

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