---
comments: true
description: Explore RT-DETR, a high-performance real-time object detector. Learn how to use pre-trained models with Ultralytics Python API for various tasks.
---

# Baidu Real-Time DETR

## Overview

Real-Time Detection Transformer (RT-DETR) is an end-to-end object detector that provides real-time performance while maintaining high accuracy. It efficiently processes multi-scale features and supports flexible adjustment of inference speed. RT-DETR outperforms many real-time object detectors on accelerated backends like CUDA with TensorRT.

## Pre-trained Models

Ultralytics RT-DETR provides several pre-trained models with different scales:

- RT-DETR-L: 53.0% AP on COCO val2017, 114 FPS on T4 GPU
- RT-DETR-X: 54.8% AP on COCO val2017, 74 FPS on T4 GPU

## Usage

### Python API

```python
from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")
model.info()  # display model information
model.predict(...)  # predict
```

### Supported Tasks

| Model Type          | Pre-trained Weights | Tasks Supported  |
|---------------------|---------------------|------------------|
| RT-DETR Large       | `rtdetr-l.pt`       | Object Detection |
| RT-DETR Extra-Large | `rtdetr-x.pt`       | Object Detection |

### Supported Modes

| Mode       | Supported          |
|------------|--------------------|
| Inference  | :heavy_check_mark: |
| Validation | :heavy_check_mark: |
| Training   | :x: (Coming soon)  |

For more information about the RT-DETR model, please refer to the [original paper](https://arxiv.org/abs/2304.08069) and the [PaddleDetection repository](https://github.com/PaddlePaddle/PaddleDetection).