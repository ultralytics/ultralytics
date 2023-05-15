---
comments: true
description: Explore RT-DETR, a high-performance real-time object detector. Learn how to use pre-trained models with Ultralytics Python API for various tasks.
---

# RT-DETR

## Overview

Real-Time Detection Transformer (RT-DETR) is an end-to-end object detector that provides real-time performance while maintaining high accuracy. It efficiently processes multi-scale features by decoupling intra-scale interaction and cross-scale fusion, and supports flexible adjustment of inference speed using different decoder layers without retraining. RT-DETR outperforms many real-time object detectors on accelerated backends like CUDA with TensorRT.

### Key Features

- **Efficient Hybrid Encoder:** RT-DETR uses an efficient hybrid encoder that processes multi-scale features by decoupling intra-scale interaction and cross-scale fusion. This design reduces computational costs and allows for real-time object detection.
- **IoU-aware Query Selection:** RT-DETR improves object query initialization by utilizing IoU-aware query selection. This allows the model to focus on the most relevant objects in the scene.
- **Adaptable Inference Speed:** RT-DETR supports flexible adjustments of inference speed by using different decoder layers without the need for retraining. This adaptability facilitates practical application in various real-time object detection scenarios.

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
model.predict("path/to/image.jpg")  # predict
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

# Citations and Acknowledgements

If you use RT-DETR in your research or development work, please cite the [original paper](https://arxiv.org/abs/2304.08069):

```bibtex
@misc{lv2023detrs,
      title={DETRs Beat YOLOs on Real-time Object Detection},
      author={Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
      year={2023},
      eprint={2304.08069},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

We would like to acknowledge Baidu's [PaddlePaddle](https://github.com/PaddlePaddle/PaddleDetection) team for creating and maintaining this valuable resource for the computer vision community.
