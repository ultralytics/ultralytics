---
comments: true
description: Discover the features and benefits of RT-DETR, Baidu’s efficient and adaptable real-time object detector powered by Vision Transformers, including pre-trained models.
keywords: RT-DETR, Baidu, Vision Transformers, object detection, real-time performance, CUDA, TensorRT, IoU-aware query selection, Ultralytics, Python API, PaddlePaddle
---

# Baidu's RT-DETR: A Vision Transformer-Based Real-Time Object Detector

## Overview

Real-Time Detection Transformer (RT-DETR), developed by Baidu, is a cutting-edge end-to-end object detector that provides real-time performance while maintaining high accuracy. It leverages the power of Vision Transformers (ViT) to efficiently process multiscale features by decoupling intra-scale interaction and cross-scale fusion. RT-DETR is highly adaptable, supporting flexible adjustment of inference speed using different decoder layers without retraining. The model excels on accelerated backends like CUDA with TensorRT, outperforming many other real-time object detectors.

![Model example image](https://user-images.githubusercontent.com/26833433/238963168-90e8483f-90aa-4eb6-a5e1-0d408b23dd33.png)
**Overview of Baidu's RT-DETR.** The RT-DETR model architecture diagram shows the last three stages of the backbone {S3, S4, S5} as the input to the encoder. The efficient hybrid encoder transforms multiscale features into a sequence of image features through intrascale feature interaction (AIFI) and cross-scale feature-fusion module (CCFM). The IoU-aware query selection is employed to select a fixed number of image features to serve as initial object queries for the decoder. Finally, the decoder with auxiliary prediction heads iteratively optimizes object queries to generate boxes and confidence scores ([source](https://arxiv.org/pdf/2304.08069.pdf)).

### Key Features

- **Efficient Hybrid Encoder:** Baidu's RT-DETR uses an efficient hybrid encoder that processes multi-scale features by decoupling intra-scale interaction and cross-scale fusion. This unique Vision Transformers-based design reduces computational costs and allows for real-time object detection.
- **IoU-aware Query Selection:** Baidu's RT-DETR improves object query initialization by utilizing IoU-aware query selection. This allows the model to focus on the most relevant objects in the scene, enhancing the detection accuracy.
- **Adaptable Inference Speed:** Baidu's RT-DETR supports flexible adjustments of inference speed by using different decoder layers without the need for retraining. This adaptability facilitates practical application in various real-time object detection scenarios.

## Pre-trained Models

The Ultralytics Python API provides pre-trained PaddlePaddle RT-DETR models with different scales:

- RT-DETR-L: 53.0% AP on COCO val2017, 114 FPS on T4 GPU
- RT-DETR-X: 54.8% AP on COCO val2017, 74 FPS on T4 GPU

## Usage

### Python API

```python
from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")
model.info()  # display model information
model.train(data="coco8.yaml")   # train
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
| Training   | :heavy_check_mark: |

# Citations and Acknowledgements

If you use Baidu's RT-DETR in your research or development work, please cite the [original paper](https://arxiv.org/abs/2304.08069):

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

We would like to acknowledge Baidu and the [PaddlePaddle](https://github.com/PaddlePaddle/PaddleDetection) team for creating and maintaining this valuable resource for the computer vision community. Their contribution to the field with the development of the Vision Transformers-based real-time object detector, RT-DETR, is greatly appreciated.

*Keywords: RT-DETR, Transformer, ViT, Vision Transformers, Baidu RT-DETR, PaddlePaddle, Paddle Paddle RT-DETR, real-time object detection, Vision Transformers-based object detection, pre-trained PaddlePaddle RT-DETR models, Baidu's RT-DETR usage, Ultralytics Python API*