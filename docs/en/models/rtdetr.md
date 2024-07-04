---
comments: true
description: Explore Baidu's RT-DETR, a Vision Transformer-based real-time object detector offering high accuracy and adaptable inference speed. Learn more with Ultralytics.
keywords: RT-DETR, Baidu, Vision Transformer, real-time object detection, PaddlePaddle, Ultralytics, pre-trained models, AI, machine learning, computer vision
---

# Baidu's RT-DETR: A Vision Transformer-Based Real-Time Object Detector

## Overview

Real-Time Detection Transformer (RT-DETR), developed by Baidu, is a cutting-edge end-to-end object detector that provides real-time performance while maintaining high accuracy. It is based on the idea of DETR (the NMS-free framework), meanwhile introducing conv-based backbone and an efficient hybrid encoder to gain real-time speed. RT-DETR efficiently processes multiscale features by decoupling intra-scale interaction and cross-scale fusion. The model is highly adaptable, supporting flexible adjustment of inference speed using different decoder layers without retraining. RT-DETR excels on accelerated backends like CUDA with TensorRT, outperforming many other real-time object detectors.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/SArFQs6CHwk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Real-Time Detection Transformer (RT-DETR)
</p>

![Model example image](https://user-images.githubusercontent.com/26833433/238963168-90e8483f-90aa-4eb6-a5e1-0d408b23dd33.png) **Overview of Baidu's RT-DETR.** The RT-DETR model architecture diagram shows the last three stages of the backbone {S3, S4, S5} as the input to the encoder. The efficient hybrid encoder transforms multiscale features into a sequence of image features through intrascale feature interaction (AIFI) and cross-scale feature-fusion module (CCFM). The IoU-aware query selection is employed to select a fixed number of image features to serve as initial object queries for the decoder. Finally, the decoder with auxiliary prediction heads iteratively optimizes object queries to generate boxes and confidence scores ([source](https://arxiv.org/pdf/2304.08069.pdf)).

### Key Features

- **Efficient Hybrid Encoder:** Baidu's RT-DETR uses an efficient hybrid encoder that processes multiscale features by decoupling intra-scale interaction and cross-scale fusion. This unique Vision Transformers-based design reduces computational costs and allows for real-time object detection.
- **IoU-aware Query Selection:** Baidu's RT-DETR improves object query initialization by utilizing IoU-aware query selection. This allows the model to focus on the most relevant objects in the scene, enhancing the detection accuracy.
- **Adaptable Inference Speed:** Baidu's RT-DETR supports flexible adjustments of inference speed by using different decoder layers without the need for retraining. This adaptability facilitates practical application in various real-time object detection scenarios.

## Pre-trained Models

The Ultralytics Python API provides pre-trained PaddlePaddle RT-DETR models with different scales:

- RT-DETR-L: 53.0% AP on COCO val2017, 114 FPS on T4 GPU
- RT-DETR-X: 54.8% AP on COCO val2017, 74 FPS on T4 GPU

## Usage Examples

This example provides simple RT-DETR training and inference examples. For full documentation on these and other [modes](../modes/index.md) see the [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) and [Export](../modes/export.md) docs pages.

!!! Example

    === "Python"

        ```python
        from ultralytics import RTDETR

        # Load a COCO-pretrained RT-DETR-l model
        model = RTDETR("rtdetr-l.pt")

        # Display model information (optional)
        model.info()

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # Run inference with the RT-DETR-l model on the 'bus.jpg' image
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Load a COCO-pretrained RT-DETR-l model and train it on the COCO8 example dataset for 100 epochs
        yolo train model=rtdetr-l.pt data=coco8.yaml epochs=100 imgsz=640

        # Load a COCO-pretrained RT-DETR-l model and run inference on the 'bus.jpg' image
        yolo predict model=rtdetr-l.pt source=path/to/bus.jpg
        ```

## Supported Tasks and Modes

This table presents the model types, the specific pre-trained weights, the tasks supported by each model, and the various modes ([Train](../modes/train.md) , [Val](../modes/val.md), [Predict](../modes/predict.md), [Export](../modes/export.md)) that are supported, indicated by ✅ emojis.

| Model Type          | Pre-trained Weights                                                                       | Tasks Supported                        | Inference | Validation | Training | Export |
| ------------------- | ----------------------------------------------------------------------------------------- | -------------------------------------- | --------- | ---------- | -------- | ------ |
| RT-DETR Large       | [rtdetr-l.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-l.pt) | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |
| RT-DETR Extra-Large | [rtdetr-x.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-x.pt) | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |

## Citations and Acknowledgements

If you use Baidu's RT-DETR in your research or development work, please cite the [original paper](https://arxiv.org/abs/2304.08069):

!!! Quote ""

    === "BibTeX"

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



## FAQ

### What is Baidu's RT-DETR and how does it improve real-time object detection?

Baidu's RT-DETR, or Real-Time Detection Transformer, is a state-of-the-art object detector designed for real-time applications. It uses a Vision Transformer-based architecture that eschews the traditional Non-Max Suppression (NMS) framework for a more efficient and accurate detection process. Key features include an efficient hybrid encoder for processing multiscale features and IoU-aware query selection for better object query initialization. The model's adaptable inference speed allows for flexible application in various real-time detection scenarios without the need for retraining. 

Learn more about RT-DETR's architecture [here](https://arxiv.org/pdf/2304.08069.pdf).

### How do I get started with training and running inference on RT-DETR using Ultralytics?

To get started with RT-DETR using the Ultralytics Python API, you can follow these simple steps:

1. **Load a pre-trained model**:
    ```python
    from ultralytics import RTDETR
    model = RTDETR("rtdetr-l.pt")
    ```

2. **Train the model**:
    ```python
    results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
    ```

3. **Run inference**:
    ```python
    results = model("path/to/bus.jpg")
    ```

For CLI usage, commands are:
```bash
yolo train model=rtdetr-l.pt data=coco8.yaml epochs=100 imgsz=640
yolo predict model=rtdetr-l.pt source=path/to/bus.jpg
```

Find additional modes and detailed instructions in the [Train](../modes/train.md) and [Predict](../modes/predict.md) documentation.

### What are the benefits of using Ultralytics RT-DETR models for object detection?

Ultralytics RT-DETR models offer several advantages:
- **High Accuracy**: Achieves over 53% AP on COCO val2017.
- **Real-Time Performance**: Capable of running at up to 114 FPS on T4 GPU.
- **Adaptable Inference Speed**: Flexible speed adjustment through decoder layers without retraining.
- **Efficient Processing**: The hybrid encoder design optimizes computational resources.

These benefits make RT-DETR models particularly suitable for applications requiring reliable and fast object detection, such as autonomous driving, surveillance, and robotics. For more details, visit the [modes page](../modes/index.md).

### How does the adaptable inference speed feature of RT-DETR work?

The adaptable inference speed in RT-DETR allows you to adjust the model's inference speed dynamically by using different decoder layers. This means you can balance between speed and accuracy based on your application's needs without retraining the model. For instance, fewer decoder layers can provide faster inference with slightly reduced accuracy, and vice versa. This feature makes RT-DETR highly versatile for various real-time object detection scenarios.

Explore the [official documentation](https://arxiv.org/abs/2304.08069) for further technical details.

### What pre-trained RT-DETR models are available, and what are their performance metrics?

Ultralytics offers pre-trained PaddlePaddle RT-DETR models in different scales:
- **RT-DETR-L**: Achieves 53.0% AP on COCO val2017 with 114 FPS on T4 GPU.
- **RT-DETR-X**: Achieves 54.8% AP on COCO val2017 with 74 FPS on T4 GPU.

These pre-trained models can be quickly integrated into your projects for training and inference. They support tasks like object detection and can be used in various modes such as training, validation, and export. Visit the [Export](../modes/export.md) page to learn more.