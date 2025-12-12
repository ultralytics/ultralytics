---
comments: true
description: Explore Baidu's RT-DETR, a Vision Transformer-based real-time object detector offering high accuracy and adaptable inference speed. Learn more with Ultralytics.
keywords: RT-DETR, Baidu, Vision Transformer, real-time object detection, PaddlePaddle, Ultralytics, pretrained models, AI, machine learning, computer vision
---

# Baidu's RT-DETR: A Vision [Transformer](https://www.ultralytics.com/glossary/transformer)-Based Real-Time Object Detector

## Overview

Real-Time Detection Transformer (RT-DETR), developed by Baidu, is a cutting-edge end-to-end object detector that provides real-time performance while maintaining high [accuracy](https://www.ultralytics.com/glossary/accuracy). It is based on the idea of DETR (the NMS-free framework), meanwhile introducing conv-based [backbone](https://www.ultralytics.com/glossary/backbone) and an efficient hybrid encoder to gain real-time speed. RT-DETR efficiently processes multiscale features by decoupling intra-scale interaction and cross-scale fusion. The model is highly adaptable, supporting flexible adjustment of inference speed using different decoder layers without retraining. RT-DETR excels on accelerated backends like CUDA with TensorRT, outperforming many other real-time object detectors.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/i70ecEGB1ro"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Use Baidu's RT-DETR for Object Detection | Inference and Benchmarking with Ultralytics  🚀
</p>

![Model example image](https://github.com/ultralytics/docs/releases/download/0/baidu-rtdetr-model-overview.avif) **Overview of Baidu's RT-DETR.** The RT-DETR model architecture diagram shows the last three stages of the backbone {S3, S4, S5} as the input to the encoder. The efficient hybrid encoder transforms multiscale features into a sequence of image features through intrascale feature interaction (AIFI) and cross-scale feature-fusion module (CCFM). The IoU-aware query selection is employed to select a fixed number of image features to serve as initial object queries for the decoder. Finally, the decoder with auxiliary prediction heads iteratively optimizes object queries to generate boxes and confidence scores ([source](https://arxiv.org/pdf/2304.08069)).

### Key Features

- **Efficient Hybrid Encoder:** Baidu's RT-DETR uses an efficient hybrid encoder that processes multiscale features by decoupling intra-scale interaction and cross-scale fusion. This unique Vision Transformers-based design reduces computational costs and allows for real-time [object detection](https://www.ultralytics.com/glossary/object-detection).
- **IoU-aware Query Selection:** Baidu's RT-DETR improves object query initialization by utilizing IoU-aware query selection. This allows the model to focus on the most relevant objects in the scene, enhancing the detection accuracy.
- **Adaptable Inference Speed:** Baidu's RT-DETR supports flexible adjustments of inference speed by using different decoder layers without the need for retraining. This adaptability facilitates practical application in various real-time object detection scenarios.
- **NMS-Free Framework:** Based on DETR, RT-DETR eliminates the need for [non-maximum suppression](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) post-processing, simplifying the detection pipeline and potentially improving efficiency.
- **Anchor-Free Detection:** As an [anchor-free detector](https://www.ultralytics.com/glossary/anchor-free-detectors), RT-DETR simplifies the detection process and may improve generalization across different datasets.

## Pretrained Models

The Ultralytics Python API provides pretrained PaddlePaddle RT-DETR models with different scales:

- RT-DETR-L: 53.0% AP on COCO val2017, 114 FPS on T4 GPU
- RT-DETR-X: 54.8% AP on COCO val2017, 74 FPS on T4 GPU

Additionally, Baidu has released RTDETRv2 in July 2024, which further improves upon the original architecture with enhanced performance metrics.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400" active-models='["RTDETRv2"]'></canvas>

## Usage Examples

This example provides simple RT-DETR training and inference examples. For full documentation on these and other [modes](../modes/index.md) see the [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) and [Export](../modes/export.md) docs pages.

!!! example

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

This table presents the model types, the specific pretrained weights, the tasks supported by each model, and the various modes ([Train](../modes/train.md) , [Val](../modes/val.md), [Predict](../modes/predict.md), [Export](../modes/export.md)) that are supported, indicated by ✅ emojis.

| Model Type          | Pretrained Weights                                                                        | Tasks Supported                        | Inference | Validation | Training | Export |
| ------------------- | ----------------------------------------------------------------------------------------- | -------------------------------------- | --------- | ---------- | -------- | ------ |
| RT-DETR Large       | [rtdetr-l.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-l.pt) | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |
| RT-DETR Extra-Large | [rtdetr-x.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-x.pt) | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |

## Ideal Use Cases

RT-DETR is particularly well-suited for applications requiring both high accuracy and real-time performance:

- **Autonomous Driving**: For reliable environmental perception in self-driving systems where both speed and accuracy are critical. [Learn more about AI in self-driving cars](https://www.ultralytics.com/solutions/ai-in-automotive).
- **Advanced Robotics**: Enabling robots to perform complex tasks requiring accurate object recognition and interaction in dynamic environments. [Explore AI's role in robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics).
- **Medical Imaging**: For applications in healthcare where precision in object detection can be crucial for diagnostics. [Discover AI in healthcare](https://www.ultralytics.com/solutions/ai-in-healthcare).
- **Surveillance Systems**: For security applications requiring real-time monitoring with high detection accuracy. [Learn about security alarm systems](https://docs.ultralytics.com/guides/security-alarm-system/).
- **Satellite Image Analysis**: For detailed analysis of high-resolution imagery where global context understanding is important. [Read about computer vision in satellite imagery](https://www.ultralytics.com/blog/using-computer-vision-to-analyze-satellite-imagery).

## Citations and Acknowledgments

If you use Baidu's RT-DETR in your research or development work, please cite the [original paper](https://arxiv.org/abs/2304.08069):

!!! quote ""

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

For RTDETRv2, you can cite the [2024 paper](https://arxiv.org/abs/2407.17140):

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{lv2024rtdetrv2,
              title={RTDETRv2: All-in-One Detection Transformer Beats YOLO and DINO},
              author={Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
              year={2024},
              eprint={2407.17140},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

We would like to acknowledge Baidu and the [PaddlePaddle](https://github.com/PaddlePaddle/PaddleDetection) team for creating and maintaining this valuable resource for the [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) community. Their contribution to the field with the development of the Vision Transformers-based real-time object detector, RT-DETR, is greatly appreciated.

## FAQ

### What is Baidu's RT-DETR model and how does it work?

Baidu's RT-DETR (Real-Time Detection Transformer) is an advanced real-time object detector built upon the Vision Transformer architecture. It efficiently processes multiscale features by decoupling intra-scale interaction and cross-scale fusion through its efficient hybrid encoder. By employing IoU-aware query selection, the model focuses on the most relevant objects, enhancing detection accuracy. Its adaptable inference speed, achieved by adjusting decoder layers without retraining, makes RT-DETR suitable for various real-time object detection scenarios. Learn more about RT-DETR features in the [RT-DETR Arxiv paper](https://arxiv.org/pdf/2304.08069).

### How can I use the pretrained RT-DETR models provided by Ultralytics?

You can leverage Ultralytics Python API to use pretrained PaddlePaddle RT-DETR models. For instance, to load an RT-DETR-l model pretrained on COCO val2017 and achieve high FPS on T4 GPU, you can utilize the following example:

!!! example

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

### Why should I choose Baidu's RT-DETR over other real-time object detectors?

Baidu's RT-DETR stands out due to its efficient hybrid encoder and IoU-aware query selection, which drastically reduce computational costs while maintaining high accuracy. Its unique ability to adjust inference speed by using different decoder layers without retraining adds significant flexibility. This makes it particularly advantageous for applications requiring real-time performance on accelerated backends like CUDA with TensorRT, outclassing many other real-time object detectors. The [transformer architecture](https://www.ultralytics.com/glossary/transformer) also provides better global context understanding compared to traditional CNN-based detectors.

### How does RT-DETR support adaptable inference speed for different real-time applications?

Baidu's RT-DETR allows flexible adjustments of inference speed by using different decoder layers without requiring retraining. This adaptability is crucial for scaling performance across various real-time object detection tasks. Whether you need faster processing for lower [precision](https://www.ultralytics.com/glossary/precision) needs or slower, more accurate detections, RT-DETR can be tailored to meet your specific requirements. This feature is particularly valuable when deploying models across devices with varying computational capabilities.

### Can I use RT-DETR models with other Ultralytics modes, such as training, validation, and export?

Yes, RT-DETR models are compatible with various Ultralytics modes including training, validation, prediction, and export. You can refer to the respective documentation for detailed instructions on how to utilize these modes: [Train](../modes/train.md), [Val](../modes/val.md), [Predict](../modes/predict.md), and [Export](../modes/export.md). This ensures a comprehensive workflow for developing and deploying your object detection solutions. The Ultralytics framework provides a consistent API across different model architectures, making it easy to work with RT-DETR models.
