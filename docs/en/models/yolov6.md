---
comments: true
description: Explore Meituan YOLOv6, a state-of-the-art object detection model striking a balance between speed and accuracy. Dive into features, pre-trained models, and Python usage.
keywords: Meituan YOLOv6, object detection, Ultralytics, YOLOv6 docs, Bi-directional Concatenation, Anchor-Aided Training, pretrained models, real-time applications
---

# Meituan YOLOv6

## Overview

[Meituan](https://about.meituan.com/) YOLOv6 is a cutting-edge object detector that offers remarkable balance between speed and accuracy, making it a popular choice for real-time applications. This model introduces several notable enhancements on its architecture and training scheme, including the implementation of a Bi-directional Concatenation (BiC) module, an anchor-aided training (AAT) strategy, and an improved backbone and neck design for state-of-the-art accuracy on the COCO dataset.

![Meituan YOLOv6](https://user-images.githubusercontent.com/26833433/240750495-4da954ce-8b3b-41c4-8afd-ddb74361d3c2.png)
![Model example image](https://user-images.githubusercontent.com/26833433/240750557-3e9ec4f0-0598-49a8-83ea-f33c91eb6d68.png)
**Overview of YOLOv6.** Model architecture diagram showing the redesigned network components and training strategies that have led to significant performance improvements. (a) The neck of YOLOv6 (N and S are shown). Note for M/L, RepBlocks is replaced with CSPStackRep. (b) The structure of a BiC module. (c) A SimCSPSPPF block. ([source](https://arxiv.org/pdf/2301.05586.pdf)).

### Key Features

- **Bidirectional Concatenation (BiC) Module:** YOLOv6 introduces a BiC module in the neck of the detector, enhancing localization signals and delivering performance gains with negligible speed degradation.
- **Anchor-Aided Training (AAT) Strategy:** This model proposes AAT to enjoy the benefits of both anchor-based and anchor-free paradigms without compromising inference efficiency.
- **Enhanced Backbone and Neck Design:** By deepening YOLOv6 to include another stage in the backbone and neck, this model achieves state-of-the-art performance on the COCO dataset at high-resolution input.
- **Self-Distillation Strategy:** A new self-distillation strategy is implemented to boost the performance of smaller models of YOLOv6, enhancing the auxiliary regression branch during training and removing it at inference to avoid a marked speed decline.

## Usage

You can use YOLOv6 for object detection tasks using the Ultralytics pip package. The following is a sample code snippet showing how to use YOLOv6 models for training:

!!! example ""

    This example provides simple training code for YOLOv6. For more options including training settings see [Train](../modes/train.md) mode. For using YOLOv6 with additional modes see [Predict](../modes/predict.md), [Val](../modes/val.md) and [Export](../modes/export.md).

    === "Python"

        PyTorch pretrained `*.pt` models as well as configuration `*.yaml` files can be passed to the `YOLO()` class to create a model instance in python:

        ```python
        from ultralytics import YOLO

        # Build a YOLOv6n model from scratch
        model = YOLO('yolov6n.yaml')

        # Display model information (optional)
        model.info()

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Run inference with the YOLOv6n model on the 'bus.jpg' image
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLI commands are available to directly run the models:

        ```bash
        # Build a YOLOv6n model from scratch and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolov6n.yaml data=coco8.yaml epochs=100 imgsz=640

        # Build a YOLOv6n model from scratch and run inference on the 'bus.jpg' image
        yolo predict model=yolov6n.yaml source=path/to/bus.jpg
        ```

### Supported Tasks

| Model Type | Pre-trained Weights                                                           | Tasks            |
|------------|-------------------------------------------------------------------------------|:----------------:|
| YOLOv6     | `yolov6-n.pt`, `yolov6-s.pt`, `yolov6-m.pt`, `yolov6-l.pt`, `yolov6-l6.pt`    | Object Detection |

## Supported Modes

| Mode       | Supported          |
|------------|:------------------:|
| Inference  | :white_check_mark: |
| Validation | :white_check_mark: |
| Training   | :white_check_mark: |

## Performance

YOLOv6 provides various pre-trained models with different scales (table retrieved 2023-11-13):

!!! Performance

    === "Detection "

        | Model                  | Size<br>(pixels) | mAP^50-95^<br>box          | Speed (FPS)<br>~Nvidia\ T4<br>TensorRT^FP16^~<br>~batch=1~ | Speed (FPS)<br>~Nvidia\ T4<br>TensorRT^FP16^~<br>~batch=32~ | Params<br>× 10^6^ | FLOPs<br>× 10^9^ |
        | ---------------------- | :--------------: | :------------------------: | :--------------------------------------------------------: | :---------------------------------------------------------: | :---------------: | :--------------: |
        | [YOLOv6-N][yolov6-N]   | 640              | 37.5                       | 779                                                        | 1187                                                        | 4.7               | 11.4             |
        | [YOLOv6-S][yolov6-S]   | 640              | 45.0                       | 339                                                        | 484                                                         | 18.5              | 45.3             |
        | [YOLOv6-M][yolov6-M]   | 640              | 50.0                       | 175                                                        | 226                                                         | 34.9              | 85.8             |
        | [YOLOv6-L][yolov6-L]   | 640              | 52.8                       | 98                                                         | 116                                                         | 59.6              | 150.7            |
        |                        |                  |                            |                                                            |                                                             |                   |                  |
        | [YOLOv6-N6][yolov6-N6] | 1280             | 44.9                       | 228                                                        | 281                                                         | 10.4              | 49.8             |
        | [YOLOv6-S6][yolov6-S6] | 1280             | 50.3                       | 98                                                         | 108                                                         | 41.4              | 198.0            |
        | [YOLOv6-M6][yolov6-M6] | 1280             | 55.2                       | 47                                                         | 55                                                          | 79.6              | 379.5            |
        | [YOLOv6-L6][yolov6-L6] | 1280             | 57.2                       | 26                                                         | 29                                                          | 140.4             | 673.4            |

YOLOv6 also provides quantized models for different precisions and models optimized for mobile platforms. Check the [meituan/YOLOv6 repository Releases](https://github.com/meituan/YOLOv6/releases) for the latest pretrained model weights.

## Citations and Acknowledgements

We would like to acknowledge the authors for their significant contributions in the field of real-time object detection:

!!! note ""

    === "BibTeX"

        ```bibtex
        @misc{li2023yolov6,
              title={YOLOv6 v3.0: A Full-Scale Reloading},
              author={Chuyi Li and Lulu Li and Yifei Geng and Hongliang Jiang and Meng Cheng and Bo Zhang and Zaidan Ke and Xiaoming Xu and Xiangxiang Chu},
              year={2023},
              eprint={2301.05586},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

The original YOLOv6 paper can be found on [arXiv](https://arxiv.org/abs/2301.05586). The authors have made their work publicly available, and the codebase can be accessed on [GitHub](https://github.com/meituan/YOLOv6). We appreciate their efforts in advancing the field and making their work accessible to the broader community.

[yolov6-N]: https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt
[yolov6-S]: https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt
[yolov6-M]: https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6m.pt
[yolov6-L]: https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6l.pt
[yolov6-N6]: https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n6.pt
[yolov6-S6]: https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s6.pt
[yolov6-M6]: https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6m6.pt
[yolov6-L6]: https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6l6.pt
