---
comments: true
description: Discover YOLO11, the latest advancement in state-of-the-art object detection, offering unmatched accuracy and efficiency for diverse computer vision tasks.
keywords: YOLO11, state-of-the-art object detection, YOLO series, Ultralytics, computer vision, AI, machine learning, deep learning
---

# Ultralytics YOLO11

## Overview

YOLO11 is the latest iteration in the [Ultralytics](https://www.ultralytics.com) YOLO series of real-time object detectors, redefining what's possible with cutting-edge [accuracy](https://www.ultralytics.com/glossary/accuracy), speed, and efficiency. Building upon the impressive advancements of previous YOLO versions, YOLO11 introduces significant improvements in architecture and training methods, making it a versatile choice for a wide range of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks.

![Ultralytics YOLO11 Comparison Plots](https://github.com/user-attachments/assets/a311a4ed-bbf2-43b5-8012-5f183a28a845)

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/rfI5vOo3-_A?si=uLCEBVVXwAHiOYqq&amp;start=5500"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics YOLO11 Announcement at <a href="https://www.ultralytics.com/events/yolovision">YOLO Vision 2024</a>
</p>

## Key Features

- **Enhanced Feature Extraction:** YOLO11 employs an improved backbone and neck architecture, which enhances [feature extraction](https://www.ultralytics.com/glossary/feature-extraction) capabilities for more precise object detection and complex task performance.
- **Optimized for Efficiency and Speed:** YOLO11 introduces refined architectural designs and optimized training pipelines, delivering faster processing speeds and maintaining an optimal balance between accuracy and performance.
- **Greater Accuracy with Fewer Parameters:** With advancements in model design, YOLO11m achieves a higher [mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP) on the COCO dataset while using 22% fewer parameters than YOLOv8m, making it computationally efficient without compromising accuracy.
- **Adaptability Across Environments:** YOLO11 can be seamlessly deployed across various environments, including edge devices, cloud platforms, and systems supporting NVIDIA GPUs, ensuring maximum flexibility.
- **Broad Range of Supported Tasks:** Whether it's object detection, instance segmentation, image classification, pose estimation, or oriented object detection (OBB), YOLO11 is designed to cater to a diverse set of computer vision challenges.

## Supported Tasks and Modes

YOLO11 builds upon the versatile model range introduced in YOLOv8, offering enhanced support across various computer vision tasks:

| Model       | Filenames                                                                                 | Task                                         | Inference | Validation | Training | Export |
| ----------- | ----------------------------------------------------------------------------------------- | -------------------------------------------- | --------- | ---------- | -------- | ------ |
| YOLO11      | `yolo11n.pt` `yolo11s.pt` `yolo11m.pt` `yolo11l.pt` `yolo11x.pt`                          | [Detection](../tasks/detect.md)              | ✅        | ✅         | ✅       | ✅     |
| YOLO11-seg  | `yolo11n-seg.pt` `yolo11s-seg.pt` `yolo11m-seg.pt` `yolo11l-seg.pt` `yolo11x-seg.pt`      | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLO11-pose | `yolo11n-pose.pt` `yolo11s-pose.pt` `yolo11m-pose.pt` `yolo11l-pose.pt` `yolo11x-pose.pt` | [Pose/Keypoints](../tasks/pose.md)           | ✅        | ✅         | ✅       | ✅     |
| YOLO11-obb  | `yolo11n-obb.pt` `yolo11s-obb.pt` `yolo11m-obb.pt` `yolo11l-obb.pt` `yolo11x-obb.pt`      | [Oriented Detection](../tasks/obb.md)        | ✅        | ✅         | ✅       | ✅     |
| YOLO11-cls  | `yolo11n-cls.pt` `yolo11s-cls.pt` `yolo11m-cls.pt` `yolo11l-cls.pt` `yolo11x-cls.pt`      | [Classification](../tasks/classify.md)       | ✅        | ✅         | ✅       | ✅     |

This table provides an overview of the YOLO11 model variants, showcasing their applicability in specific tasks and compatibility with operational modes such as Inference, Validation, Training, and Export. This flexibility makes YOLO11 suitable for a wide range of applications in computer vision, from real-time detection to complex segmentation tasks.

## Performance Metrics

!!! performance

    === "Detection (COCO)"

        See [Detection Docs](../tasks/detect.md) for usage examples with these models trained on [COCO](../datasets/detect/coco.md), which include 80 pre-trained classes.

        | Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
        | ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | --------------------------------------- | ------------------ | ----------------- |
        | [YOLO11n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) | 640                   | 39.5                 | 56.1 ± 0.8                     | 1.5 ± 0.0                               | 2.6                | 6.5               |
        | [YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) | 640                   | 47.0                 | 90.0 ± 1.2                     | 2.5 ± 0.0                               | 9.4                | 21.5              |
        | [YOLO11m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt) | 640                   | 51.5                 | 183.2 ± 2.0                    | 4.7 ± 0.1                               | 20.1               | 68.0              |
        | [YOLO11l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt) | 640                   | 53.4                 | 238.6 ± 1.4                    | 6.2 ± 0.1                               | 25.3               | 86.9              |
        | [YOLO11x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt) | 640                   | 54.7                 | 462.8 ± 6.7                    | 11.3 ± 0.2                              | 56.9               | 194.9             |

    === "Segmentation (COCO)"

        See [Segmentation Docs](../tasks/segment.md) for usage examples with these models trained on [COCO](../datasets/segment/coco.md), which include 80 pre-trained classes.

        | Model                                                                                        | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
        | -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | --------------------------------------- | ------------------ | ----------------- |
        | [YOLO11n-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt) | 640                   | 38.9                 | 32.0                  | 65.9 ± 1.1                     | 1.8 ± 0.0                               | 2.9                | 10.4              |
        | [YOLO11s-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt) | 640                   | 46.6                 | 37.8                  | 117.6 ± 4.9                    | 2.9 ± 0.0                               | 10.1               | 35.5              |
        | [YOLO11m-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt) | 640                   | 51.5                 | 41.5                  | 281.6 ± 1.2                    | 6.3 ± 0.1                               | 22.4               | 123.3             |
        | [YOLO11l-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt) | 640                   | 53.4                 | 42.9                  | 344.2 ± 3.2                    | 7.8 ± 0.2                               | 27.6               | 142.2             |
        | [YOLO11x-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt) | 640                   | 54.7                 | 43.8                  | 664.5 ± 3.2                    | 15.8 ± 0.7                              | 62.1               | 319.0             |

    === "Classification (ImageNet)"

        See [Classification Docs](../tasks/classify.md) for usage examples with these models trained on [ImageNet](../datasets/classify/imagenet.md), which include 1000 pre-trained classes.

        | Model                                                                                        | size<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
        | -------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | ------------------------------ | --------------------------------------- | ------------------ | ------------------------ |
        | [YOLO11n-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt) | 224                   | 70.0             | 89.4             | 5.0 ± 0.3                      | 1.1 ± 0.0                               | 1.6                | 3.3                      |
        | [YOLO11s-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-cls.pt) | 224                   | 75.4             | 92.7             | 7.9 ± 0.2                      | 1.3 ± 0.0                               | 5.5                | 12.1                     |
        | [YOLO11m-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-cls.pt) | 224                   | 77.3             | 93.9             | 17.2 ± 0.4                     | 2.0 ± 0.0                               | 10.4               | 39.3                     |
        | [YOLO11l-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-cls.pt) | 224                   | 78.3             | 94.3             | 23.2 ± 0.3                     | 2.8 ± 0.0                               | 12.9               | 49.4                     |
        | [YOLO11x-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-cls.pt) | 224                   | 79.5             | 94.9             | 41.4 ± 0.9                     | 3.8 ± 0.0                               | 28.4               | 110.4                    |

    === "Pose (COCO)"

        See [Pose Estimation Docs](../tasks/pose.md) for usage examples with these models trained on [COCO](../datasets/pose/coco.md), which include 1 pre-trained class, 'person'.

        | Model                                                                                          | size<br><sup>(pixels) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
        | ---------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ------------------ | ------------------------------ | --------------------------------------- | ------------------ | ----------------- |
        | [YOLO11n-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt) | 640                   | 50.0                  | 81.0               | 52.4 ± 0.5                     | 1.7 ± 0.0                               | 2.9                | 7.6               |
        | [YOLO11s-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt) | 640                   | 58.9                  | 86.3               | 90.5 ± 0.6                     | 2.6 ± 0.0                               | 9.9                | 23.2              |
        | [YOLO11m-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt) | 640                   | 64.9                  | 89.4               | 187.3 ± 0.8                    | 4.9 ± 0.1                               | 20.9               | 71.7              |
        | [YOLO11l-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-pose.pt) | 640                   | 66.1                  | 89.9               | 247.7 ± 1.1                    | 6.4 ± 0.1                               | 26.2               | 90.7              |
        | [YOLO11x-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt) | 640                   | 69.5                  | 91.1               | 488.0 ± 13.9                   | 12.1 ± 0.2                              | 58.8               | 203.3             |

    === "OBB (DOTAv1)"

        See [Oriented Detection Docs](../tasks/obb.md) for usage examples with these models trained on [DOTAv1](../datasets/obb/dota-v2.md#dota-v10), which include 15 pre-trained classes.

        | Model                                                                                        | size<br><sup>(pixels) | mAP<sup>test<br>50 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
        | -------------------------------------------------------------------------------------------- | --------------------- | ------------------ | ------------------------------ | --------------------------------------- | ------------------ | ----------------- |
        | [YOLO11n-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt) | 1024                  | 78.4               | 117.6 ± 0.8                    | 4.4 ± 0.0                               | 2.7                | 17.2              |
        | [YOLO11s-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-obb.pt) | 1024                  | 79.5               | 219.4 ± 4.0                    | 5.1 ± 0.0                               | 9.7                | 57.5              |
        | [YOLO11m-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-obb.pt) | 1024                  | 80.9               | 562.8 ± 2.9                    | 10.1 ± 0.4                              | 20.9               | 183.5             |
        | [YOLO11l-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-obb.pt) | 1024                  | 81.0               | 712.5 ± 5.0                    | 13.5 ± 0.6                              | 26.2               | 232.0             |
        | [YOLO11x-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-obb.pt) | 1024                  | 81.3               | 1408.6 ± 7.7                   | 28.6 ± 1.0                              | 58.8               | 520.2             |

## Usage Examples

This section provides simple YOLO11 training and inference examples. For full documentation on these and other [modes](../modes/index.md), see the [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md), and [Export](../modes/export.md) docs pages.

Note that the example below is for YOLO11 [Detect](../tasks/detect.md) models for [object detection](https://www.ultralytics.com/glossary/object-detection). For additional supported tasks, see the [Segment](../tasks/segment.md), [Classify](../tasks/classify.md), [OBB](../tasks/obb.md), and [Pose](../tasks/pose.md) docs.

!!! example

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) pretrained `*.pt` models as well as configuration `*.yaml` files can be passed to the `YOLO()` class to create a model instance in Python:

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLO11n model
        model = YOLO("yolo11n.pt")

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # Run inference with the YOLO11n model on the 'bus.jpg' image
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        CLI commands are available to directly run the models:

        ```bash
        # Load a COCO-pretrained YOLO11n model and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolo11n.pt data=coco8.yaml epochs=100 imgsz=640

        # Load a COCO-pretrained YOLO11n model and run inference on the 'bus.jpg' image
        yolo predict model=yolo11n.pt source=path/to/bus.jpg
        ```

## Citations and Acknowledgements

If you use YOLO11 or any other software from this repository in your work, please cite it using the following format:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @software{yolo11_ultralytics,
          author = {Glenn Jocher and Jing Qiu},
          title = {Ultralytics YOLO11},
          version = {11.0.0},
          year = {2024},
          url = {https://github.com/ultralytics/ultralytics},
          orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
          license = {AGPL-3.0}
        }
        ```

Please note that the DOI is pending and will be added to the citation once it is available. YOLO11 models are provided under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) and [Enterprise](https://www.ultralytics.com/license) licenses.

## FAQ

### What are the key improvements in Ultralytics YOLO11 compared to previous versions?

Ultralytics YOLO11 introduces several significant advancements over its predecessors. Key improvements include:

- **Enhanced Feature Extraction:** YOLO11 employs an improved backbone and neck architecture, enhancing [feature extraction](https://www.ultralytics.com/glossary/feature-extraction) capabilities for more precise object detection.
- **Optimized Efficiency and Speed:** Refined architectural designs and optimized training pipelines deliver faster processing speeds while maintaining a balance between accuracy and performance.
- **Greater Accuracy with Fewer Parameters:** YOLO11m achieves higher mean Average [Precision](https://www.ultralytics.com/glossary/precision) (mAP) on the COCO dataset with 22% fewer parameters than YOLOv8m, making it computationally efficient without compromising accuracy.
- **Adaptability Across Environments:** YOLO11 can be deployed across various environments, including edge devices, cloud platforms, and systems supporting NVIDIA GPUs.
- **Broad Range of Supported Tasks:** YOLO11 supports diverse computer vision tasks such as object detection, [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation), image classification, pose estimation, and oriented object detection (OBB).

### How do I train a YOLO11 model for object detection?

Training a YOLO11 model for object detection can be done using Python or CLI commands. Below are examples for both methods:

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLO11n model
        model = YOLO("yolo11n.pt")

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Load a COCO-pretrained YOLO11n model and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolo11n.pt data=coco8.yaml epochs=100 imgsz=640
        ```

For more detailed instructions, refer to the [Train](../modes/train.md) documentation.

### What tasks can YOLO11 models perform?

YOLO11 models are versatile and support a wide range of computer vision tasks, including:

- **Object Detection:** Identifying and locating objects within an image.
- **Instance Segmentation:** Detecting objects and delineating their boundaries.
- **[Image Classification](https://www.ultralytics.com/glossary/image-classification):** Categorizing images into predefined classes.
- **Pose Estimation:** Detecting and tracking keypoints on human bodies.
- **Oriented Object Detection (OBB):** Detecting objects with rotation for higher precision.

For more information on each task, see the [Detection](../tasks/detect.md), [Instance Segmentation](../tasks/segment.md), [Classification](../tasks/classify.md), [Pose Estimation](../tasks/pose.md), and [Oriented Detection](../tasks/obb.md) documentation.

### How does YOLO11 achieve greater accuracy with fewer parameters?

YOLO11 achieves greater accuracy with fewer parameters through advancements in model design and optimization techniques. The improved architecture allows for efficient feature extraction and processing, resulting in higher mean Average Precision (mAP) on datasets like COCO while using 22% fewer parameters than YOLOv8m. This makes YOLO11 computationally efficient without compromising on accuracy, making it suitable for deployment on resource-constrained devices.

### Can YOLO11 be deployed on edge devices?

Yes, YOLO11 is designed for adaptability across various environments, including edge devices. Its optimized architecture and efficient processing capabilities make it suitable for deployment on edge devices, cloud platforms, and systems supporting NVIDIA GPUs. This flexibility ensures that YOLO11 can be used in diverse applications, from real-time detection on mobile devices to complex segmentation tasks in cloud environments. For more details on deployment options, refer to the [Export](../modes/export.md) documentation.
