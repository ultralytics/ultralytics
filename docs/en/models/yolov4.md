---
title: YOLOv4 Real-Time Object Detection
comments: true
description: "YOLOv4 real-time object detector by Bochkovskiy et al.: CSPDarknet53 backbone, PANet neck, bag-of-freebies training, COCO results, and Darknet usage."
keywords: YOLOv4, object detection, real-time detection, Alexey Bochkovskiy, neural networks, machine learning, computer vision
---

# YOLOv4: High-Speed and Precise Object Detection

YOLOv4 is a real-time object detector released in 2020 by Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao in the [AlexeyAB/darknet repository](https://github.com/AlexeyAB/darknet). It was designed for an optimal balance of speed and accuracy.

![YOLOv4 architecture diagram](https://cdn.ul.run/i/1736ab1f573ec8f68d7c41cc21916222.avif) **YOLOv4 architecture diagram**. The diagram shows the backbone, neck, and head components of YOLOv4 and how their layers connect.

## Introduction

YOLOv4 stands for You Only Look Once version 4. It is a real-time [object detection](https://www.ultralytics.com/glossary/object-detection) model developed to address the limitations of previous YOLO versions like [YOLOv3](yolov3.md) and other object detection models. Most earlier [convolutional neural network](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) (CNN) based object detectors were accurate enough only for recommendation-style use; by making real-time detection more accurate, YOLOv4 also targets standalone process management and reduced human input. It runs in real time on a conventional [graphics processing unit (GPU)](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit) and needs only one such GPU for training, which keeps it affordable to use at scale.

## Architecture

YOLOv4 combines several techniques to improve performance. These include Weighted-Residual-Connections (WRC), Cross-Stage-Partial-connections (CSP), Cross mini-[Batch Normalization](https://www.ultralytics.com/glossary/batch-normalization) (CmBN), Self-adversarial-training (SAT), Mish-activation, Mosaic [data augmentation](https://www.ultralytics.com/glossary/data-augmentation), DropBlock [regularization](https://www.ultralytics.com/glossary/regularization), and CIoU loss. Together, these techniques improve accuracy while keeping inference fast.

A typical object detector is composed of several parts including the input, the [backbone](https://www.ultralytics.com/glossary/backbone), the neck, and the head. The backbone is pretrained on [ImageNet](../datasets/classify/imagenet.md) and extracts features from the input image. Common detector backbones include VGG, ResNet, ResNeXt, and DenseNet; YOLOv4 uses CSPDarknet53. The neck part of the detector is used to collect [feature maps](https://www.ultralytics.com/glossary/feature-maps) from different stages and usually includes several bottom-up paths and several top-down paths. The head uses these features to predict the classes and [bounding boxes](https://www.ultralytics.com/glossary/bounding-box) of objects.

## Bag of Freebies

YOLOv4 also makes use of methods known as "bag of freebies," which are techniques that improve the [accuracy](https://www.ultralytics.com/glossary/accuracy) of the model during training without increasing the cost of inference. [Data augmentation](https://www.ultralytics.com/blog/the-ultimate-guide-to-data-augmentation-in-2025) is a common bag of freebies technique used in object detection, which increases the variability of the input images to improve the robustness of the model. Some examples of data augmentation include photometric distortions (adjusting the brightness, contrast, hue, saturation, and noise of an image) and geometric distortions (adding random scaling, cropping, flipping, and rotating). These techniques help the model to generalize better to different types of images.

## Features and Performance

YOLOv4 is designed for optimal speed and accuracy in object detection. The architecture of YOLOv4 includes CSPDarknet53 as the backbone, PANet as the neck, and YOLOv3 as the [detection head](https://www.ultralytics.com/glossary/detection-head). This design lets YOLOv4 run fast enough for real-time applications, and at release it achieved state-of-the-art accuracy on object detection benchmarks like [COCO](../datasets/detect/coco.md).

When compared to other models in the YOLO family, such as [YOLOv5](yolov5.md) and [YOLOv7](yolov7.md), YOLOv4 remains a reference point for the balance between speed and accuracy. Newer models offer higher accuracy and simpler deployment, but YOLOv4's architectural innovations remain relevant for applications requiring real-time performance.

## Usage Examples

YOLOv4 is a Darknet-based model and is **not natively supported** by the Ultralytics Python package: there are no `yolov4.pt` pretrained weights published on [ultralytics/assets](https://github.com/ultralytics/assets/releases) and no `ultralytics/cfg/models/v4/` YAMLs. This page is kept as an architectural reference. Users interested in running YOLOv4 should refer directly to the YOLOv4 GitHub repository for installation and usage instructions.

Here is a brief overview of the typical steps you might take to use YOLOv4:

1. Visit the YOLOv4 GitHub repository: [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet).

2. Follow the instructions provided in the README file for installation. This typically involves cloning the repository, installing necessary dependencies, and setting up any necessary environment variables.

3. Once installation is complete, you can train and use the model as per the usage instructions provided in the repository. This usually involves preparing your dataset, configuring the model parameters, training the model, and then using the trained model to perform object detection.

The exact steps depend on your use case and the current state of the YOLOv4 repository, so follow the instructions provided in the YOLOv4 GitHub repository.

For training and inference within the Ultralytics framework, see [YOLO26](yolo26.md) or [YOLO11](yolo11.md).

## Conclusion

YOLOv4 is an efficient object detection model that balances speed and accuracy. Its architectural choices and bag-of-freebies training techniques let it perform well in real-time object detection tasks. YOLOv4 can be trained and used by anyone with a conventional GPU, making it accessible and practical for a wide range of applications including [surveillance systems](https://www.ultralytics.com/blog/shattering-the-surveillance-status-quo-with-vision-ai), [autonomous vehicles](https://www.ultralytics.com/solutions/computer-vision-in-automotive), and [industrial automation](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision).

For those looking to implement object detection in their projects, YOLOv4 remains a strong contender, especially when real-time performance is a priority. While Ultralytics currently focuses on supporting newer YOLO versions like YOLO26 and YOLO11, the architectural innovations introduced in YOLOv4 have influenced the development of these later models.

## Citations and Acknowledgments

We would like to acknowledge the YOLOv4 authors for their significant contributions in the field of real-time object detection:

!!! quote "Citation"

    === "BibTeX"

        ```bibtex
        @misc{bochkovskiy2020yolov4,
              title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
              author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
              year={2020},
              eprint={2004.10934},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

The original YOLOv4 paper can be found on [arXiv](https://arxiv.org/abs/2004.10934). The authors have made their work publicly available, and the codebase can be accessed on [GitHub](https://github.com/AlexeyAB/darknet). We appreciate their efforts in advancing the field and making their work accessible to the broader community.

## FAQ

### What is YOLOv4 and why should I use it for object detection?

YOLOv4, which stands for "You Only Look Once version 4," is a real-time object detection model developed by Alexey Bochkovskiy and colleagues in 2020. It balances speed and accuracy, making it suitable for real-time applications. YOLOv4's architecture incorporates techniques such as Weighted-Residual-Connections (WRC), Cross-Stage-Partial-connections (CSP), and Self-adversarial-training (SAT), which together achieved state-of-the-art results at release. YOLOv4 runs efficiently on a single conventional GPU; for native training and deployment with the Ultralytics package, use YOLO26.

### How does the architecture of YOLOv4 enhance its performance?

The architecture of YOLOv4 includes several key components: the backbone, the neck, and the head. The backbone, CSPDarknet53, is pretrained on ImageNet and extracts features. The neck, utilizing PANet, aggregates feature maps from different stages. Finally, the head, which follows YOLOv3, predicts the classes and bounding boxes of objects. YOLOv4 also employs "bag of freebies" techniques like mosaic data augmentation and DropBlock regularization, further optimizing its speed and accuracy.

### What are "bag of freebies" in the context of YOLOv4?

"Bag of freebies" refers to methods that improve the training accuracy of YOLOv4 without increasing the cost of inference. These techniques include various forms of data augmentation like photometric distortions (adjusting brightness, contrast, etc.) and geometric distortions (scaling, cropping, flipping, rotating). By increasing the variability of the input images, these augmentations help YOLOv4 generalize better to different types of images, thereby improving its robustness and accuracy without compromising its real-time performance.

### Why is YOLOv4 considered suitable for real-time object detection on conventional GPUs?

YOLOv4 is designed to optimize both speed and accuracy for real-time object detection. It runs on a single conventional GPU for both training and inference, which reduces hardware cost and makes it practical for applications ranging from [recommendation systems](https://www.ultralytics.com/glossary/recommendation-system) to standalone process management.

### How can I get started with YOLOv4 if Ultralytics does not currently support it?

To get started with YOLOv4, you should visit the official [YOLOv4 GitHub repository](https://github.com/AlexeyAB/darknet). Follow the installation instructions provided in the README file, which typically include cloning the repository, installing dependencies, and setting up environment variables. Once installed, you can train the model by preparing your dataset, configuring the model parameters, and following the usage instructions provided. Since Ultralytics does not currently support YOLOv4, it is recommended to refer directly to the YOLOv4 GitHub for the most up-to-date and detailed guidance.
