---
comments: true
description: Explore our detailed guide on YOLOv4, a state-of-the-art real-time object detector. Understand its architectural highlights, innovative features, and application examples.
keywords: ultralytics, YOLOv4, object detection, neural network, real-time detection, object detector, machine learning
---

# YOLOv4: High-Speed and Precise Object Detection

Welcome to the Ultralytics documentation page for YOLOv4, a state-of-the-art, real-time object detector launched in 2020 by Alexey Bochkovskiy at [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet). YOLOv4 is designed to provide the optimal balance between speed and accuracy, making it an excellent choice for many applications.

![YOLOv4 architecture diagram](https://user-images.githubusercontent.com/26833433/246185689-530b7fe8-737b-4bb0-b5dd-de10ef5aface.png)
**YOLOv4 architecture diagram**. Showcasing the intricate network design of YOLOv4, including the backbone, neck, and head components, and their interconnected layers for optimal real-time object detection.

## Introduction

YOLOv4 stands for You Only Look Once version 4. It is a real-time object detection model developed to address the limitations of previous YOLO versions like [YOLOv3](yolov3.md) and other object detection models. Unlike other convolutional neural network (CNN) based object detectors, YOLOv4 is not only applicable for recommendation systems but also for standalone process management and human input reduction. Its operation on conventional graphics processing units (GPUs) allows for mass usage at an affordable price, and it is designed to work in real-time on a conventional GPU while requiring only one such GPU for training.

## Architecture

YOLOv4 makes use of several innovative features that work together to optimize its performance. These include Weighted-Residual-Connections (WRC), Cross-Stage-Partial-connections (CSP), Cross mini-Batch Normalization (CmBN), Self-adversarial-training (SAT), Mish-activation, Mosaic data augmentation, DropBlock regularization, and CIoU loss. These features are combined to achieve state-of-the-art results.

A typical object detector is composed of several parts including the input, the backbone, the neck, and the head. The backbone of YOLOv4 is pre-trained on ImageNet and is used to predict classes and bounding boxes of objects. The backbone could be from several models including VGG, ResNet, ResNeXt, or DenseNet. The neck part of the detector is used to collect feature maps from different stages and usually includes several bottom-up paths and several top-down paths. The head part is what is used to make the final object detections and classifications.

## Bag of Freebies

YOLOv4 also makes use of methods known as "bag of freebies," which are techniques that improve the accuracy of the model during training without increasing the cost of inference. Data augmentation is a common bag of freebies technique used in object detection, which increases the variability of the input images to improve the robustness of the model. Some examples of data augmentation include photometric distortions (adjusting the brightness, contrast, hue, saturation, and noise of an image) and geometric distortions (adding random scaling, cropping, flipping, and rotating). These techniques help the model to generalize better to different types of images.

## Features and Performance

YOLOv4 is designed for optimal speed and accuracy in object detection. The architecture of YOLOv4 includes CSPDarknet53 as the backbone, PANet as the neck, and YOLOv3 as the detection head. This design allows YOLOv4 to perform object detection at an impressive speed, making it suitable for real-time applications. YOLOv4 also excels in accuracy, achieving state-of-the-art results in object detection benchmarks.

## Usage Examples

As of the time of writing, Ultralytics does not currently support YOLOv4 models. Therefore, any users interested in using YOLOv4 will need to refer directly to the YOLOv4 GitHub repository for installation and usage instructions.

Here is a brief overview of the typical steps you might take to use YOLOv4:

1. Visit the YOLOv4 GitHub repository: [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet).

2. Follow the instructions provided in the README file for installation. This typically involves cloning the repository, installing necessary dependencies, and setting up any necessary environment variables.

3. Once installation is complete, you can train and use the model as per the usage instructions provided in the repository. This usually involves preparing your dataset, configuring the model parameters, training the model, and then using the trained model to perform object detection.

Please note that the specific steps may vary depending on your specific use case and the current state of the YOLOv4 repository. Therefore, it is strongly recommended to refer directly to the instructions provided in the YOLOv4 GitHub repository.

We regret any inconvenience this may cause and will strive to update this document with usage examples for Ultralytics once support for YOLOv4 is implemented.

## Conclusion

YOLOv4 is a powerful and efficient object detection model that strikes a balance between speed and accuracy. Its use of unique features and bag of freebies techniques during training allows it to perform excellently in real-time object detection tasks. YOLOv4 can be trained and used by anyone with a conventional GPU, making it accessible and practical for a wide range of applications.

## Citations and Acknowledgements

We would like to acknowledge the YOLOv4 authors for their significant contributions in the field of real-time object detection:

!!! Note ""

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

The original YOLOv4 paper can be found on [arXiv](https://arxiv.org/pdf/2004.10934.pdf). The authors have made their work publicly available, and the codebase can be accessed on [GitHub](https://github.com/AlexeyAB/darknet). We appreciate their efforts in advancing the field and making their work accessible to the broader community.
