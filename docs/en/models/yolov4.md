---
comments: true
description: Explore YOLOv4, a state-of-the-art real-time object detection model by Alexey Bochkovskiy. Discover its architecture, features, and performance.
keywords: YOLOv4, object detection, real-time detection, Alexey Bochkovskiy, neural networks, machine learning, computer vision
---

# YOLOv4: High-Speed and Precise Object Detection

Welcome to the Ultralytics documentation page for YOLOv4, a state-of-the-art, real-time object detector launched in 2020 by Alexey Bochkovskiy at [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet). YOLOv4 is designed to provide the optimal balance between speed and accuracy, making it an excellent choice for many applications.

![YOLOv4 architecture diagram](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/yolov4-architecture-diagram.avif) **YOLOv4 architecture diagram**. Showcasing the intricate network design of YOLOv4, including the backbone, neck, and head components, and their interconnected layers for optimal real-time object detection.

## Introduction

YOLOv4 stands for You Only Look Once version 4. It is a real-time object detection model developed to address the limitations of previous YOLO versions like [YOLOv3](yolov3.md) and other object detection models. Unlike other [convolutional neural network](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) (CNN) based object detectors, YOLOv4 is not only applicable for recommendation systems but also for standalone process management and human input reduction. Its operation on conventional graphics processing units (GPUs) allows for mass usage at an affordable price, and it is designed to work in real-time on a conventional GPU while requiring only one such GPU for training.

## Architecture

YOLOv4 makes use of several innovative features that work together to optimize its performance. These include Weighted-Residual-Connections (WRC), Cross-Stage-Partial-connections (CSP), Cross mini-[Batch Normalization](https://www.ultralytics.com/glossary/batch-normalization) (CmBN), Self-adversarial-training (SAT), Mish-activation, Mosaic [data augmentation](https://www.ultralytics.com/glossary/data-augmentation), DropBlock [regularization](https://www.ultralytics.com/glossary/regularization), and CIoU loss. These features are combined to achieve state-of-the-art results.

A typical object detector is composed of several parts including the input, the [backbone](https://www.ultralytics.com/glossary/backbone), the neck, and the head. The backbone of YOLOv4 is pretrained on [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/) and is used to predict classes and [bounding boxes](https://www.ultralytics.com/glossary/bounding-box) of objects. The backbone could be from several models including VGG, ResNet, ResNeXt, or DenseNet. The neck part of the detector is used to collect [feature maps](https://www.ultralytics.com/glossary/feature-maps) from different stages and usually includes several bottom-up paths and several top-down paths. The head part is what is used to make the final object detections and classifications.

## Bag of Freebies

YOLOv4 also makes use of methods known as "bag of freebies," which are techniques that improve the [accuracy](https://www.ultralytics.com/glossary/accuracy) of the model during training without increasing the cost of inference. [Data augmentation](https://www.ultralytics.com/blog/the-ultimate-guide-to-data-augmentation-in-2025) is a common bag of freebies technique used in [object detection](https://www.ultralytics.com/glossary/object-detection), which increases the variability of the input images to improve the robustness of the model. Some examples of data augmentation include photometric distortions (adjusting the brightness, contrast, hue, saturation, and noise of an image) and geometric distortions (adding random scaling, cropping, flipping, and rotating). These techniques help the model to generalize better to different types of images.

## Features and Performance

YOLOv4 is designed for optimal speed and accuracy in object detection. The architecture of YOLOv4 includes CSPDarknet53 as the backbone, PANet as the neck, and YOLOv3 as the [detection head](https://www.ultralytics.com/glossary/detection-head). This design allows YOLOv4 to perform object detection at an impressive speed, making it suitable for real-time applications. YOLOv4 also excels in accuracy, achieving state-of-the-art results in object detection benchmarks like [COCO](https://docs.ultralytics.com/datasets/detect/coco/).

When compared to other models in the YOLO family, such as [YOLOv5](https://docs.ultralytics.com/models/yolov5/) and [YOLOv7](https://docs.ultralytics.com/models/yolov7/), YOLOv4 maintains a strong position in the balance between speed and accuracy. While newer models may offer certain advantages, YOLOv4's architectural innovations continue to make it relevant for many applications requiring real-time performance.

## Usage Examples

As of the time of writing, Ultralytics does not currently support YOLOv4 models. Therefore, any users interested in using YOLOv4 will need to refer directly to the YOLOv4 GitHub repository for installation and usage instructions.

Here is a brief overview of the typical steps you might take to use YOLOv4:

1. Visit the YOLOv4 GitHub repository: [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet).

2. Follow the instructions provided in the README file for installation. This typically involves cloning the repository, installing necessary dependencies, and setting up any necessary environment variables.

3. Once installation is complete, you can train and use the model as per the usage instructions provided in the repository. This usually involves preparing your dataset, configuring the model parameters, training the model, and then using the trained model to perform object detection.

Please note that the specific steps may vary depending on your specific use case and the current state of the YOLOv4 repository. Therefore, it is strongly recommended to refer directly to the instructions provided in the YOLOv4 GitHub repository.

We regret any inconvenience this may cause and will strive to update this document with usage examples for Ultralytics once support for YOLOv4 is implemented.

## Conclusion

YOLOv4 is a powerful and efficient object detection model that strikes a balance between speed and accuracy. Its use of unique features and bag of freebies techniques during training allows it to perform excellently in real-time object detection tasks. YOLOv4 can be trained and used by anyone with a conventional GPU, making it accessible and practical for a wide range of applications including [surveillance systems](https://www.ultralytics.com/blog/shattering-the-surveillance-status-quo-with-vision-ai), [autonomous vehicles](https://www.ultralytics.com/solutions/ai-in-automotive), and [industrial automation](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision).

For those looking to implement object detection in their projects, YOLOv4 remains a strong contender, especially when real-time performance is a priority. While Ultralytics currently focuses on supporting newer YOLO versions like [YOLOv8](https://docs.ultralytics.com/models/yolov8/) and [YOLO11](https://docs.ultralytics.com/models/yolo11/), the architectural innovations introduced in YOLOv4 have influenced the development of these later models.

## Citations and Acknowledgments

We would like to acknowledge the YOLOv4 authors for their significant contributions in the field of real-time object detection:

!!! quote ""

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

### What is YOLOv4 and why should I use it for [object detection](https://www.ultralytics.com/glossary/object-detection)?

YOLOv4, which stands for "You Only Look Once version 4," is a state-of-the-art real-time object detection model developed by Alexey Bochkovskiy in 2020. It achieves an optimal balance between speed and [accuracy](https://www.ultralytics.com/glossary/accuracy), making it highly suitable for real-time applications. YOLOv4's architecture incorporates several innovative features like Weighted-Residual-Connections (WRC), Cross-Stage-Partial-connections (CSP), and Self-adversarial-training (SAT), among others, to achieve state-of-the-art results. If you're looking for a high-performance model that operates efficiently on conventional GPUs, YOLOv4 is an excellent choice.

### How does the architecture of YOLOv4 enhance its performance?

The architecture of YOLOv4 includes several key components: the [backbone](https://www.ultralytics.com/glossary/backbone), the neck, and the head. The backbone, which can be models like VGG, ResNet, or CSPDarknet53, is pretrained to predict classes and bounding boxes. The neck, utilizing PANet, connects [feature maps](https://www.ultralytics.com/glossary/feature-maps) from different stages for comprehensive data extraction. Finally, the head, which uses configurations from YOLOv3, makes the final object detections. YOLOv4 also employs "bag of freebies" techniques like mosaic data augmentation and DropBlock regularization, further optimizing its speed and accuracy.

### What are "bag of freebies" in the context of YOLOv4?

"Bag of freebies" refers to methods that improve the training accuracy of YOLOv4 without increasing the cost of inference. These techniques include various forms of data augmentation like photometric distortions (adjusting brightness, contrast, etc.) and geometric distortions (scaling, cropping, flipping, rotating). By increasing the variability of the input images, these augmentations help YOLOv4 generalize better to different types of images, thereby improving its robustness and accuracy without compromising its real-time performance.

### Why is YOLOv4 considered suitable for real-time object detection on conventional GPUs?

YOLOv4 is designed to optimize both speed and accuracy, making it ideal for real-time object detection tasks that require quick and reliable performance. It operates efficiently on conventional GPUs, needing only one for both training and inference. This makes it accessible and practical for various applications ranging from [recommendation systems](https://www.ultralytics.com/glossary/recommendation-system) to standalone process management, thereby reducing the need for extensive hardware setups and making it a cost-effective solution for real-time object detection.

### How can I get started with YOLOv4 if Ultralytics does not currently support it?

To get started with YOLOv4, you should visit the official [YOLOv4 GitHub repository](https://github.com/AlexeyAB/darknet). Follow the installation instructions provided in the README file, which typically include cloning the repository, installing dependencies, and setting up environment variables. Once installed, you can train the model by preparing your dataset, configuring the model parameters, and following the usage instructions provided. Since Ultralytics does not currently support YOLOv4, it is recommended to refer directly to the YOLOv4 GitHub for the most up-to-date and detailed guidance.
