---
comments: true
description: Optimize YOLOv8 models for mobile and embedded devices by exporting to NCNN format. Enhance performance in resource-constrained environments.
keywords: Ultralytics, YOLOv8, NCNN, model export, machine learning, deployment, mobile, embedded systems, deep learning, AI models
---

# How to Export to NCNN from YOLOv8 for Smooth Deployment

Deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models on devices with limited computational power, such as mobile or embedded systems, can be tricky. You need to make sure you use a format optimized for optimal performance. This makes sure that even devices with limited processing power can handle advanced computer vision tasks well.

The export to NCNN format feature allows you to optimize your [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) models for lightweight device-based applications. In this guide, we'll walk you through how to convert your models to the NCNN format, making it easier for your models to perform well on various mobile and embedded devices.

## Why should you export to NCNN?

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/ncnn-overview.avif" alt="NCNN overview">
</p>

The [NCNN](https://github.com/Tencent/ncnn) framework, developed by Tencent, is a high-performance [neural network](https://www.ultralytics.com/glossary/neural-network-nn) inference computing framework optimized specifically for mobile platforms, including mobile phones, embedded devices, and IoT devices. NCNN is compatible with a wide range of platforms, including Linux, Android, iOS, and macOS.

NCNN is known for its fast processing speed on mobile CPUs and enables rapid deployment of [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models to mobile platforms. This makes it easier to build smart apps, putting the power of AI right at your fingertips.

## Key Features of NCNN Models

NCNN models offer a wide range of key features that enable on-device [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) by helping developers run their models on mobile, embedded, and edge devices:

- **Efficient and High-Performance**: NCNN models are made to be efficient and lightweight, optimized for running on mobile and embedded devices like Raspberry Pi with limited resources. They can also achieve high performance with high [accuracy](https://www.ultralytics.com/glossary/accuracy) on various computer vision-based tasks.

- **Quantization**: NCNN models often support quantization which is a technique that reduces the [precision](https://www.ultralytics.com/glossary/precision) of the model's weights and activations. This leads to further improvements in performance and reduces memory footprint.

- **Compatibility**: NCNN models are compatible with popular deep learning frameworks like [TensorFlow](https://www.tensorflow.org/), [Caffe](https://caffe.berkeleyvision.org/), and [ONNX](https://onnx.ai/). This compatibility allows developers to use existing models and workflows easily.

- **Easy to Use**: NCNN models are designed for easy integration into various applications, thanks to their compatibility with popular deep learning frameworks. Additionally, NCNN offers user-friendly tools for converting models between different formats, ensuring smooth interoperability across the development landscape.

## Deployment Options with NCNN

Before we look at the code for exporting YOLOv8 models to the NCNN format, let's understand how NCNN models are normally used.

NCNN models, designed for efficiency and performance, are compatible with a variety of deployment platforms:

- **Mobile Deployment**: Specifically optimized for Android and iOS, allowing for seamless integration into mobile applications for efficient on-device inference.

- **Embedded Systems and IoT Devices**: If you find that running inference on a Raspberry Pi with the [Ultralytics Guide](../guides/raspberry-pi.md) isn't fast enough, switching to an NCNN exported model could help speed things up. NCNN is great for devices like Raspberry Pi and NVIDIA Jetson, especially in situations where you need quick processing right on the device.

- **Desktop and Server Deployment**: Capable of being deployed in desktop and server environments across Linux, Windows, and macOS, supporting development, training, and evaluation with higher computational capacities.

## Export to NCNN: Converting Your YOLOv8 Model

You can expand model compatibility and deployment flexibility by converting YOLOv8 models to NCNN format.

### Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLOv8
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLOv8, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

Before diving into the usage instructions, it's important to note that while all [Ultralytics YOLOv8 models](../models/index.md) are available for exporting, you can ensure that the model you select supports export functionality [here](../modes/export.md).

!!! example "Usage"

    === "Python"

          ```python
          from ultralytics import YOLO

          # Load the YOLOv8 model
          model = YOLO("yolov8n.pt")

          # Export the model to NCNN format
          model.export(format="ncnn")  # creates '/yolov8n_ncnn_model'

          # Load the exported NCNN model
          ncnn_model = YOLO("./yolov8n_ncnn_model")

          # Run inference
          results = ncnn_model("https://ultralytics.com/images/bus.jpg")
          ```

    === "CLI"

          ```bash
          # Export a YOLOv8n PyTorch model to NCNN format
          yolo export model=yolov8n.pt format=ncnn  # creates '/yolov8n_ncnn_model'

          # Run inference with the exported model
          yolo predict model='./yolov8n_ncnn_model' source='https://ultralytics.com/images/bus.jpg'
          ```

For more details about supported export options, visit the [Ultralytics documentation page on deployment options](../guides/model-deployment-options.md).

## Deploying Exported YOLOv8 NCNN Models

After successfully exporting your Ultralytics YOLOv8 models to NCNN format, you can now deploy them. The primary and recommended first step for running a NCNN model is to utilize the YOLO("./model_ncnn_model") method, as outlined in the previous usage code snippet. However, for in-depth instructions on deploying your NCNN models in various other settings, take a look at the following resources:

- **[Android](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android)**: This blog explains how to use NCNN models for performing tasks like [object detection](https://www.ultralytics.com/glossary/object-detection) through Android applications.

- **[macOS](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-macos)**: Understand how to use NCNN models for performing tasks through macOS.

- **[Linux](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux)**: Explore this page to learn how to deploy NCNN models on limited resource devices like Raspberry Pi and other similar devices.

- **[Windows x64 using VS2017](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-windows-x64-using-visual-studio-community-2017)**: Explore this blog to learn how to deploy NCNN models on windows x64 using Visual Studio Community 2017.

## Summary

In this guide, we've gone over exporting Ultralytics YOLOv8 models to the NCNN format. This conversion step is crucial for improving the efficiency and speed of YOLOv8 models, making them more effective and suitable for limited-resource computing environments.

For detailed instructions on usage, please refer to the [official NCNN documentation](https://ncnn.readthedocs.io/en/latest/index.html).

Also, if you're interested in exploring other integration options for Ultralytics YOLOv8, be sure to visit our [integration guide page](index.md) for further insights and information.

## FAQ

### How do I export Ultralytics YOLOv8 models to NCNN format?

To export your Ultralytics YOLOv8 model to NCNN format, follow these steps:

- **Python**: Use the `export` function from the YOLO class.

    ```python
    from ultralytics import YOLO

    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Export to NCNN format
    model.export(format="ncnn")  # creates '/yolov8n_ncnn_model'
    ```

- **CLI**: Use the `yolo` command with the `export` argument.
    ```bash
    yolo export model=yolov8n.pt format=ncnn  # creates '/yolov8n_ncnn_model'
    ```

For detailed export options, check the [Export](../modes/export.md) page in the documentation.

### What are the advantages of exporting YOLOv8 models to NCNN?

Exporting your Ultralytics YOLOv8 models to NCNN offers several benefits:

- **Efficiency**: NCNN models are optimized for mobile and embedded devices, ensuring high performance even with limited computational resources.
- **Quantization**: NCNN supports techniques like quantization that improve model speed and reduce memory usage.
- **Broad Compatibility**: You can deploy NCNN models on multiple platforms, including Android, iOS, Linux, and macOS.

For more details, see the [Export to NCNN](#why-should-you-export-to-ncnn) section in the documentation.

### Why should I use NCNN for my mobile AI applications?

NCNN, developed by Tencent, is specifically optimized for mobile platforms. Key reasons to use NCNN include:

- **High Performance**: Designed for efficient and fast processing on mobile CPUs.
- **Cross-Platform**: Compatible with popular frameworks such as [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) and ONNX, making it easier to convert and deploy models across different platforms.
- **Community Support**: Active community support ensures continual improvements and updates.

To understand more, visit the [NCNN overview](#key-features-of-ncnn-models) in the documentation.

### What platforms are supported for NCNN [model deployment](https://www.ultralytics.com/glossary/model-deployment)?

NCNN is versatile and supports various platforms:

- **Mobile**: Android, iOS.
- **Embedded Systems and IoT Devices**: Devices like Raspberry Pi and NVIDIA Jetson.
- **Desktop and Servers**: Linux, Windows, and macOS.

If running models on a Raspberry Pi isn't fast enough, converting to the NCNN format could speed things up as detailed in our [Raspberry Pi Guide](../guides/raspberry-pi.md).

### How can I deploy Ultralytics YOLOv8 NCNN models on Android?

To deploy your YOLOv8 models on Android:

1. **Build for Android**: Follow the [NCNN Build for Android](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android) guide.
2. **Integrate with Your App**: Use the NCNN Android SDK to integrate the exported model into your application for efficient on-device inference.

For step-by-step instructions, refer to our guide on [Deploying YOLOv8 NCNN Models](#deploying-exported-yolov8-ncnn-models).

For more advanced guides and use cases, visit the [Ultralytics documentation page](../guides/model-deployment-options.md).
