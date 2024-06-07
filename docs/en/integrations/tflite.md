---
comments: true
description: Learn how to convert YOLOv8 models to TFLite for edge device deployment. Optimize performance and ensure seamless execution on various platforms.
keywords: YOLOv8, TFLite, model export, TensorFlow Lite, edge devices, deployment, Ultralytics, machine learning, on-device inference, model optimization
---

# A Guide on YOLOv8 Model Export to TFLite for Deployment

<p align="center">
  <img width="75%" src="https://github.com/ultralytics/ultralytics/assets/26833433/6ecf34b9-9187-4d6f-815c-72394290a4d3" alt="TFLite Logo">
</p>

Deploying computer vision models on edge devices or embedded devices requires a format that can ensure seamless performance.

The TensorFlow Lite or TFLite export format allows you to optimize your [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) models for tasks like object detection and image classification in edge device-based applications. In this guide, we'll walk through the steps for converting your models to the TFLite format, making it easier for your models to perform well on various edge devices.

## Why should you export to TFLite?

Introduced by Google in May 2017 as part of their TensorFlow framework, [TensorFlow Lite](https://www.tensorflow.org/lite/guide), or TFLite for short, is an open-source deep learning framework designed for on-device inference, also known as edge computing. It gives developers the necessary tools to execute their trained models on mobile, embedded, and IoT devices, as well as traditional computers.

TensorFlow Lite is compatible with a wide range of platforms, including embedded Linux, Android, iOS, and MCU. Exporting your model to TFLite makes your applications faster, more reliable, and capable of running offline.

## Key Features of TFLite Models

TFLite models offer a wide range of key features that enable on-device machine learning by helping developers run their models on mobile, embedded, and edge devices:

- **On-device Optimization**: TFLite optimizes for on-device ML, reducing latency by processing data locally, enhancing privacy by not transmitting personal data, and minimizing model size to save space.

- **Multiple Platform Support**: TFLite offers extensive platform compatibility, supporting Android, iOS, embedded Linux, and microcontrollers.

- **Diverse Language Support**: TFLite is compatible with various programming languages, including Java, Swift, Objective-C, C++, and Python.

- **High Performance**: Achieves superior performance through hardware acceleration and model optimization.

## Deployment Options in TFLite

Before we look at the code for exporting YOLOv8 models to the TFLite format, let's understand how TFLite models are normally used.

TFLite offers various on-device deployment options for machine learning models, including:

- **Deploying with Android and iOS**: Both Android and iOS applications with TFLite can analyze edge-based camera feeds and sensors to detect and identify objects. TFLite also offers native iOS libraries written in [Swift](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/swift) and [Objective-C](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/objc). The architecture diagram below shows the process of deploying a trained model onto Android and iOS platforms using TensorFlow Lite.

 <p align="center">
  <img width="75%" src="https://1.bp.blogspot.com/-6fS9FD8KD7g/XhJ1l8y2S4I/AAAAAAAACKw/MW9MQZ8gtiYmUe0naRdN0n2FwkT1l4trACLcBGAsYHQ/s1600/architecture.png" alt="Architecture">
</p>

- **Implementing with Embedded Linux**: If running inferences on a [Raspberry Pi](https://www.raspberrypi.org/) using the [Ultralytics Guide](../guides/raspberry-pi.md) does not meet the speed requirements for your use case, you can use an exported TFLite model to accelerate inference times. Additionally, it's possible to further improve performance by utilizing a [Coral Edge TPU device](https://coral.withgoogle.com/).

- **Deploying with Microcontrollers**: TFLite models can also be deployed on microcontrollers and other devices with only a few kilobytes of memory. The core runtime just fits in 16 KB on an Arm Cortex M3 and can run many basic models. It doesn't require operating system support, any standard C or C++ libraries, or dynamic memory allocation.

## Export to TFLite: Converting Your YOLOv8 Model

You can improve on-device model execution efficiency and optimize performance by converting them to TFLite format.

### Installation

To install the required packages, run:

!!! Tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLOv8
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLOv8, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

Before diving into the usage instructions, it's important to note that while all [Ultralytics YOLOv8 models](../models/index.md) are available for exporting, you can ensure that the model you select supports export functionality [here](../modes/export.md).

!!! Example "Usage"

    === "Python"

          ```python
          from ultralytics import YOLO

          # Load the YOLOv8 model
          model = YOLO("yolov8n.pt")

          # Export the model to TFLite format
          model.export(format="tflite")  # creates 'yolov8n_float32.tflite'

          # Load the exported TFLite model
          tflite_model = YOLO("yolov8n_float32.tflite")

          # Run inference
          results = tflite_model("https://ultralytics.com/images/bus.jpg")
          ```

    === "CLI"

          ```bash
          # Export a YOLOv8n PyTorch model to TFLite format
          yolo export model=yolov8n.pt format=tflite  # creates 'yolov8n_float32.tflite'
          
          # Run inference with the exported model
          yolo predict model='yolov8n_float32.tflite' source='https://ultralytics.com/images/bus.jpg'
          ```

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLOv8 TFLite Models

After successfully exporting your Ultralytics YOLOv8 models to TFLite format, you can now deploy them. The primary and recommended first step for running a TFLite model is to utilize the YOLO("model.tflite") method, as outlined in the previous usage code snippet. However, for in-depth instructions on deploying your TFLite models in various other settings, take a look at the following resources:

- **[Android](https://www.tensorflow.org/lite/android/quickstart)**: A quick start guide for integrating TensorFlow Lite into Android applications, providing easy-to-follow steps for setting up and running machine learning models.

- **[iOS](https://www.tensorflow.org/lite/guide/ios)**: Check out this detailed guide for developers on integrating and deploying TensorFlow Lite models in iOS applications, offering step-by-step instructions and resources.

- **[End-To-End Examples](https://www.tensorflow.org/lite/examples)**: This page provides an overview of various TensorFlow Lite examples, showcasing practical applications and tutorials designed to help developers implement TensorFlow Lite in their machine learning projects on mobile and edge devices.

## Summary

In this guide, we focused on how to export to TFLite format. By converting your Ultralytics YOLOv8 models to TFLite model format, you can improve the efficiency and speed of YOLOv8 models, making them more effective and suitable for edge computing environments.

For further details on usage, visit the [TFLite official documentation](https://www.tensorflow.org/lite/guide).

Also, if you're curious about other Ultralytics YOLOv8 integrations, make sure to check out our [integration guide page](../integrations/index.md). You'll find tons of helpful info and insights waiting for you there.
