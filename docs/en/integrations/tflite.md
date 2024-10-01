---
comments: true
description: Learn how to convert YOLO11 models to TFLite for edge device deployment. Optimize performance and ensure seamless execution on various platforms.
keywords: YOLO11, TFLite, model export, TensorFlow Lite, edge devices, deployment, Ultralytics, machine learning, on-device inference, model optimization
---

# A Guide on YOLO11 Model Export to TFLite for Deployment

<p align="center">
  <img width="75%" src="https://github.com/ultralytics/docs/releases/download/0/tflite-logo.avif" alt="TFLite Logo">
</p>

Deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models on edge devices or embedded devices requires a format that can ensure seamless performance.

The TensorFlow Lite or TFLite export format allows you to optimize your [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models for tasks like [object detection](https://www.ultralytics.com/glossary/object-detection) and [image classification](https://www.ultralytics.com/glossary/image-classification) in edge device-based applications. In this guide, we'll walk through the steps for converting your models to the TFLite format, making it easier for your models to perform well on various edge devices.

## Why should you export to TFLite?

Introduced by Google in May 2017 as part of their TensorFlow framework, [TensorFlow Lite](https://ai.google.dev/edge/litert), or TFLite for short, is an open-source deep learning framework designed for on-device inference, also known as edge computing. It gives developers the necessary tools to execute their trained models on mobile, embedded, and IoT devices, as well as traditional computers.

TensorFlow Lite is compatible with a wide range of platforms, including embedded Linux, Android, iOS, and MCU. Exporting your model to TFLite makes your applications faster, more reliable, and capable of running offline.

## Key Features of TFLite Models

TFLite models offer a wide range of key features that enable on-device machine learning by helping developers run their models on mobile, embedded, and edge devices:

- **On-device Optimization**: TFLite optimizes for on-device ML, reducing latency by processing data locally, enhancing privacy by not transmitting personal data, and minimizing model size to save space.

- **Multiple Platform Support**: TFLite offers extensive platform compatibility, supporting Android, iOS, embedded Linux, and microcontrollers.

- **Diverse Language Support**: TFLite is compatible with various programming languages, including Java, Swift, Objective-C, C++, and Python.

- **High Performance**: Achieves superior performance through hardware acceleration and model optimization.

## Deployment Options in TFLite

Before we look at the code for exporting YOLO11 models to the TFLite format, let's understand how TFLite models are normally used.

TFLite offers various on-device deployment options for machine learning models, including:

- **Deploying with Android and iOS**: Both Android and iOS applications with TFLite can analyze edge-based camera feeds and sensors to detect and identify objects. TFLite also offers native iOS libraries written in [Swift](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/swift) and [Objective-C](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/objc). The architecture diagram below shows the process of deploying a trained model onto Android and iOS platforms using TensorFlow Lite.

 <p align="center">
  <img width="75%" src="https://github.com/ultralytics/docs/releases/download/0/architecture-diagram-tflite-deployment.avif" alt="Architecture">
</p>

- **Implementing with Embedded Linux**: If running inferences on a [Raspberry Pi](https://www.raspberrypi.org/) using the [Ultralytics Guide](../guides/raspberry-pi.md) does not meet the speed requirements for your use case, you can use an exported TFLite model to accelerate inference times. Additionally, it's possible to further improve performance by utilizing a [Coral Edge TPU device](https://coral.withgoogle.com/).

- **Deploying with Microcontrollers**: TFLite models can also be deployed on microcontrollers and other devices with only a few kilobytes of memory. The core runtime just fits in 16 KB on an Arm Cortex M3 and can run many basic models. It doesn't require operating system support, any standard C or C++ libraries, or dynamic memory allocation.

## Export to TFLite: Converting Your YOLO11 Model

You can improve on-device model execution efficiency and optimize performance by converting them to TFLite format.

### Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO11
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLO11, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

Before diving into the usage instructions, it's important to note that while all [Ultralytics YOLO11 models](../models/index.md) are available for exporting, you can ensure that the model you select supports export functionality [here](../modes/export.md).

!!! example "Usage"

    === "Python"

          ```python
          from ultralytics import YOLO

          # Load the YOLO11 model
          model = YOLO("yolo11n.pt")

          # Export the model to TFLite format
          model.export(format="tflite")  # creates 'yolo11n_float32.tflite'

          # Load the exported TFLite model
          tflite_model = YOLO("yolo11n_float32.tflite")

          # Run inference
          results = tflite_model("https://ultralytics.com/images/bus.jpg")
          ```

    === "CLI"

          ```bash
          # Export a YOLO11n PyTorch model to TFLite format
          yolo export model=yolo11n.pt format=tflite  # creates 'yolo11n_float32.tflite'

          # Run inference with the exported model
          yolo predict model='yolo11n_float32.tflite' source='https://ultralytics.com/images/bus.jpg'
          ```

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLO11 TFLite Models

After successfully exporting your Ultralytics YOLO11 models to TFLite format, you can now deploy them. The primary and recommended first step for running a TFLite model is to utilize the YOLO("model.tflite") method, as outlined in the previous usage code snippet. However, for in-depth instructions on deploying your TFLite models in various other settings, take a look at the following resources:

- **[Android](https://ai.google.dev/edge/litert/android)**: A quick start guide for integrating [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) Lite into Android applications, providing easy-to-follow steps for setting up and running [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models.

- **[iOS](https://ai.google.dev/edge/litert/ios/quickstart)**: Check out this detailed guide for developers on integrating and deploying TensorFlow Lite models in iOS applications, offering step-by-step instructions and resources.

- **[End-To-End Examples](https://github.com/tensorflow/examples/tree/master/lite/examples)**: This page provides an overview of various TensorFlow Lite examples, showcasing practical applications and tutorials designed to help developers implement TensorFlow Lite in their machine learning projects on mobile and edge devices.

## Summary

In this guide, we focused on how to export to TFLite format. By converting your Ultralytics YOLO11 models to TFLite model format, you can improve the efficiency and speed of YOLO11 models, making them more effective and suitable for [edge computing](https://www.ultralytics.com/glossary/edge-computing) environments.

For further details on usage, visit the [TFLite official documentation](https://ai.google.dev/edge/litert).

Also, if you're curious about other Ultralytics YOLO11 integrations, make sure to check out our [integration guide page](../integrations/index.md). You'll find tons of helpful info and insights waiting for you there.

## FAQ

### How do I export a YOLO11 model to TFLite format?

To export a YOLO11 model to TFLite format, you can use the Ultralytics library. First, install the required package using:

```bash
pip install ultralytics
```

Then, use the following code snippet to export your model:

```python
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Export the model to TFLite format
model.export(format="tflite")  # creates 'yolo11n_float32.tflite'
```

For CLI users, you can achieve this with:

```bash
yolo export model=yolo11n.pt format=tflite  # creates 'yolo11n_float32.tflite'
```

For more details, visit the [Ultralytics export guide](../modes/export.md).

### What are the benefits of using TensorFlow Lite for YOLO11 [model deployment](https://www.ultralytics.com/glossary/model-deployment)?

TensorFlow Lite (TFLite) is an open-source [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) framework designed for on-device inference, making it ideal for deploying YOLO11 models on mobile, embedded, and IoT devices. Key benefits include:

- **On-device optimization**: Minimize latency and enhance privacy by processing data locally.
- **Platform compatibility**: Supports Android, iOS, embedded Linux, and MCU.
- **Performance**: Utilizes hardware acceleration to optimize model speed and efficiency.

To learn more, check out the [TFLite guide](https://ai.google.dev/edge/litert).

### Is it possible to run YOLO11 TFLite models on Raspberry Pi?

Yes, you can run YOLO11 TFLite models on Raspberry Pi to improve inference speeds. First, export your model to TFLite format as explained [here](#how-do-i-export-a-yolo11-model-to-tflite-format). Then, use a tool like TensorFlow Lite Interpreter to execute the model on your Raspberry Pi.

For further optimizations, you might consider using [Coral Edge TPU](https://coral.withgoogle.com/). For detailed steps, refer to our [Raspberry Pi deployment guide](../guides/raspberry-pi.md).

### Can I use TFLite models on microcontrollers for YOLO11 predictions?

Yes, TFLite supports deployment on microcontrollers with limited resources. TFLite's core runtime requires only 16 KB of memory on an Arm Cortex M3 and can run basic YOLO11 models. This makes it suitable for deployment on devices with minimal computational power and memory.

To get started, visit the [TFLite Micro for Microcontrollers guide](https://ai.google.dev/edge/litert/microcontrollers/overview).

### What platforms are compatible with TFLite exported YOLO11 models?

TensorFlow Lite provides extensive platform compatibility, allowing you to deploy YOLO11 models on a wide range of devices, including:

- **Android and iOS**: Native support through TFLite Android and iOS libraries.
- **Embedded Linux**: Ideal for single-board computers such as Raspberry Pi.
- **Microcontrollers**: Suitable for MCUs with constrained resources.

For more information on deployment options, see our detailed [deployment guide](#deploying-exported-yolo11-tflite-models).

### How do I troubleshoot common issues during YOLO11 model export to TFLite?

If you encounter errors while exporting YOLO11 models to TFLite, common solutions include:

- **Check package compatibility**: Ensure you're using compatible versions of Ultralytics and TensorFlow. Refer to our [installation guide](../quickstart.md).
- **Model support**: Verify that the specific YOLO11 model supports TFLite export by checking [here](../modes/export.md).

For additional troubleshooting tips, visit our [Common Issues guide](../guides/yolo-common-issues.md).
