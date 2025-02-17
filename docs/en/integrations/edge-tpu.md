---
comments: true
description: Learn how to export YOLO11 models to TFLite Edge TPU format for high-speed, low-power inferencing on mobile and embedded devices.
keywords: YOLO11, TFLite Edge TPU, TensorFlow Lite, model export, machine learning, edge computing, neural networks, Ultralytics
---

# Learn to Export to TFLite Edge TPU Format From YOLO11 Model

Deploying computer vision models on devices with limited computational power, such as mobile or embedded systems, can be tricky. Using a model format that is optimized for faster performance simplifies the process. The [TensorFlow Lite](https://ai.google.dev/edge/litert) [Edge TPU](https://coral.ai/docs/edgetpu/models-intro/) or TFLite Edge TPU model format is designed to use minimal power while delivering fast performance for neural networks.

The export to TFLite Edge TPU format feature allows you to optimize your [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models for high-speed and low-power inferencing. In this guide, we'll walk you through converting your models to the TFLite Edge TPU format, making it easier for your models to perform well on various mobile and embedded devices.

## Why Should You Export to TFLite Edge TPU?

Exporting models to [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) Edge TPU makes [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) tasks fast and efficient. This technology suits applications with limited power, computing resources, and connectivity. The Edge TPU is a hardware accelerator by Google. It speeds up TensorFlow Lite models on edge devices. The image below shows an example of the process involved.

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/tflite-edge-tpu-compile-workflow.avif" alt="TFLite Edge TPU">
</p>

The Edge TPU works with quantized models. Quantization makes models smaller and faster without losing much [accuracy](https://www.ultralytics.com/glossary/accuracy). It is ideal for the limited resources of edge computing, allowing applications to respond quickly by reducing latency and allowing for quick data processing locally, without cloud dependency. Local processing also keeps user data private and secure since it's not sent to a remote server.

## Key Features of TFLite Edge TPU

Here are the key features that make TFLite Edge TPU a great model format choice for developers:

- **Optimized Performance on Edge Devices**: The TFLite Edge TPU achieves high-speed neural networking performance through quantization, model optimization, hardware acceleration, and compiler optimization. Its minimalistic architecture contributes to its smaller size and cost-efficiency.

- **High Computational Throughput**: TFLite Edge TPU combines specialized hardware acceleration and efficient runtime execution to achieve high computational throughput. It is well-suited for deploying machine learning models with stringent performance requirements on edge devices.

- **Efficient Matrix Computations**: The TensorFlow Edge TPU is optimized for matrix operations, which are crucial for [neural network](https://www.ultralytics.com/glossary/neural-network-nn) computations. This efficiency is key in machine learning models, particularly those requiring numerous and complex matrix multiplications and transformations.

## Deployment Options with TFLite Edge TPU

Before we jump into how to export YOLO11 models to the TFLite Edge TPU format, let's understand where TFLite Edge TPU models are usually used.

TFLite Edge TPU offers various deployment options for machine learning models, including:

- **On-Device Deployment**: TensorFlow Edge TPU models can be directly deployed on mobile and embedded devices. On-device deployment allows the models to execute directly on the hardware, eliminating the need for cloud connectivity.

- **Edge Computing with Cloud TensorFlow TPUs**: In scenarios where edge devices have limited processing capabilities, TensorFlow Edge TPUs can offload inference tasks to cloud servers equipped with TPUs.

- **Hybrid Deployment**: A hybrid approach combines on-device and cloud deployment and offers a versatile and scalable solution for deploying machine learning models. Advantages include on-device processing for quick responses and [cloud computing](https://www.ultralytics.com/glossary/cloud-computing) for more complex computations.

## Exporting YOLO11 Models to TFLite Edge TPU

You can expand model compatibility and deployment flexibility by converting YOLO11 models to TensorFlow Edge TPU.

### Installation

To install the required package, run:

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

        # Export the model to TFLite Edge TPU format
        model.export(format="edgetpu")  # creates 'yolo11n_full_integer_quant_edgetpu.tflite'

        # Load the exported TFLite Edge TPU model
        edgetpu_model = YOLO("yolo11n_full_integer_quant_edgetpu.tflite")

        # Run inference
        results = edgetpu_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to TFLite Edge TPU format
        yolo export model=yolo11n.pt format=edgetpu  # creates 'yolo11n_full_integer_quant_edgetpu.tflite'

        # Run inference with the exported model
        yolo predict model=yolo11n_full_integer_quant_edgetpu.tflite source='https://ultralytics.com/images/bus.jpg'
        ```

### Export Arguments

| Argument | Type             | Default     | Description                                                                                                                       |
| -------- | ---------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'edgetpu'` | Target format for the exported model, defining compatibility with various deployment environments.                                |
| `imgsz`  | `int` or `tuple` | `640`       | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions. |

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLO11 TFLite Edge TPU Models

After successfully exporting your Ultralytics YOLO11 models to TFLite Edge TPU format, you can now deploy them. The primary and recommended first step for running a TFLite Edge TPU model is to use the YOLO("model_edgetpu.tflite") method, as outlined in the previous usage code snippet.

However, for in-depth instructions on deploying your TFLite Edge TPU models, take a look at the following resources:

- **[Coral Edge TPU on a Raspberry Pi with Ultralytics YOLO11](../guides/coral-edge-tpu-on-raspberry-pi.md)**: Discover how to integrate Coral Edge TPUs with Raspberry Pi for enhanced machine learning capabilities.

- **[Code Examples](https://coral.ai/docs/edgetpu/compiler/)**: Access practical TensorFlow Edge TPU deployment examples to kickstart your projects.

- **[Run Inference on the Edge TPU with Python](https://coral.ai/docs/edgetpu/tflite-python/#overview)**: Explore how to use the TensorFlow Lite Python API for Edge TPU applications, including setup and usage guidelines.

## Summary

In this guide, we've learned how to export Ultralytics YOLO11 models to TFLite Edge TPU format. By following the steps mentioned above, you can increase the speed and power of your [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) applications.

For further details on usage, visit the [Edge TPU official website](https://cloud.google.com/tpu).

Also, for more information on other Ultralytics YOLO11 integrations, please visit our [integration guide page](index.md). There, you'll discover valuable resources and insights.

## FAQ

### How do I export a YOLO11 model to TFLite Edge TPU format?

To export a YOLO11 model to TFLite Edge TPU format, you can follow these steps:

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO11 model
        model = YOLO("yolo11n.pt")

        # Export the model to TFLite Edge TPU format
        model.export(format="edgetpu")  # creates 'yolo11n_full_integer_quant_edgetpu.tflite'

        # Load the exported TFLite Edge TPU model
        edgetpu_model = YOLO("yolo11n_full_integer_quant_edgetpu.tflite")

        # Run inference
        results = edgetpu_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to TFLite Edge TPU format
        yolo export model=yolo11n.pt format=edgetpu  # creates 'yolo11n_full_integer_quant_edgetpu.tflite'

        # Run inference with the exported model
        yolo predict model=yolo11n_full_integer_quant_edgetpu.tflite source='https://ultralytics.com/images/bus.jpg'
        ```

For complete details on exporting models to other formats, refer to our [export guide](../modes/export.md).

### What are the benefits of exporting YOLO11 models to TFLite Edge TPU?

Exporting YOLO11 models to TFLite Edge TPU offers several benefits:

- **Optimized Performance**: Achieve high-speed neural network performance with minimal power consumption.
- **Reduced Latency**: Quick local data processing without the need for cloud dependency.
- **Enhanced Privacy**: Local processing keeps user data private and secure.

This makes it ideal for applications in [edge computing](https://www.ultralytics.com/glossary/edge-computing), where devices have limited power and computational resources. Learn more about [why you should export](#why-should-you-export-to-tflite-edge-tpu).

### Can I deploy TFLite Edge TPU models on mobile and embedded devices?

Yes, TensorFlow Lite Edge TPU models can be deployed directly on mobile and embedded devices. This deployment approach allows models to execute directly on the hardware, offering faster and more efficient inferencing. For integration examples, check our [guide on deploying Coral Edge TPU on Raspberry Pi](../guides/coral-edge-tpu-on-raspberry-pi.md).

### What are some common use cases for TFLite Edge TPU models?

Common use cases for TFLite Edge TPU models include:

- **Smart Cameras**: Enhancing real-time image and video analysis.
- **IoT Devices**: Enabling smart home and industrial automation.
- **Healthcare**: Accelerating medical imaging and diagnostics.
- **Retail**: Improving inventory management and customer behavior analysis.

These applications benefit from the high performance and low power consumption of TFLite Edge TPU models. Discover more about [usage scenarios](#deployment-options-with-tflite-edge-tpu).

### How can I troubleshoot issues while exporting or deploying TFLite Edge TPU models?

If you encounter issues while exporting or deploying TFLite Edge TPU models, refer to our [Common Issues guide](../guides/yolo-common-issues.md) for troubleshooting tips. This guide covers common problems and solutions to help you ensure smooth operation. For additional support, visit our [Help Center](https://docs.ultralytics.com/help/).
