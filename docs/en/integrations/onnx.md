---
comments: true
description: Learn how to export YOLO11 models to ONNX format for flexible deployment across various platforms with enhanced performance.
keywords: YOLO11, ONNX, model export, Ultralytics, ONNX Runtime, machine learning, model deployment, computer vision, deep learning
---

# ONNX Export for YOLO11 Models

Often, when deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models, you'll need a model format that's both flexible and compatible with multiple platforms.

Exporting [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models to ONNX format streamlines deployment and ensures optimal performance across various environments. This guide will show you how to easily convert your YOLO11 models to ONNX and enhance their scalability and effectiveness in real-world applications.

## ONNX and ONNX Runtime

[ONNX](https://onnx.ai/), which stands for Open [Neural Network](https://www.ultralytics.com/glossary/neural-network-nn) Exchange, is a community project that Facebook and Microsoft initially developed. The ongoing development of ONNX is a collaborative effort supported by various organizations like IBM, Amazon (through AWS), and Google. The project aims to create an open file format designed to represent [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models in a way that allows them to be used across different AI frameworks and hardware.

ONNX models can be used to transition between different frameworks seamlessly. For instance, a [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) model trained in PyTorch can be exported to ONNX format and then easily imported into TensorFlow.

<p align="center">
  <img width="100%" src="https://www.aurigait.com/wp-content/uploads/2023/01/1_unnamed.png" alt="ONNX">
</p>

Alternatively, ONNX models can be used with ONNX Runtime. [ONNX Runtime](https://onnxruntime.ai/) is a versatile cross-platform accelerator for machine learning models that is compatible with frameworks like PyTorch, [TensorFlow](https://www.ultralytics.com/glossary/tensorflow), TFLite, scikit-learn, etc.

ONNX Runtime optimizes the execution of ONNX models by leveraging hardware-specific capabilities. This optimization allows the models to run efficiently and with high performance on various hardware platforms, including CPUs, GPUs, and specialized accelerators.

<p align="center">
  <img width="100%" src="https://www.aurigait.com/wp-content/uploads/2023/01/unnamed-1.png" alt="ONNX with ONNX Runtime">
</p>

Whether used independently or in tandem with ONNX Runtime, ONNX provides a flexible solution for machine learning [model deployment](https://www.ultralytics.com/glossary/model-deployment) and compatibility.

## Key Features of ONNX Models

The ability of ONNX to handle various formats can be attributed to the following key features:

- **Common Model Representation**: ONNX defines a common set of operators (like convolutions, layers, etc.) and a standard data format. When a model is converted to ONNX format, its architecture and weights are translated into this common representation. This uniformity ensures that the model can be understood by any framework that supports ONNX.

- **Versioning and Backward Compatibility**: ONNX maintains a versioning system for its operators. This ensures that even as the standard evolves, models created in older versions remain usable. Backward compatibility is a crucial feature that prevents models from becoming obsolete quickly.

- **Graph-based Model Representation**: ONNX represents models as computational graphs. This graph-based structure is a universal way of representing machine learning models, where nodes represent operations or computations, and edges represent the tensors flowing between them. This format is easily adaptable to various frameworks which also represent models as graphs.

- **Tools and Ecosystem**: There is a rich ecosystem of tools around ONNX that assist in model conversion, visualization, and optimization. These tools make it easier for developers to work with ONNX models and to convert models between different frameworks seamlessly.

## Common Usage of ONNX

Before we jump into how to export YOLO11 models to the ONNX format, let's take a look at where ONNX models are usually used.

### CPU Deployment

ONNX models are often deployed on CPUs due to their compatibility with ONNX Runtime. This runtime is optimized for CPU execution. It significantly improves inference speed and makes real-time CPU deployments feasible.

### Supported Deployment Options

While ONNX models are commonly used on CPUs, they can also be deployed on the following platforms:

- **GPU Acceleration**: ONNX fully supports GPU acceleration, particularly NVIDIA CUDA. This enables efficient execution on NVIDIA GPUs for tasks that demand high computational power.

- **Edge and Mobile Devices**: ONNX extends to edge and mobile devices, perfect for on-device and real-time inference scenarios. It's lightweight and compatible with edge hardware.

- **Web Browsers**: ONNX can run directly in web browsers, powering interactive and dynamic web-based AI applications.

## Exporting YOLO11 Models to ONNX

You can expand model compatibility and deployment flexibility by converting YOLO11 models to ONNX format.

### Installation

To install the required package, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO11
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [YOLO11 Installation guide](../quickstart.md). While installing the required packages for YOLO11, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

Before diving into the usage instructions, be sure to check out the range of [YOLO11 models offered by Ultralytics](../models/index.md). This will help you choose the most appropriate model for your project requirements.

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO11 model
        model = YOLO("yolo11n.pt")

        # Export the model to ONNX format
        model.export(format="onnx")  # creates 'yolo11n.onnx'

        # Load the exported ONNX model
        onnx_model = YOLO("yolo11n.onnx")

        # Run inference
        results = onnx_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to ONNX format
        yolo export model=yolo11n.pt format=onnx  # creates 'yolo11n.onnx'

        # Run inference with the exported model
        yolo predict model=yolo11n.onnx source='https://ultralytics.com/images/bus.jpg'
        ```

### Export Arguments

| Argument   | Type             | Default | Description                                                                                                                                 |
| ---------- | ---------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `onnx`  | Target format for the exported model, defining compatibility with various deployment environments.                                          |
| `imgsz`    | `int` or `tuple` | `640`   | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.           |
| `half`     | `bool`           | `False` | Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.                |
| `dynamic`  | `bool`           | `False` | Allows dynamic input sizes, enhancing flexibility in handling varying image dimensions.                                                     |
| `simplify` | `bool`           | `True`  | Simplifies the model graph with `onnxslim`, potentially improving performance and compatibility.                                            |
| `opset`    | `int`            | `None`  | Specifies the ONNX opset version for compatibility with different ONNX parsers and runtimes. If not set, uses the latest supported version. |
| `nms`      | `bool`           | `False` | Adds Non-Maximum Suppression (NMS), essential for accurate and efficient detection post-processing.                                         |
| `batch`    | `int`            | `1`     | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.     |

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLO11 ONNX Models

Once you've successfully exported your Ultralytics YOLO11 models to ONNX format, the next step is deploying these models in various environments. For detailed instructions on deploying your ONNX models, take a look at the following resources:

- **[ONNX Runtime Python API Documentation](https://onnxruntime.ai/docs/api/python/api_summary.html)**: This guide provides essential information for loading and running ONNX models using ONNX Runtime.

- **[Deploying on Edge Devices](https://onnxruntime.ai/docs/tutorials/iot-edge/)**: Check out this docs page for different examples of deploying ONNX models on edge.

- **[ONNX Tutorials on GitHub](https://github.com/onnx/tutorials)**: A collection of comprehensive tutorials that cover various aspects of using and implementing ONNX models in different scenarios.

## Summary

In this guide, you've learned how to export Ultralytics YOLO11 models to ONNX format to increase their interoperability and performance across various platforms. You were also introduced to the ONNX Runtime and ONNX deployment options.

For further details on usage, visit the [ONNX official documentation](https://onnx.ai/onnx/intro/).

Also, if you'd like to know more about other Ultralytics YOLO11 integrations, visit our [integration guide page](../integrations/index.md). You'll find plenty of useful resources and insights there.

## FAQ

### How do I export YOLO11 models to ONNX format using Ultralytics?

To export your YOLO11 models to ONNX format using Ultralytics, follow these steps:

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO11 model
        model = YOLO("yolo11n.pt")

        # Export the model to ONNX format
        model.export(format="onnx")  # creates 'yolo11n.onnx'

        # Load the exported ONNX model
        onnx_model = YOLO("yolo11n.onnx")

        # Run inference
        results = onnx_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to ONNX format
        yolo export model=yolo11n.pt format=onnx  # creates 'yolo11n.onnx'

        # Run inference with the exported model
        yolo predict model=yolo11n.onnx source='https://ultralytics.com/images/bus.jpg'
        ```

For more details, visit the [export documentation](../modes/export.md).

### What are the advantages of using ONNX Runtime for deploying YOLO11 models?

Using ONNX Runtime for deploying YOLO11 models offers several advantages:

- **Cross-platform compatibility**: ONNX Runtime supports various platforms, such as Windows, macOS, and Linux, ensuring your models run smoothly across different environments.
- **Hardware acceleration**: ONNX Runtime can leverage hardware-specific optimizations for CPUs, GPUs, and dedicated accelerators, providing high-performance inference.
- **Framework interoperability**: Models trained in popular frameworks like [PyTorch](https://www.ultralytics.com/glossary/pytorch) or TensorFlow can be easily converted to ONNX format and run using ONNX Runtime.

Learn more by checking the [ONNX Runtime documentation](https://onnxruntime.ai/docs/api/python/api_summary.html).

### What deployment options are available for YOLO11 models exported to ONNX?

YOLO11 models exported to ONNX can be deployed on various platforms including:

- **CPUs**: Utilizing ONNX Runtime for optimized CPU inference.
- **GPUs**: Leveraging NVIDIA CUDA for high-performance GPU acceleration.
- **Edge devices**: Running lightweight models on edge and mobile devices for real-time, on-device inference.
- **Web browsers**: Executing models directly within web browsers for interactive web-based applications.

For more information, explore our guide on [model deployment options](../guides/model-deployment-options.md).

### Why should I use ONNX format for Ultralytics YOLO11 models?

Using ONNX format for Ultralytics YOLO11 models provides numerous benefits:

- **Interoperability**: ONNX allows models to be transferred between different machine learning frameworks seamlessly.
- **Performance Optimization**: ONNX Runtime can enhance model performance by utilizing hardware-specific optimizations.
- **Flexibility**: ONNX supports various deployment environments, enabling you to use the same model on different platforms without modification.

Refer to the comprehensive guide on [exporting YOLO11 models to ONNX](https://www.ultralytics.com/blog/export-and-optimize-a-yolov8-model-for-inference-on-openvino).

### How can I troubleshoot issues when exporting YOLO11 models to ONNX?

When exporting YOLO11 models to ONNX, you might encounter common issues such as mismatched dependencies or unsupported operations. To troubleshoot these problems:

1. Verify that you have the correct version of required dependencies installed.
2. Check the official [ONNX documentation](https://onnx.ai/onnx/intro/) for supported operators and features.
3. Review the error messages for clues and consult the [Ultralytics Common Issues guide](../guides/yolo-common-issues.md).

If issues persist, contact Ultralytics support for further assistance.
