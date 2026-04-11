---
comments: true
description: Optimize YOLO26 models for mobile and embedded devices by exporting to NCNN format. Enhance performance in resource-constrained environments.
keywords: Ultralytics, YOLO26, NCNN, model export, machine learning, deployment, mobile, embedded systems, deep learning, AI models, Vulkan, GPU acceleration
---

# Ultralytics YOLO NCNN Export

Deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models on devices with limited computational power, such as mobile or embedded systems, requires careful format selection. Using an optimized format ensures that even resource-constrained devices can handle advanced computer vision tasks efficiently.

Exporting to NCNN format allows you to optimize your [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) models for lightweight device-based applications. This guide covers how to convert your models to NCNN format for improved performance on mobile and embedded devices.

## Why Export to NCNN?

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ncnn-overview.avif" alt="NCNN high-performance neural network inference framework">
</p>

The [NCNN](https://github.com/Tencent/ncnn) framework, developed by Tencent, is a high-performance [neural network](https://www.ultralytics.com/glossary/neural-network-nn) inference computing framework optimized specifically for mobile platforms, including mobile phones, embedded devices, and IoT devices. NCNN is compatible with a wide range of platforms, including Linux, Android, iOS, and macOS.

NCNN is known for its fast processing speed on mobile CPUs and enables rapid deployment of [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models to mobile platforms, making it an excellent choice for building AI-powered applications.

## Key Features of NCNN Models

NCNN models provide several key features that enable on-device [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml), helping developers deploy models on mobile, embedded, and edge devices:

- **Efficient and High-Performance**: NCNN models are lightweight and optimized for mobile and embedded devices like Raspberry Pi with limited resources, while maintaining high [accuracy](https://www.ultralytics.com/glossary/accuracy) on computer vision tasks.

- **Quantization**: NCNN supports quantization, a technique that reduces the [precision](https://www.ultralytics.com/glossary/precision) of model weights and activations to improve performance and reduce memory footprint.

- **Compatibility**: NCNN models are compatible with popular deep learning frameworks including [TensorFlow](https://www.tensorflow.org/), [Caffe](https://caffe.berkeleyvision.org/), and [ONNX](https://onnx.ai/), allowing developers to leverage existing models and workflows.

- **Ease of Use**: NCNN provides user-friendly tools for converting models between formats, ensuring smooth interoperability across different development environments.

- **Vulkan GPU Acceleration**: NCNN supports Vulkan for GPU-accelerated inference across multiple vendors including AMD, Intel, and other non-NVIDIA GPUs, enabling high-performance deployment on a wider range of hardware.

## Deployment Options with NCNN

NCNN models are compatible with a variety of deployment platforms:

- **Mobile Deployment**: Optimized for Android and iOS, enabling seamless integration into mobile applications for efficient on-device inference.

- **Embedded Systems and IoT Devices**: Ideal for resource-constrained devices like Raspberry Pi and NVIDIA Jetson. If standard inference on a Raspberry Pi with the [Ultralytics Guide](../guides/raspberry-pi.md) is insufficient, NCNN can provide significant performance improvements.

- **Desktop and Server Deployment**: Supports deployment across Linux, Windows, and macOS for development, training, and evaluation workflows.

## Vulkan GPU Acceleration

NCNN supports GPU acceleration through Vulkan, enabling high-performance inference on a wide range of GPUs including AMD, Intel, and other non-NVIDIA graphics cards. This is particularly useful for:

- **Cross-Vendor GPU Support**: Unlike CUDA, which is limited to NVIDIA GPUs, Vulkan works across multiple GPU vendors.
- **Multi-GPU Systems**: Select a specific Vulkan device in systems with multiple GPUs using `device="vulkan:0"`, `device="vulkan:1"`, etc.
- **Edge and Desktop Deployments**: Leverage GPU acceleration on devices where CUDA is not available.

To use Vulkan acceleration, specify the Vulkan device when running inference:

!!! example "Vulkan Inference"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported NCNN model
        ncnn_model = YOLO("./yolo26n_ncnn_model")

        # Run inference with Vulkan GPU acceleration (first Vulkan device)
        results = ncnn_model("https://ultralytics.com/images/bus.jpg", device="vulkan:0")

        # Use second Vulkan device in multi-GPU systems
        results = ncnn_model("https://ultralytics.com/images/bus.jpg", device="vulkan:1")
        ```

    === "CLI"

        ```bash
        # Run inference with Vulkan GPU acceleration
        yolo predict model='./yolo26n_ncnn_model' source='https://ultralytics.com/images/bus.jpg' device=vulkan:0
        ```

!!! tip "Vulkan Requirements"

    Ensure you have Vulkan drivers installed for your GPU. Most modern GPU drivers include Vulkan support by default. You can verify Vulkan availability using tools like `vulkaninfo` on Linux or the Vulkan SDK on Windows.

## Export to NCNN: Converting Your YOLO26 Model

You can expand model compatibility and deployment flexibility by converting YOLO26 models to NCNN format.

### Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO26
        pip install ultralytics
        ```

For detailed instructions and best practices, see the [Ultralytics Installation guide](../quickstart.md). If you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions.

### Usage

All [Ultralytics YOLO26 models](../models/index.md) are designed to support export out of the box, making it easy to integrate them into your preferred deployment workflow. You can [view the full list of supported export formats and configuration options](../modes/export.md) to choose the best setup for your application.

!!! example "Usage"

    === "Python"

          ```python
          from ultralytics import YOLO

          # Load the YOLO26 model
          model = YOLO("yolo26n.pt")

          # Export the model to NCNN format
          model.export(format="ncnn")  # creates '/yolo26n_ncnn_model'

          # Load the exported NCNN model
          ncnn_model = YOLO("./yolo26n_ncnn_model")

          # Run inference
          results = ncnn_model("https://ultralytics.com/images/bus.jpg")
          ```

    === "CLI"

          ```bash
          # Export a YOLO26n PyTorch model to NCNN format
          yolo export model=yolo26n.pt format=ncnn # creates '/yolo26n_ncnn_model'

          # Run inference with the exported model
          yolo predict model='./yolo26n_ncnn_model' source='https://ultralytics.com/images/bus.jpg'
          ```

### Export Arguments

| Argument | Type             | Default  | Description                                                                                                                             |
| -------- | ---------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'ncnn'` | Target format for the exported model, defining compatibility with various deployment environments.                                      |
| `imgsz`  | `int` or `tuple` | `640`    | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.       |
| `half`   | `bool`           | `False`  | Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.            |
| `batch`  | `int`            | `1`      | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode. |
| `device` | `str`            | `None`   | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`).                         |

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLO26 NCNN Models

After exporting your Ultralytics YOLO26 models to NCNN format, you can deploy them using the `YOLO("yolo26n_ncnn_model/")` method as shown in the usage example above. For platform-specific deployment instructions, see the following resources:

- **[Android](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android)**: Build and integrate NCNN models for [object detection](https://www.ultralytics.com/glossary/object-detection) in Android applications.

- **[macOS](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-macos)**: Deploy NCNN models on macOS systems.

- **[Linux](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux)**: Deploy NCNN models on Linux devices including Raspberry Pi and similar embedded systems.

- **[Windows x64](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-windows-x64-using-visual-studio-community-2017)**: Deploy NCNN models on Windows x64 using Visual Studio.

## Summary

This guide covered exporting Ultralytics YOLO26 models to NCNN format for improved efficiency and speed on resource-constrained devices.

For additional details, refer to the [official NCNN documentation](https://ncnn.readthedocs.io/en/latest/index.html). For other export options, visit our [integration guide page](index.md).

## FAQ

### How do I export Ultralytics YOLO26 models to NCNN format?

To export your Ultralytics YOLO26 model to NCNN format:

- **Python**: Use the `export` method from the YOLO class.

    ```python
    from ultralytics import YOLO

    # Load the YOLO26 model
    model = YOLO("yolo26n.pt")

    # Export to NCNN format
    model.export(format="ncnn")  # creates '/yolo26n_ncnn_model'
    ```

- **CLI**: Use the `yolo export` command.

    ```bash
    yolo export model=yolo26n.pt format=ncnn # creates '/yolo26n_ncnn_model'
    ```

For detailed export options, see the [Export](../modes/export.md) documentation.

### What are the advantages of exporting YOLO26 models to NCNN?

Exporting your Ultralytics YOLO26 models to NCNN offers several benefits:

- **Efficiency**: NCNN models are optimized for mobile and embedded devices, ensuring high performance even with limited computational resources.
- **Quantization**: NCNN supports techniques like quantization that improve model speed and reduce memory usage.
- **Broad Compatibility**: You can deploy NCNN models on multiple platforms, including Android, iOS, Linux, and macOS.
- **Vulkan GPU Acceleration**: Leverage GPU acceleration on AMD, Intel, and other non-NVIDIA GPUs via Vulkan for faster inference.

For more details, see the [Why Export to NCNN?](#why-export-to-ncnn) section.

### Why should I use NCNN for my mobile AI applications?

NCNN, developed by Tencent, is specifically optimized for mobile platforms. Key reasons to use NCNN include:

- **High Performance**: Designed for efficient and fast processing on mobile CPUs.
- **Cross-Platform**: Compatible with popular frameworks such as [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) and ONNX, making it easier to convert and deploy models across different platforms.
- **Community Support**: Active community support ensures continual improvements and updates.

For more information, see the [Key Features of NCNN Models](#key-features-of-ncnn-models) section.

### What platforms are supported for NCNN [model deployment](https://www.ultralytics.com/glossary/model-deployment)?

NCNN is versatile and supports various platforms:

- **Mobile**: Android, iOS.
- **Embedded Systems and IoT Devices**: Devices like Raspberry Pi and NVIDIA Jetson.
- **Desktop and Servers**: Linux, Windows, and macOS.

For improved performance on Raspberry Pi, consider using NCNN format as detailed in our [Raspberry Pi Guide](../guides/raspberry-pi.md).

### How can I deploy Ultralytics YOLO26 NCNN models on Android?

To deploy your YOLO26 models on Android:

1. **Build for Android**: Follow the [NCNN Build for Android](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android) guide.
2. **Integrate with Your App**: Use the NCNN Android SDK to integrate the exported model into your application for efficient on-device inference.

For detailed instructions, see [Deploying Exported YOLO26 NCNN Models](#deploying-exported-yolo26-ncnn-models).

For more advanced guides and use cases, visit the [Ultralytics deployment guide](../guides/model-deployment-options.md).

### How do I use Vulkan GPU acceleration with NCNN models?

NCNN supports Vulkan for GPU acceleration on AMD, Intel, and other non-NVIDIA GPUs. To use Vulkan:

```python
from ultralytics import YOLO

# Load NCNN model and run with Vulkan GPU
model = YOLO("yolo26n_ncnn_model")
results = model("image.jpg", device="vulkan:0")  # Use first Vulkan device
```

For multi-GPU systems, specify the device index (e.g., `vulkan:1` for the second GPU). Ensure Vulkan drivers are installed for your GPU. See the [Vulkan GPU Acceleration](#vulkan-gpu-acceleration) section for more details.
