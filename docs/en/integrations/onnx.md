---
comments: true
description: Learn how to export YOLO26 models to ONNX format for flexible deployment across various platforms with enhanced performance.
keywords: YOLO26, ONNX, model export, Ultralytics, ONNX Runtime, machine learning, model deployment, computer vision, deep learning
---

# ONNX Export for YOLO26 Models

???+ tip "~43% faster inference."

    - Exporting the Ultralytics YOLO26 model to ONNX can deliver up to a 43% boost in inference speed, enabling faster and more efficient deployment.

Often, when deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models, you'll need a model format that's both flexible and compatible with multiple platforms.

Exporting [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) models to ONNX format streamlines deployment and ensures optimal performance across various environments. This guide will show you how to easily convert your YOLO26 models to ONNX and enhance their scalability and effectiveness in real-world applications.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/cxU5E2SkivU"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics YOLO26 vs Ultralytics YOLO11 ONNX Inference Test | ~43% Faster Inference with YOLO26 🚀
</p>

## ONNX and ONNX Runtime

[ONNX](https://onnx.ai/), which stands for Open [Neural Network](https://www.ultralytics.com/glossary/neural-network-nn) Exchange, is a community project that Facebook and Microsoft initially developed. The ongoing development of ONNX is a collaborative effort supported by various organizations like IBM, Amazon (through AWS), and Google. The project aims to create an open file format designed to represent [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models in a way that allows them to be used across different AI frameworks and hardware.

ONNX models can be used to transition between different frameworks seamlessly. For instance, a [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) model trained in PyTorch can be exported to ONNX format and then easily imported into TensorFlow.

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/onnx-model-portability.avif" alt="ONNX model portability across deep learning frameworks">
</p>

Alternatively, ONNX models can be used with ONNX Runtime. [ONNX Runtime](https://onnxruntime.ai/) is a versatile cross-platform accelerator for machine learning models that is compatible with frameworks like PyTorch, [TensorFlow](https://www.ultralytics.com/glossary/tensorflow), scikit-learn, etc.

ONNX Runtime optimizes the execution of ONNX models by leveraging hardware-specific capabilities. This optimization allows the models to run efficiently and with high performance on various hardware platforms, including CPUs, GPUs, and specialized accelerators.

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/onnx-and-onnx-runtime.avif" alt="ONNX Runtime cross-platform inference acceleration">
</p>

Whether used independently or in tandem with ONNX Runtime, ONNX provides a flexible solution for machine learning [model deployment](https://www.ultralytics.com/glossary/model-deployment) and compatibility.

## Key Features of ONNX Models

The ability of ONNX to handle various formats can be attributed to the following key features:

- **Common Model Representation**: ONNX defines a common set of operators (like convolutions, layers, etc.) and a standard data format. When a model is converted to ONNX format, its architecture and weights are translated into this common representation. This uniformity ensures that the model can be understood by any framework that supports ONNX.

- **Versioning and Backward Compatibility**: ONNX maintains a versioning system for its operators. This ensures that even as the standard evolves, models created in older versions remain usable. Backward compatibility is a crucial feature that prevents models from becoming obsolete quickly.

- **Graph-based Model Representation**: ONNX represents models as computational graphs. This graph-based structure is a universal way of representing machine learning models, where nodes represent operations or computations, and edges represent the tensors flowing between them. This format is easily adaptable to various frameworks which also represent models as graphs.

- **Tools and Ecosystem**: There is a rich ecosystem of tools around ONNX that assist in model conversion, visualization, and optimization. These tools make it easier for developers to work with ONNX models and to convert models between different frameworks seamlessly.

## Common Usage of ONNX

Before we jump into how to export YOLO26 models to the ONNX format, let's take a look at where ONNX models are usually used.

### CPU Deployment

ONNX models are often deployed on CPUs due to their compatibility with ONNX Runtime. This runtime is optimized for CPU execution. It significantly improves inference speed and makes real-time CPU deployments feasible.

### Supported Deployment Options

While ONNX models are commonly used on CPUs, they can also be deployed on the following platforms:

- **GPU Acceleration (NVIDIA)**: ONNX fully supports GPU acceleration via NVIDIA CUDA. This enables efficient execution on NVIDIA GPUs for tasks that demand high computational power.

- **GPU Acceleration (AMD)**: ONNX supports GPU acceleration via AMD ROCm and the MIGraphX Execution Provider. This enables efficient execution on AMD Instinct and Radeon GPUs for high-performance inference.

- **Edge and Mobile Devices**: ONNX extends to edge and mobile devices, perfect for on-device and real-time inference scenarios. It's lightweight and compatible with edge hardware, and serves as the basis for vendor NPU formats such as [Qualcomm QNN](qnn.md) for Snapdragon devices and [RKNN](rockchip-rknn.md) for Rockchip NPUs.

- **Web Browsers**: ONNX can run directly in web browsers, powering interactive and dynamic web-based AI applications.

## Exporting YOLO26 Models to ONNX

You can expand model compatibility and deployment flexibility by converting YOLO26 models to ONNX format. [Ultralytics YOLO26](../models/yolo26.md) provides a straightforward export process that can significantly enhance your model's performance across different platforms.

### Installation

To install the required package, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO26
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [YOLO26 Installation guide](../quickstart.md). While installing the required packages for YOLO26, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

Before diving into the usage instructions, be sure to check out the range of [YOLO26 models offered by Ultralytics](../models/index.md). This will help you choose the most appropriate model for your project requirements.

The ONNX format supports the [Export](../modes/export.md), [Predict](../modes/predict.md), and [Validate](../modes/val.md) modes. Export your model, then load the exported model to run inference or validate its accuracy.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to ONNX format
        model.export(format="onnx")  # creates 'yolo26n.onnx'

        # Export an INT8-quantized ONNX model with calibration data
        model.export(format="onnx", quantize=8, data="coco8.yaml")  # creates 'yolo26n_int8.onnx'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to ONNX format
        yolo export model=yolo26n.pt format=onnx # creates 'yolo26n.onnx'

        # Export an INT8-quantized ONNX model with calibration data
        yolo export model=yolo26n.pt format=onnx quantize=8 data=coco8.yaml # creates 'yolo26n_int8.onnx'
        ```

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported ONNX model
        model = YOLO("yolo26n.onnx")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported ONNX model
        yolo predict model=yolo26n.onnx source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported ONNX model
        model = YOLO("yolo26n.onnx")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported ONNX model
        yolo val model=yolo26n.onnx data=coco8.yaml
        ```

### Export Arguments

When exporting your YOLO26 model to ONNX format, you can customize the process using various arguments to optimize for your specific deployment needs:

| Argument   | Type             | Default  | Description                                                                                                                                                                                                                     |
| ---------- | ---------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'onnx'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                              |
| `imgsz`    | `int` or `tuple` | `640`    | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                                                               |
| `quantize` | `int` or `str`   | `None`   | Quantization precision: `16` (FP16) or `8` (INT8 static quantization with ONNX Runtime using calibration images from `data`, producing an `_int8.onnx` model); `32`/unset is FP32. Replaces the deprecated `half`/`int8` flags. |
| `data`     | `str`            | `None`   | Dataset YAML used for INT8 calibration. If omitted with `quantize=8`, Ultralytics selects the default calibration dataset for the model task.                                                                                   |
| `fraction` | `float`          | `1.0`    | Fraction of calibration images to use for INT8 quantization.                                                                                                                                                                    |
| `dynamic`  | `bool`           | `False`  | Allows dynamic input sizes, enhancing flexibility in handling varying image dimensions.                                                                                                                                         |
| `simplify` | `bool`           | `True`   | Simplifies the model graph with `onnxslim`, potentially improving performance and compatibility.                                                                                                                                |
| `opset`    | `int`            | `None`   | Specifies the ONNX opset version for compatibility with different ONNX parsers and runtimes. If not set, uses the latest supported version.                                                                                     |
| `nms`      | `bool`           | `False`  | Adds Non-Maximum Suppression (NMS), essential for accurate and efficient detection post-processing.                                                                                                                             |
| `batch`    | `int`            | `1`      | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                                                         |
| `device`   | `str`            | `None`   | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`).                                                                                                                 |

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLO26 ONNX Models

Once you've successfully exported your Ultralytics YOLO26 models to ONNX format, the next step is deploying these models in various environments. For detailed instructions on deploying your ONNX models, take a look at the following resources:

- **[ONNX Runtime Python API Documentation](https://onnxruntime.ai/docs/api/python/api_summary.html)**: This guide provides essential information for loading and running ONNX models using ONNX Runtime.

- **[Deploying on Edge Devices](https://onnxruntime.ai/docs/tutorials/iot-edge/)**: Check out this docs page for different examples of deploying ONNX models on edge.

- **[ONNX Tutorials on GitHub](https://github.com/onnx/tutorials)**: A collection of comprehensive tutorials that cover various aspects of using and implementing ONNX models in different scenarios.

- **[Triton Inference Server](../guides/triton-inference-server.md)**: Learn how to deploy your ONNX models with NVIDIA's Triton Inference Server for high-performance, scalable deployments.

## AMD GPU Inference with MIGraphX

Ultralytics supports ONNX Runtime inference on AMD GPUs via the [MIGraphX Execution Provider](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html). When PyTorch is built with ROCm (HIP) and `onnxruntime-migraphx` is available, the ONNX backend selects MIGraphX for GPU-accelerated inference.

### Prerequisites

- AMD GPU with ROCm support (for example Instinct MI300/MI350 or Radeon AI PRO R9700)
- ROCm 7.2+ ([installation guide](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html))
- MIGraphX C++ library installed (required at runtime)
- PyTorch ROCm build (`python -c "import torch; print(torch.version.hip)"`)
- Linux x86_64
- Python 3.10 or 3.12 (current ROCm 7.2 `onnxruntime-migraphx` wheels)

If the `migraphx` library is not yet installed on your system, you can add the ROCm repository and install it via `apt` (Ubuntu/Debian example):

```bash
# Add ROCm 7.2.x repository
sudo apt update && sudo apt install -y wget gnupg2 lsb-release
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo tee /etc/apt/keyrings/rocm.asc > /dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.asc] https://repo.radeon.com/rocm/apt/7.2 $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/rocm.list

# Install MIGraphX
sudo apt update
sudo apt install -y migraphx

# Add ROCm libraries to the linker path (required for MIGraphX EP to load at runtime)
echo "/opt/rocm/lib" | sudo tee /etc/ld.so.conf.d/rocm.conf
sudo ldconfig
```

### Installation

!!! tip "Installation"

    ```bash
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
    pip install ultralytics
    pip install onnxruntime-migraphx --extra-index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
    ```

!!! note "Supported ROCm version"

    The command targets `rocm-rel-7.2`, the ROCm/MIGraphX stack validated by Ultralytics `Dockerfile-amd`. Patch releases `rocm-rel-7.2.x` ship the same wheel version and are compatible. For other ROCm minors, AMD publishes matching wheel folders under [`repo.radeon.com/rocm/manylinux/`](https://repo.radeon.com/rocm/manylinux/).

!!! warning "Package Conflict"

    `onnxruntime-migraphx`, `onnxruntime-gpu`, and `onnxruntime` provide the same `onnxruntime` Python module. Keep only one installed:

    ```bash
    pip uninstall onnxruntime onnxruntime-gpu onnxruntime-migraphx -y
    pip install onnxruntime-migraphx --extra-index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
    ```

    If MIGraphX disappears after export or other operations, check for a conflicting install (`pip list | grep onnxruntime`), uninstall all variants, then reinstall `onnxruntime-migraphx`.

### Usage

No code changes are needed. On ROCm systems, the ONNX backend detects HIP and selects `MIGraphXExecutionProvider`.

!!! note

    If `onnx` and/or `onnxruntime-migraphx` are missing, Ultralytics installs them automatically the first time you run ONNX export or ONNX inference.

!!! example "AMD GPU Inference"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load an ONNX model
        model = YOLO("yolo26n.onnx")

        # Run inference on AMD GPU - MIGraphX EP is selected automatically
        results = model.predict("https://ultralytics.com/images/bus.jpg", device=0)
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo26n.onnx source='https://ultralytics.com/images/bus.jpg' device=0
        ```

Expected log output:

```
Using ONNX Runtime X.Y.Z with MIGraphXExecutionProvider
```

!!! warning "MIGraphX EP: Unsupported Tasks (ROCm 7.2)"

    **Segmentation** and **pose** models are not supported by MIGraphX EP in ROCm 7.2. Use `device="cpu"` or PyTorch `.pt` inference for these tasks. Detection, classification, and OBB work fully.

## Summary

In this guide, you've learned how to export Ultralytics YOLO26 models to ONNX format to increase their interoperability and performance across various platforms. You were also introduced to the ONNX Runtime and ONNX deployment options.

ONNX export is just one of many [export formats](../modes/export.md) supported by Ultralytics YOLO26, allowing you to deploy your models in virtually any environment. Depending on your specific needs, you might also want to explore other export options like [TensorRT](../integrations/tensorrt.md) for maximum GPU performance or [CoreML](../integrations/coreml.md) for Apple devices.

For further details on usage, visit the [ONNX official documentation](https://onnx.ai/onnx/intro/).

Also, if you'd like to know more about other Ultralytics YOLO26 integrations, visit our [integration guide page](../integrations/index.md). You'll find plenty of useful resources and insights there.

## FAQ

### How do I export YOLO26 models to ONNX format using Ultralytics?

To export your YOLO26 models to ONNX format using Ultralytics, follow these steps:

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to ONNX format
        model.export(format="onnx")  # creates 'yolo26n.onnx'

        # Load the exported ONNX model
        onnx_model = YOLO("yolo26n.onnx")

        # Run inference
        results = onnx_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to ONNX format
        yolo export model=yolo26n.pt format=onnx # creates 'yolo26n.onnx'

        # Run inference with the exported model
        yolo predict model=yolo26n.onnx source='https://ultralytics.com/images/bus.jpg'
        ```

For more details, visit the [export documentation](../modes/export.md).

### What are the advantages of using ONNX Runtime for deploying YOLO26 models?

Using ONNX Runtime for deploying YOLO26 models offers several advantages:

- **Cross-platform compatibility**: ONNX Runtime supports various platforms, such as Windows, macOS, and Linux, ensuring your models run smoothly across different environments.
- **Hardware acceleration**: ONNX Runtime can leverage hardware-specific optimizations for CPUs, GPUs, and dedicated accelerators, providing high-performance inference.
- **Framework interoperability**: Models trained in popular frameworks like [PyTorch](https://www.ultralytics.com/glossary/pytorch) or TensorFlow can be easily converted to ONNX format and run using ONNX Runtime.
- **Performance optimization**: ONNX Runtime can provide up to 3x CPU speedup compared to native PyTorch models, making it ideal for deployment scenarios where GPU resources are limited.

Learn more by checking the [ONNX Runtime documentation](https://onnxruntime.ai/docs/api/python/api_summary.html).

### What deployment options are available for YOLO26 models exported to ONNX?

YOLO26 models exported to ONNX can be deployed on various platforms including:

- **CPUs**: Utilizing ONNX Runtime for optimized CPU inference.
- **NVIDIA GPUs**: Leveraging NVIDIA CUDA for high-performance GPU acceleration.
- **AMD GPUs**: Using AMD ROCm and MIGraphX Execution Provider for high-performance GPU acceleration on Linux.
- **Edge devices**: Running lightweight models on edge and mobile devices for real-time, on-device inference.
- **Web browsers**: Executing models directly within web browsers for interactive web-based applications.
- **Cloud services**: Deploying on cloud platforms that support ONNX format for scalable inference.

For more information, explore our guide on [model deployment options](../guides/model-deployment-options.md).

### Why should I use ONNX format for Ultralytics YOLO26 models?

Using ONNX format for Ultralytics YOLO26 models provides numerous benefits:

- **Interoperability**: ONNX allows models to be transferred between different machine learning frameworks seamlessly.
- **Performance Optimization**: ONNX Runtime can enhance model performance by utilizing hardware-specific optimizations.
- **Flexibility**: ONNX supports various deployment environments, enabling you to use the same model on different platforms without modification.
- **Standardization**: ONNX provides a standardized format that is widely supported across the industry, ensuring long-term compatibility.

Refer to the comprehensive guide on [exporting YOLO26 models to ONNX](../integrations/onnx.md).

### How can I troubleshoot issues when exporting YOLO26 models to ONNX?

When exporting YOLO26 models to ONNX, you might encounter common issues such as mismatched dependencies or unsupported operations. To troubleshoot these problems:

1. Verify that you have the correct version of required dependencies installed.
2. Check the official [ONNX documentation](https://onnx.ai/onnx/intro/) for supported operators and features.
3. Review the error messages for clues and consult the [Ultralytics Common Issues guide](../guides/yolo-common-issues.md).
4. Try using different export arguments like `simplify=True` or adjusting the `opset` version.
5. For dynamic input size issues, set `dynamic=True` during export.

If issues persist, contact Ultralytics support for further assistance.
