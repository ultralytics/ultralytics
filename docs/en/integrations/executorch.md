---
comments: true
description: Export YOLO11 models to ExecuTorch format for efficient on-device inference on mobile and edge devices. Optimize your AI models for iOS, Android, and embedded systems.
keywords: Ultralytics, YOLO11, ExecuTorch, model export, PyTorch, edge AI, mobile deployment, on-device inference, XNNPACK, embedded systems
---

# How to Export to ExecuTorch from YOLO11 for Edge Deployment

Deploying computer vision models on edge devices like smartphones, tablets, and embedded systems requires an optimized runtime that balances performance with resource constraints. ExecuTorch, PyTorch's solution for edge computing, enables efficient on-device inference for [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models.

This guide shows you how to export YOLO11 models to ExecuTorch format, enabling you to deploy your models on mobile and edge devices with optimized performance.

## Why Export to ExecuTorch?

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/pytorch-ecosystem.avif" alt="PyTorch ExecuTorch overview">
</p>

[ExecuTorch](https://pytorch.org/executorch/) is PyTorch's end-to-end solution for enabling on-device inference capabilities across mobile and edge devices. Built with the goal of being portable and efficient, ExecuTorch can be used to run PyTorch programs on a wide variety of computing platforms.

Key advantages of ExecuTorch:

- **PyTorch Native**: Direct integration with the PyTorch ecosystem, leveraging existing models and workflows
- **Optimized Performance**: Hardware-specific optimizations through backends like XNNPACK for mobile CPUs
- **Cross-Platform**: Write once, deploy anywhere - iOS, Android, embedded Linux, and more
- **Ahead-of-Time Compilation**: Models are compiled ahead of time, reducing startup latency and memory footprint
- **Flexible Delegation**: Support for hardware acceleration through various backends (CPU, GPU, NPU)

## Key Features of ExecuTorch Models

ExecuTorch provides several powerful features for deploying YOLO11 models on edge devices:

- **Portable Model Format**: ExecuTorch uses the `.pte` (PyTorch ExecuTorch) format, which is optimized for size and loading speed on resource-constrained devices.

- **XNNPACK Backend**: Default integration with XNNPACK provides highly optimized inference on mobile CPUs, delivering excellent performance without requiring specialized hardware.

- **Quantization Support**: Built-in support for quantization techniques to reduce model size and improve inference speed while maintaining accuracy.

- **Memory Efficiency**: Optimized memory management reduces runtime memory footprint, making it suitable for devices with limited RAM.

- **Model Metadata**: Exported models include metadata (image size, class names, etc.) in a separate YAML file for easy integration.

## Deployment Options with ExecuTorch

ExecuTorch models can be deployed across various edge and mobile platforms:

- **Mobile Applications**: Deploy on iOS and Android applications with native performance, enabling real-time object detection in mobile apps.

- **Embedded Systems**: Run on embedded Linux devices like Raspberry Pi, NVIDIA Jetson, and other ARM-based systems with optimized performance.

- **Edge AI Devices**: Deploy on specialized edge AI hardware with custom delegates for accelerated inference.

- **IoT Devices**: Integrate into IoT devices for on-device inference without cloud connectivity requirements.

## Export to ExecuTorch: Converting Your YOLO11 Model

Converting YOLO11 models to ExecuTorch format enables efficient deployment on mobile and edge devices.

### Requirements

ExecuTorch export requires Python 3.10 or higher and specific dependencies:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install Ultralytics package
        pip install ultralytics
        
        # Install ExecuTorch (includes dependencies)
        pip install executorch
        ```

For detailed installation instructions, check our [Ultralytics Installation guide](../quickstart.md). If you encounter any issues, consult our [Common Issues guide](../guides/yolo-common-issues.md).

### Usage

Exporting YOLO11 models to ExecuTorch is straightforward:

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO11 model
        model = YOLO("yolo11n.pt")

        # Export the model to ExecuTorch format
        model.export(format="executorch")  # creates 'yolo11n_executorch_model' directory
        
        # The exported directory contains:
        # - yolo11n.pte (model file)
        # - metadata.yaml (model metadata)
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to ExecuTorch format
        yolo export model=yolo11n.pt format=executorch  # creates 'yolo11n_executorch_model' directory
        ```

### Export Arguments

When exporting to ExecuTorch format, you can specify the following arguments:

| Argument   | Type           | Default | Description                                                      |
|------------|----------------|---------|------------------------------------------------------------------|
| `imgsz`    | `int` or `list`| `640`   | Image size for model input (height, width)                       |
| `device`   | `str`          | `'cpu'` | Device to use for export (`'cpu'`)                               |

!!! example "Export with Custom Arguments"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        # Export with custom image size
        model.export(format="executorch", imgsz=320)
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=executorch imgsz=320
        ```

### Output Structure

The ExecuTorch export creates a directory containing the model and metadata:

```text
yolo11n_executorch_model/
├── yolo11n.pte              # ExecuTorch model file
└── metadata.yaml            # Model metadata (classes, image size, etc.)
```

## Using Exported ExecuTorch Models

After exporting your model, you'll need to integrate it into your target application using the ExecuTorch runtime.

### Mobile Integration

For mobile applications (iOS/Android), you'll need to:

1. **Add ExecuTorch Runtime**: Include the ExecuTorch runtime library in your mobile project
2. **Load Model**: Load the `.pte` file in your application
3. **Run Inference**: Process images and get predictions

Example iOS integration (Objective-C/C++):

```objc
// iOS uses C++ APIs for model loading and inference
// See https://pytorch.org/executorch/stable/using-executorch-ios.html for complete examples

#include <executorch/extension/module/module.h>

using namespace ::executorch::extension;

// Load the model
Module module("/path/to/yolo11n.pte");

// Create input tensor
float input[1 * 3 * 640 * 640];
auto tensor = from_blob(input, {1, 3, 640, 640});

// Run inference
const auto result = module.forward(tensor);
```

Example Android integration (Kotlin):

```kotlin
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

// Load the model
val module = Module.load("/path/to/yolo11n.pte")

// Prepare input tensor
val inputTensor = Tensor.fromBlob(floatData, longArrayOf(1, 3, 640, 640))
val inputEValue = EValue.from(inputTensor)

// Run inference
val outputs = module.forward(inputEValue)
val scores = outputs[0].toTensor().dataAsFloatArray
```

### Embedded Linux

For embedded Linux systems, use the ExecuTorch C++ API:

```cpp
#include <executorch/extension/module/module.h>

// Load model
auto module = torch::executor::Module("yolo11n.pte");

// Prepare input
std::vector<float> input_data = preprocessImage(image);
auto input_tensor = torch::executor::Tensor(input_data, {1, 3, 640, 640});

// Run inference  
auto outputs = module.forward({input_tensor});
```

For more details on integrating ExecuTorch into your applications, visit the [ExecuTorch Documentation](https://pytorch.org/executorch/).

## Performance Optimization

### Model Size Optimization

To reduce model size for deployment:

- **Use Smaller Models**: Start with YOLO11n (nano) for the smallest footprint
- **Lower Input Resolution**: Use smaller image sizes (e.g., `imgsz=320` or `imgsz=416`)
- **Quantization**: Apply quantization techniques (supported in future ExecuTorch versions)

### Inference Speed Optimization

For faster inference:

- **XNNPACK Backend**: The default XNNPACK backend provides optimized CPU inference
- **Hardware Acceleration**: Use platform-specific delegates (e.g., CoreML for iOS)
- **Batch Processing**: Process multiple images when possible

## Comparing ExecuTorch to Other Formats

When choosing an export format, consider the following comparison:

| Format       | Model Size | CPU Speed | Mobile Support | Hardware Accel | Ease of Use |
|--------------|------------|-----------|----------------|----------------|-------------|
| PyTorch      | Large      | Baseline  | ❌             | ✅             | ⭐⭐⭐⭐⭐   |
| TorchScript  | Large      | Good      | ✅             | Limited        | ⭐⭐⭐⭐    |
| ONNX         | Medium     | Fast      | ✅             | ✅             | ⭐⭐⭐⭐    |
| CoreML       | Small      | Fast      | iOS Only       | ✅             | ⭐⭐⭐      |
| TFLite       | Small      | Fast      | ✅             | ✅             | ⭐⭐⭐      |
| **ExecuTorch** | **Small** | **Fast**  | ✅             | ✅             | ⭐⭐⭐⭐    |

ExecuTorch offers an excellent balance of model size, performance, and ease of use, especially for applications already using PyTorch.

## Troubleshooting

### Common Issues

**Issue**: `Python version error`

**Solution**: ExecuTorch requires Python 3.10 or higher. Upgrade your Python installation:

    ```bash
    # Using conda
    conda create -n executorch python=3.10
    conda activate executorch
    ```

**Issue**: `Export fails during first run`

**Solution**: ExecuTorch may need to download and compile components on first use. Ensure you have:

    ```bash
    pip install --upgrade executorch
    ```

**Issue**: `Import errors for ExecuTorch modules`

**Solution**: Ensure ExecuTorch is properly installed:

    ```bash
    pip install executorch --force-reinstall
    ```

For more troubleshooting help, visit the [Ultralytics GitHub Issues](https://github.com/ultralytics/ultralytics/issues) or the [ExecuTorch Documentation](https://pytorch.org/executorch/stable/getting-started-setup.html).

## Summary

Exporting YOLO11 models to ExecuTorch format enables efficient deployment on mobile and edge devices. With PyTorch-native integration, cross-platform support, and optimized performance, ExecuTorch is an excellent choice for edge AI applications.

Key takeaways:

- ExecuTorch provides PyTorch-native edge deployment with excellent performance
- Export is simple with `format='executorch'` parameter
- Models are optimized for mobile CPUs via XNNPACK backend
- Supports iOS, Android, and embedded Linux platforms
- Requires Python 3.10+ and FlatBuffers compiler

## FAQ

### How do I export a YOLO11 model to ExecuTorch format?

Export a YOLO11 model to ExecuTorch using either Python or CLI:

    ```python
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.export(format="executorch")
    ```

or

    ```bash
    yolo export model=yolo11n.pt format=executorch
    ```

### What are the system requirements for ExecuTorch export?

ExecuTorch export requires:

- Python 3.10 or higher
- `executorch` package (install via `pip install executorch`)
- PyTorch (installed automatically with ultralytics)

Note: During the first export, ExecuTorch will download and compile necessary components including the FlatBuffers compiler automatically.

### Can I run inference with ExecuTorch models directly in Python?

ExecuTorch models (`.pte` files) are designed for deployment on mobile and edge devices using the ExecuTorch runtime. They cannot be directly loaded with `YOLO()` for inference in Python. You need to integrate them into your target application using the ExecuTorch runtime libraries.

### What platforms are supported by ExecuTorch?

ExecuTorch supports:

- **Mobile**: iOS and Android
- **Embedded Linux**: Raspberry Pi, NVIDIA Jetson, and other ARM devices
- **Desktop**: Linux, macOS, and Windows (for development)

### How does ExecuTorch compare to TFLite for mobile deployment?

Both ExecuTorch and TFLite are excellent for mobile deployment:

- **ExecuTorch**: Better PyTorch integration, native PyTorch workflow, growing ecosystem
- **TFLite**: More mature, wider hardware support, more deployment examples

Choose ExecuTorch if you're already using PyTorch and want a native deployment path. Choose TFLite for maximum compatibility and mature tooling.

### Can I use ExecuTorch models with GPU acceleration?

Yes! ExecuTorch supports hardware acceleration through various backends:

- **Mobile GPU**: Via Vulkan, Metal, or OpenCL delegates
- **NPU/DSP**: Via platform-specific delegates
- **Default**: XNNPACK for optimized CPU inference

Refer to the [ExecuTorch Documentation](https://pytorch.org/executorch/stable/native-delegates-executorch.html) for backend-specific setup.
