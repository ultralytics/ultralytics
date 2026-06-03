---
comments: true
description: Learn how to export YOLO26 models to RKNN format, including floating-point and INT8 quantized models, for efficient deployment on Rockchip platforms.
keywords: YOLO26, RKNN, model export, Ultralytics, Rockchip, INT8 quantization, FP16, machine learning, model deployment, computer vision, deep learning, edge AI, NPU, embedded devices
---

# Rockchip RKNN Export for Ultralytics YOLO26 Models

When deploying computer vision models on embedded devices, especially those powered by Rockchip processors, having a compatible model format is essential. Exporting [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) models to RKNN format ensures optimized performance and compatibility with Rockchip's hardware. This guide will walk you through converting your YOLO26 models to RKNN format, including floating-point and INT8 quantized exports, enabling efficient deployment on Rockchip platforms.

<p align="center">
  <img width="50%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/rockchip-rknn-overview.avif" alt="Rockchip RKNN export for NPU deployment">
</p>

!!! note

    This guide has been tested with [Radxa Rock 5B](https://radxa.com/products/rock5/5b/) which is based on Rockchip RK3588 and [Radxa Zero 3W](https://radxa.com/products/zeros/zero3w/) which is based on Rockchip RK3566. It is expected to work across other Rockchip-based devices that support [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) such as RK3576, RK3568, RK3562, RK2118, RV1126B, RV1103, RV1106, RV1103B and RV1106B. INT8-only targets such as RV1103 and RV1106 require `int8=True`.

## What is Rockchip?

Renowned for delivering versatile and power-efficient solutions, Rockchip designs advanced System-on-Chips (SoCs) that power a wide range of consumer electronics, industrial applications, and AI technologies. With ARM-based architecture, built-in Neural Processing Units (NPUs), and high-resolution multimedia support, Rockchip SoCs enable cutting-edge performance for devices like tablets, smart TVs, IoT systems, and [edge AI applications](https://www.ultralytics.com/blog/understanding-the-real-world-applications-of-edge-ai). Companies like Radxa, ASUS, Pine64, Orange Pi, Odroid, Khadas, and Banana Pi offer a variety of products based on Rockchip SoCs, further extending their reach and impact across diverse markets.

## RKNN Toolkit

The [RKNN Toolkit](https://github.com/airockchip/rknn-toolkit2) is a set of tools and libraries provided by Rockchip to facilitate the deployment of deep learning models on their hardware platforms. RKNN, or Rockchip Neural Network, is the proprietary format used by these tools. RKNN models are designed to take full advantage of the hardware acceleration provided by Rockchip's NPU (Neural Processing Unit), ensuring high performance in AI tasks on devices like RK3588, RK3566, RV1103, RV1106, and other Rockchip-powered systems.

## Key Features of RKNN Models

RKNN models offer several advantages for deployment on Rockchip platforms:

- **Optimized for NPU**: RKNN models are specifically optimized to run on Rockchip's NPUs, ensuring maximum performance and efficiency.
- **Low Latency**: The RKNN format minimizes inference latency, which is critical for real-time applications on edge devices.
- **Platform-Specific Customization**: RKNN models can be tailored to specific Rockchip platforms, enabling better utilization of hardware resources.
- **Power Efficiency**: By leveraging dedicated NPU hardware, RKNN models consume less power than CPU or GPU-based processing, extending battery life for portable devices.

## Flash OS to Rockchip hardware

The first step after getting your hands on a Rockchip-based device is to flash an OS so that the hardware can boot into a working environment. In this guide we will point to getting started guides of the two devices that we tested which are Radxa Rock 5B and Radxa Zero 3W.

- [Radxa Rock 5B Getting Started Guide](https://docs.radxa.com/en/rock5/rock5b)
- [Radxa Zero 3W Getting Started Guide](https://docs.radxa.com/en/zero/zero3)

## Export to RKNN: Converting Your YOLO26 Model

Export an Ultralytics YOLO26 model to RKNN format and run inference with the exported model.

!!! note

    Make sure to use an X86-based Linux PC to export the model to RKNN because exporting on Rockchip-based devices (ARM64) is not supported.

### Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO26
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLO26, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

!!! note

    Export is currently only supported for detection models. More model support will be coming in the future.

The RKNN format supports the [Export](../modes/export.md), [Predict](../modes/predict.md), and [Validate](../modes/val.md) modes. Inference and validation run on Rockchip NPU hardware. Export your model, then load the exported model to run inference or validate its accuracy. By default, RKNN export uses the existing floating-point build path with `half=True` for FP16-capable Rockchip targets. Use `int8=True` to build an INT8-quantized RKNN model with calibration data. RKNN export does not expose a separate FP32 mode; leaving `int8=False` does not request FP32.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to RKNN format
        model.export(format="rknn", name="rk3588")  # creates '/yolo26n_rknn_model'

        # Export an INT8-quantized RKNN model with calibration data
        model.export(format="rknn", name="rk3588", int8=True, data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to RKNN format
        yolo export model=yolo26n.pt format=rknn name=rk3588 # creates '/yolo26n_rknn_model'

        # Export an INT8-quantized RKNN model with calibration data
        yolo export model=yolo26n.pt format=rknn name=rk3588 int8=True data=coco8.yaml
        ```

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported RKNN model
        model = YOLO("./yolo26n_rknn_model")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported RKNN model
        yolo predict model=./yolo26n_rknn_model source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported RKNN model
        model = YOLO("./yolo26n_rknn_model")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported RKNN model
        yolo val model=./yolo26n_rknn_model data=coco8.yaml
        ```

### Export Arguments

| Argument   | Type             | Default    | Description                                                                                                                                                                       |
| ---------- | ---------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'rknn'`   | Target format for the exported model, defining compatibility with Rockchip deployment environments.                                                                               |
| `imgsz`    | `int` or `tuple` | `640`      | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                 |
| `batch`    | `int`            | `1`        | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                           |
| `name`     | `str`            | `'rk3588'` | Specifies the Rockchip target, such as rk3588, rk3576, rk3566, rk3568, rk3562, rk2118, rv1126b, rv1103, rv1106, rv1103b or rv1106b.                                               |
| `half`     | `bool`           | `True`     | Enables the default floating-point RKNN export path for FP16-capable targets. Mutually exclusive with `int8=True`.                                                                |
| `int8`     | `bool`           | `False`    | Enables INT8 quantization. Required for INT8-only targets such as RV1103 and RV1106. When `False`, RKNN Toolkit builds a floating-point model for FP16-capable targets, not FP32. |
| `data`     | `str`            | `None`     | Dataset YAML used for INT8 calibration. If omitted with `int8=True`, Ultralytics selects the default calibration dataset for the model task.                                      |
| `fraction` | `float`          | `1.0`      | Fraction of calibration images to use for INT8 quantization.                                                                                                                      |
| `device`   | `str`            | `None`     | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`).                                                                                                         |

!!! tip

    Please make sure to use an x86 Linux machine when exporting to RKNN.

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLO26 RKNN Models

Once you've successfully exported your Ultralytics YOLO26 models to RKNN format, the next step is deploying these models on Rockchip-based devices.

### Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO26
        pip install ultralytics
        ```

Once installed, run inference and validation on your Rockchip device exactly as shown in the [Usage](#usage) section above — the exported `_rknn_model` loads directly with `YOLO(...)`.

!!! note

    If you encounter a log message indicating that the RKNN runtime version does not match the RKNN Toolkit version and the inference fails, please replace `/usr/lib/librknnrt.so` with official [librknnrt.so file](https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so).

    ![RKNN export screenshot](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/rockchip-rknn-export-log.avif)

## Real-World Applications

Rockchip-powered devices with YOLO26 RKNN models can be used in various applications:

- **Smart Surveillance**: Deploy efficient object detection systems for security monitoring with low power consumption.
- **Industrial Automation**: Implement quality control and defect detection directly on embedded devices.
- **Retail Analytics**: Track customer behavior and inventory management in real-time without cloud dependency.
- **Smart Agriculture**: Monitor crop health and detect pests using [computer vision in agriculture](https://www.ultralytics.com/solutions/ai-in-agriculture).
- **Autonomous Robotics**: Enable vision-based navigation and obstacle detection on resource-constrained platforms.

## Benchmarks

YOLO26 benchmarks below were run by the Ultralytics team on Radxa Rock 5B based on Rockchip RK3588 with `rknn` model format measuring speed and accuracy.

!!! tip "Performance"

    | Model   | Format | Status | Size (MB) | mAP50-95(B) | Inference time (ms/im) |
    | ------- | ------ | ------ | --------- | ----------- | ---------------------- |
    | YOLO26n | `rknn` | ✅     | 7.1       | 0.479       | 65.7                   |
    | YOLO26s | `rknn` | ✅     | 20.9      | 0.571       | 99.2                   |
    | YOLO26m | `rknn` | ✅     | 42.5      | 0.610       | 235.3                  |
    | YOLO26l | `rknn` | ✅     | 52.1      | 0.630       | 280.5                  |
    | YOLO26x | `rknn` | ✅     | 112.2     | 0.666       | 669.1                  |

    Benchmarked with `ultralytics 8.4.23`

    !!! note

        Validation for the above benchmarks were done using COCO128 dataset. Inference time does not include pre/post-processing.

## Summary

In this guide, you've learned how to export Ultralytics YOLO26 models to RKNN format to enhance their deployment on Rockchip platforms. You were also introduced to the RKNN Toolkit and the specific advantages of using RKNN models for edge AI applications.

The combination of [Ultralytics YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) and Rockchip's NPU technology provides an efficient solution for running advanced computer vision tasks on embedded devices. This approach enables real-time [object detection](https://www.ultralytics.com/blog/a-guide-to-deep-dive-into-object-detection-in-2025) and other vision AI applications with minimal power consumption and high performance.

For further details on usage, visit the [RKNN official documentation](https://github.com/airockchip/rknn-toolkit2).

Also, if you'd like to know more about other Ultralytics YOLO26 integrations, visit our [integration guide page](../integrations/index.md). You'll find plenty of useful resources and insights there.

## FAQ

### How do I export my Ultralytics YOLO model to RKNN format?

You can easily export your Ultralytics YOLO model to RKNN format using the `export()` method in the Ultralytics Python package or via the command-line interface (CLI). Ensure you are using an x86-based Linux PC for the export process, as ARM64 devices like Rockchip are not supported for this operation. You can specify the target Rockchip platform using the `name` argument, such as `rk3588`, `rk3566`, or others. This process generates an optimized RKNN model ready for deployment on your Rockchip device, taking advantage of its Neural Processing Unit (NPU) for accelerated inference.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load your YOLO model
        model = YOLO("yolo26n.pt")

        # Export to RKNN format for a specific Rockchip platform
        model.export(format="rknn", name="rk3588")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=rknn name=rk3588
        ```

### What are the benefits of using RKNN models on Rockchip devices?

RKNN models are specifically designed to leverage the hardware acceleration capabilities of Rockchip's Neural Processing Units (NPUs). This optimization results in significantly faster inference speeds and reduced latency compared to running generic model formats like ONNX or TensorFlow Lite on the same hardware. Using RKNN models allows for more efficient use of the device's resources, leading to lower power consumption and better overall performance, especially critical for real-time applications on edge devices. By converting your Ultralytics YOLO models to RKNN, you can achieve optimal performance on devices powered by Rockchip SoCs like the RK3588, RK3566, and others.

### Can I deploy RKNN models on devices from other manufacturers like NVIDIA or Google?

RKNN models are specifically optimized for Rockchip platforms and their integrated NPUs. While you can technically run an RKNN model on other platforms using software emulation, you will not benefit from the hardware acceleration provided by Rockchip devices. For optimal performance on other platforms, it's recommended to export your Ultralytics YOLO models to formats specifically designed for those platforms, such as TensorRT for NVIDIA GPUs or [TensorFlow Lite](https://docs.ultralytics.com/integrations/tflite) for Google's Edge TPU. Ultralytics supports exporting to a wide range of formats, ensuring compatibility with various hardware accelerators.

### What Rockchip platforms are supported for RKNN model deployment?

The Ultralytics YOLO export to RKNN format supports Rockchip platforms with floating-point RKNN builds, including RK3588, RK3576, RK3566, RK3568, RK3562, RK2118 and RV1126B. It also supports INT8 quantized RKNN export with `int8=True`, which is required for INT8-only targets such as RV1103, RV1106, RV1103B and RV1106B. These platforms are commonly found in devices from manufacturers like Radxa, ASUS, Pine64, Orange Pi, Odroid, Khadas, and Banana Pi, allowing you to deploy your optimized RKNN models on a range of Rockchip-powered devices from single-board computers to industrial systems.

### How does the performance of RKNN models compare to other formats on Rockchip devices?

RKNN models generally outperform other formats like ONNX or TensorFlow Lite on Rockchip devices due to their optimization for Rockchip's NPUs. For instance, benchmarks on the Radxa Rock 5B (RK3588) show that [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26) in RKNN format achieves an inference time of 65.7 ms/image, significantly faster than other formats. This performance advantage is consistent across various YOLO26 model sizes, as demonstrated in the [benchmarks section](#benchmarks). By leveraging the dedicated NPU hardware, RKNN models minimize latency and maximize throughput, making them ideal for real-time applications on Rockchip-based edge devices.
