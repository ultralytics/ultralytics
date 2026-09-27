---
comments: true
description: Export Ultralytics YOLO models to TI Deep Learning (TIDL) artifacts for accelerated INT8 inference on the C7x NPU in Texas Instruments TDA4x and AM6xA edge AI processors.
keywords: Texas Instruments, TI Edge AI, TIDL, TDA4VH, AM62A, AM69A, J784S4, C7x NPU, edgeai-tidl-runtime, edgeai-tidl-tools, YOLO26, model export, INT8 quantization, embedded vision, automotive ADAS, industrial AI, Ultralytics
---

# Texas Instruments Edge AI Export for Ultralytics YOLO Models

Deploying computer vision models on [Texas Instruments](https://www.ti.com/) Edge AI processors requires compiling them with the TI Deep Learning (TIDL) toolchain so they run on the on-chip C7x digital signal processor and matrix multiply accelerator (the C7x NPU). Exporting [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models to the `ti` format produces TIDL artifacts ready to copy onto TDA4x and AM6xA devices used across automotive, industrial, and robotics applications.

## What is TI Edge AI?

[TI Edge AI](https://github.com/TexasInstruments/edgeai) is Texas Instruments' open-source ecosystem for deploying AI models on its MPU processors. At its core is the **TIDL** runtime, which compiles standard ONNX models into device-specific artifacts executed by the C7x NPU through ONNX Runtime with TIDL offload. Layers that TIDL cannot accelerate fall back to the Arm cores.

The Ultralytics `ti` export uses the `edgeai-tidl-runtime` Python package, which bundles the native TIDL tools and a TIDL-enabled ONNX Runtime build. Compilation runs entirely on the host, so you can export on a regular x86-64 Linux machine without a TI device attached or a TI account.

## TI Edge AI Export Format

The Ultralytics exporter first writes an intermediate ONNX graph, fixes it to a static input shape, then compiles it with TIDL for the device family you target with the `name` argument. INT8 calibration uses images from the `data` dataset. The result is a self-contained model directory holding the compiled TIDL artifacts and their Ultralytics metadata.

## Key Features of TI Edge AI Models

- **Dedicated NPU**: Compiled models execute on the C7x NPU rather than the Arm application cores, delivering much higher throughput for real-time camera pipelines.
- **Host-side compilation**: TIDL compilation needs no attached device, so export runs in CI, in containers, or on a workstation.
- **ONNX-first**: The standard [Ultralytics ONNX export](onnx.md) feeds directly into TIDL with no proprietary intermediate format.
- **INT8 quantization**: TIDL runs quantized fixed-point inference, calibrated during export on a sample of your dataset.
- **Industrial-grade hardware**: TDA4x (Jacinto) devices are qualified for automotive and industrial use, with long supply lifecycles.

## Supported Devices

Pass the target device family with `name`. TIDL artifacts are compiled for one device family, so `name` must match the board you deploy to.

| `name`   | Devices                                                                                                                                     |
| :------- | :------------------------------------------------------------------------------------------------------------------------------------------ |
| `j784s4` | [TDA4VH](https://www.ti.com/product/TDA4VH-Q1) · [TDA4AH](https://www.ti.com/product/TDA4AH-Q1) · [AM69A](https://www.ti.com/product/AM69A) |
| `am62a`  | [AM62A3](https://www.ti.com/product/AM62A3) · [AM62A7](https://www.ti.com/product/AM62A7)                                                   |

Other TIDL-compatible families such as J721E (TDA4VM), J721S2 (TDA4VE, TDA4AL, AM68A), and J722S (TDA4AEN, AM67A) can be targeted with TI's own toolchain; see [Using the TI Toolchain Directly](#using-the-ti-toolchain-directly). See the [edgeai-tidl-tools SDK version compatibility table](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/sdk_version_compatibility_table.md) to match the TIDL version to the Processor SDK on your device.

## Export to TI Edge AI: Converting Your YOLO Model

!!! note

    TI Edge AI export is supported only on x86-64 Linux, and `edgeai-tidl-runtime` requires Python 3.10.

### Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required packages for YOLO and TI Edge AI export
        pip install ultralytics edgeai-tidl-runtime
        ```

If `edgeai-tidl-runtime` is missing, Ultralytics attempts to install it automatically on the first export. For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). If you encounter any difficulties while installing the required packages, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

The TI Edge AI format supports the [Export](../modes/export.md), [Predict](../modes/predict.md), and [Validate](../modes/val.md) modes. Inference and validation run on TI hardware. Export your model, then load the exported model directory to run inference or validate its accuracy.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to TI Edge AI format for a TDA4VH / AM69A device
        model.export(format="ti", name="j784s4", data="coco8.yaml")

        # Load the exported TI model and run inference on the device
        ti_model = YOLO("yolo26n_ti_model")
        results = ti_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to TI Edge AI format
        yolo export model=yolo26n.pt format=ti name=j784s4 data=coco8.yaml

        # Run inference with the exported model on the device
        yolo predict model=yolo26n_ti_model source='https://ultralytics.com/images/bus.jpg'
        ```

### Export Arguments

| Argument   | Type             | Default    | Description                                                                                                                                |
| :--------- | :--------------- | :--------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'ti'`     | Target format for the exported model, defining compatibility with TI Edge AI processors.                                                   |
| `name`     | `str`            | `'j784s4'` | Target device family, `'j784s4'` or `'am62a'`; see [Supported Devices](#supported-devices). Must match your deployment device.             |
| `imgsz`    | `int` or `tuple` | `640`      | Desired image size for the model input, compiled into the artifacts as a static shape.                                                     |
| `batch`    | `int`            | `1`        | Static batch size compiled into the artifacts.                                                                                             |
| `quantize` | `int`            | `8`        | Quantization precision. TI Edge AI export is INT8-only and auto-enables `8` if not specified. Replaces the deprecated `half`/`int8` flags. |
| `data`     | `str`            | `None`     | Dataset configuration file used for INT8 calibration, e.g. `'coco8.yaml'`. Defaults to the task's default dataset if not specified.        |
| `fraction` | `float`          | `1.0`      | Fraction of the calibration dataset to use for INT8 quantization. Lower values speed up export for large datasets.                         |
| `opset`    | `int`            | `None`     | ONNX opset for the intermediate graph. Defaults to the latest version supported by the installed ONNX.                                     |
| `simplify` | `bool`           | `True`     | Simplifies the intermediate ONNX graph with `onnxslim`.                                                                                    |
| `device`   | `str`            | `None`     | Device for the intermediate ONNX export step, CPU by default (`device=cpu`). TIDL compilation does not use it.                             |

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

### Output Structure

After a successful export, a model directory is created holding the compiled TIDL artifacts and a `metadata.yaml` file. The metadata contains class names, image size, the target device family, and other information used by the Ultralytics inference pipeline. Copy the whole directory to the device; the artifacts are valid only for the exported input shape and device family.

## Deploying Exported YOLO TI Edge AI Models

Once you've exported your model, copy the `yolo26n_ti_model` directory to a TI device running the [TI Processor SDK](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md) with `edgeai-tidl-runtime` installed, then run inference exactly as you would with any other Ultralytics format. The runtime loads the artifacts for the device family recorded in `metadata.yaml`.

## Using the TI Toolchain Directly

For fine-grained control over compilation, custom post-processing, or device families not listed above, use TI's own tools with an Ultralytics [ONNX export](onnx.md):

- **[edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner)**: A high-level CLI and Python API that compiles, benchmarks, and evaluates models from a per-model YAML configuration.
- **[edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools)**: The low-level TIDL SDK for full control over compilation options, model partitioning, and quantization. See the [TIDL user guide](https://github.com/TexasInstruments/edgeai-tidl-tools#user-guide) for supported operators and runtime options.
- **[TI Edge AI Model Hub](https://github.com/TexasInstruments/edgeai-modelhub/tree/main/models/vision/detection)**: Pre-validated YOLO26, YOLO11, and YOLOv8 detection models with per-model configuration files for `edgeai-tidlrunner`.

## Recommended Workflow

1. **Train** your model using Ultralytics [Train Mode](../modes/train.md), or start from a pre-trained checkpoint.
2. **Export** to TI Edge AI on an x86-64 Linux host, passing the `name` that matches your target device and a `data` file representative of your deployment images for calibration.
3. **Copy** the exported model directory to the TI device.
4. **Deploy** with `edgeai-tidl-runtime` for on-device inference.

## Real-World Applications

YOLO models running on TI Edge AI hardware suit a wide range of embedded and industrial vision applications:

- **Automotive ADAS**: Pedestrian detection, driver and occupant monitoring, and surround-view perception on TDA4x devices.
- **Industrial Automation**: High-speed quality inspection and defect detection on production lines.
- **Robotics**: On-board perception for autonomous mobile robots and collaborative arms.
- **Smart Surveillance**: Real-time multi-camera object detection on edge gateways without a central server.

## Summary

In this guide, you have learned how to export Ultralytics YOLO models to the TI Edge AI format with TIDL. Export runs entirely on an x86-64 Linux host, calibrates INT8 quantization on your data, and produces a self-contained model directory that targets a TI device family through a single `name` argument.

For more details, visit the [TI Edge AI documentation](https://github.com/TexasInstruments/edgeai/tree/main/edgeai-mpu).

Also, if you'd like to know more about other Ultralytics YOLO integrations, visit our [integration guide page](index.md) to find plenty of helpful resources.

## FAQ

### How do I export my Ultralytics YOLO model to TI Edge AI format?

Install Ultralytics and `edgeai-tidl-runtime` on an x86-64 Linux machine with Python 3.10, then export with `model.export(format="ti", name="j784s4", data="coco8.yaml")` in Python or `yolo export model=yolo26n.pt format=ti name=j784s4 data=coco8.yaml` from the CLI. Set `name` to the device family of your deployment board.

### Do I need a TI device to export a model?

No. TIDL compilation runs entirely on the host, so export works on a standard x86-64 Linux machine with no TI device attached. You only need TI hardware to run inference on the exported model.

### Why does TI Edge AI export only support INT8?

The C7x NPU runs fixed-point inference, so the exporter always compiles INT8 artifacts and calibrates them on images from `data`. Requesting `quantize=32` or `quantize=16` raises an error rather than producing a model that cannot run on the accelerator.

### Which TI devices are supported?

The `ti` export targets the `j784s4` (TDA4VH, TDA4AH, AM69A) and `am62a` (AM62A3, AM62A7) device families. Other TIDL-compatible devices can use an Ultralytics ONNX export with [TI's toolchain](#using-the-ti-toolchain-directly).

### What is the difference between edgeai-tidl-tools and edgeai-tidlrunner?

[edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) is the low-level TIDL SDK that gives full control over compilation, model partitioning, and quantization. [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner) is a higher-level CLI and Python API that wraps TIDL tools and automates benchmarking and accuracy evaluation from a single YAML configuration file. The Ultralytics `ti` export handles compilation for you, so you only need these tools for advanced workflows.
