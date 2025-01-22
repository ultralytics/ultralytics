---
comments: true
description: Discover how to get started with Seeed Studio reCamera for edge AI applications using Ultralytics YOLO11. Learn about its powerful features, real-world applications, and how to export YOLO11 models to ONNX format for seamless integration.
keywords: Seeed Studio reCamera, YOLO11, ONNX export, edge AI, computer vision, real-time detection, personal protective equipment detection, fire detection, waste detection, fall detection, modular AI devices, Ultralytics
---

# Quick Start Guide: Seeed Studio reCamera with Ultralytics YOLO11

[reCamera](https://www.seeedstudio.com/recamera) was introduced for the AI community at [YOLO Vision 2024 (YV24)](https://www.youtube.com/watch?v=rfI5vOo3-_A), [Ultralytics](https://ultralytics.com/) annual hybrid event. It is mainly designed for edge AI applications, offering powerful processing capabilities and effortless deployment.

With support for diverse hardware configurations and open-source resources, it serves as an ideal platform for prototyping and deploying innovative [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) [solutions](https://docs.ultralytics.com/solutions/#solutions) at the edge.

![Seeed Studio reCamera](https://github.com/ultralytics/docs/releases/download/0/saeed-studio-recamera.avif)

## Why Choose reCamera?

reCamera series is purpose-built for edge AI applications, tailored to meet the needs of developers and innovators. Here's why it stands out:

- **RISC-V Powered Performance**: At its core is the SG200X processor, built on the RISC-V architecture, delivering exceptional performance for edge AI tasks while maintaining energy efficiency. With the ability to execute 1 trillion operations per second (1 TOPS), it handles demanding tasks like real-time object detection easily.

- **Optimized Video Technologies**: Supports advanced video compression standards, including H.264 and H.265, to reduce storage and bandwidth requirements without sacrificing quality. Features like HDR imaging, 3D noise reduction, and lens correction ensure professional visuals, even in challenging environments.

- **Energy-Efficient Dual Processing**: While the SG200X handles complex AI tasks, a smaller 8-bit microcontroller manages simpler operations to conserve power, making the reCamera ideal for battery-operated or low-power setups.

- **Modular and Upgradable Design**: The reCamera is built with a modular structure, consisting of three main components: the core board, sensor board, and baseboard. This design allows developers to easily swap or upgrade components, ensuring flexibility and future-proofing for evolving projects.

## Quick Hardware Setup of reCamera

Please follow [reCamera Quick Start Guide](https://wiki.seeedstudio.com/recamera_getting_started) for initial onboarding of the device such as connecting the device to a WiFi network and access the [Node-RED](https://nodered.org) web UI for quick previewing of detection redsults with the pre-installed Ultralytics YOLO models.

## Export to cvimodel: Converting Your YOLO11 Model

Here we will first convert `PyTorch` model to `ONNX` and then convert it to `MLIR` model format. Finally `MLIR` will be converted to `cvimodel` in order to inference on-device

<p align="center">
  <img width="80%" src="https://github.com/ultralytics/assets/releases/download/v0.0.0/recamera-toolchain-workflow.avif" alt="reCamera Toolchain">
</p>

### Export to ONNX

Export an Ultralytics YOLO11 model to ONNX model format.

#### Installation

To install the required packages, run:

!!! Tip "Installation"

    === "CLI"

        ```bash
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLO11, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

#### Usage

!!! Example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO11 model
        model = YOLO("yolo11n.pt")

        # Export the model to ONNX format
        model.export(format="onnx")  # creates 'yolo11n.onnx'
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to ONNX format
        yolo export model=yolo11n.pt format=onnx  # creates 'yolo11n.onnx'
        ```

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

### Export ONNX to MLIR and cvimodel

After obtaining an ONNX model, refer to [Convert and Quantize AI Models](https://wiki.seeedstudio.com/recamera_model_conversion) page to convert the ONNX model to MLIR and then to cvimodel.

!!! note

    We're actively working on adding reCamera support directly into the Ultralytics package, and it will be available soon. In the meantime, check out our blog on [Integrating Ultralytics YOLO Models with Seeed Studio's reCamera](https://www.ultralytics.com/blog/integrating-ultralytics-yolo-models-on-seeed-studios-recamera) for more insights.

## Benchmarks

Coming soon.

## Real-World Applications of reCamera

reCamera advanced computer vision capabilities and modular design make it suitable for a wide range of real-world scenarios, helping developers and businesses tackle unique challenges with ease.

- **Fall Detection**: Designed for safety and healthcare applications, the reCamera can detect falls in real-time, making it ideal for elderly care, hospitals, and industrial settings where rapid response is critical.

- **Personal Protective Equipment Detection**: The reCamera can be used to ensure workplace safety by detecting PPE compliance in real-time. It helps identify whether workers are wearing helmets, gloves, or other safety gear, reducing risks in industrial environments.

![Personal protective equipment detection](https://github.com/ultralytics/docs/releases/download/0/personal-protective-equipment-detection.avif)

- **Fire Detection**: The reCamera's real-time processing capabilities make it an excellent choice for fire detection in industrial and residential areas, providing early warnings to prevent potential disasters.

- **Waste Detection**: It can also be utilized for waste detection applications, making it an excellent tool for environmental monitoring and waste management.

- **Car Parts Detection**: In manufacturing and automotive industries, it aids in detecting and analyzing car parts for quality control, assembly line monitoring, and inventory management.

![Car parts detection](https://github.com/ultralytics/docs/releases/download/0/carparts-detection.avif)
