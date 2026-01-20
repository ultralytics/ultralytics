---
comments: true
description: Discover how to get started with Seeed Studio reCamera for edge AI applications using Ultralytics YOLO26. Learn about its powerful features, real-world applications, and how to export YOLO26 models to ONNX format for seamless integration.
keywords: Seeed Studio reCamera, YOLO26, ONNX export, edge AI, computer vision, real-time detection, personal protective equipment detection, fire detection, waste detection, fall detection, modular AI devices, Ultralytics
---

# Quick Start Guide: Seeed Studio reCamera with Ultralytics YOLO26

[reCamera](https://www.seeedstudio.com/recamera) was introduced for the AI community at [YOLO Vision 2024 (YV24)](https://www.youtube.com/watch?v=rfI5vOo3-_A), [Ultralytics](https://www.ultralytics.com/) annual hybrid event. It is mainly designed for [edge AI applications](https://www.ultralytics.com/blog/understanding-the-real-world-applications-of-edge-ai), offering powerful processing capabilities and effortless deployment.

With support for diverse hardware configurations and open-source resources, it serves as an ideal platform for prototyping and deploying innovative [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) [solutions](https://docs.ultralytics.com/solutions/#solutions) at the edge.

![Seeed Studio reCamera](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/saeed-studio-recamera.avif)

## Why Choose reCamera?

reCamera series is purpose-built for edge AI applications, tailored to meet the needs of developers and innovators. Here's why it stands out:

- **RISC-V Powered Performance**: At its core is the SG200X processor, built on the RISC-V architecture, delivering exceptional performance for edge AI tasks while maintaining energy efficiency. With the ability to execute 1 trillion operations per second (1 TOPS), it handles demanding tasks like real-time [object detection](https://docs.ultralytics.com/tasks/detect/) easily.

- **Optimized Video Technologies**: Supports advanced video compression standards, including H.264 and H.265, to reduce storage and bandwidth requirements without sacrificing quality. Features like HDR imaging, 3D noise reduction, and lens correction ensure professional visuals, even in challenging environments.

- **Energy-Efficient Dual Processing**: While the SG200X handles complex AI tasks, a smaller 8-bit microcontroller manages simpler operations to conserve power, making the reCamera ideal for battery-operated or low-power setups.

- **Modular and Upgradable Design**: The reCamera is built with a modular structure, consisting of three main components: the core board, sensor board, and baseboard. This design allows developers to easily swap or upgrade components, ensuring flexibility and future-proofing for evolving projects.

## Quick Hardware Setup of reCamera

Please follow [reCamera Quick Start Guide](https://wiki.seeedstudio.com/recamera_getting_started/) for initial onboarding of the device such as connecting the device to a WiFi network and access the [Node-RED](https://nodered.org/) web UI for quick previewing of detection results.

## Inference Using Pre-installed YOLO26 Models

reCamera comes pre-installed with four Ultralytics YOLO26 models and you can simply choose your desired model within the Node-RED dashboard.

- [Detection (YOLO26n)](../tasks/detect.md)
- [Classification (YOLO26n-cls)](../tasks/classify.md)
- [Segmentation (YOLO26n-seg)](../tasks/segment.md)
- [Pose Estimation (YOLO26n-pose)](../tasks/pose.md)

Step 1: If you have connected reCamera to a network, enter the IP address of reCamera on a web browser to open the Node-RED dashboard. If you have connected the reCamera to a PC via USB, you can enter `192.168.42.1`. Here you will see YOLO26n detection model is loaded by default.

![reCamera YOLO11n demo](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/recamera-yolo11n-demo.avif)

Step 2: Click the green color circle at the bottom right corner to access the Node-RED flow editor.

Step 3: Click the `model` node and click `On Device`.

![Node-RED model selection](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/recamera-nodered-model-select.avif)

Step 4: Choose one of the four different pre-installed YOLO26n models and click `Done`. For example, here we will select `YOLO26n Pose`

<p align="center">
  <img width="50%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/recamera-nodered-yolo11n-pose.avif" alt="Node-RED YOLO11n-pose select">
</p>

Step 5: Click `Deploy` and when it finishes deploying, click `Dashboard`.

![reCamera Node-RED deploy](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/recamera-nodered-deploy.avif)

Now you will be able to see YOLO26n pose estimation model in action!

![reCamera YOLO11n-pose demo](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/recamera-yolo11n-pose-demo.avif)

## Export to cvimodel: Converting Your YOLO26 Model

If you want to use a [custom-trained YOLO26 model](../modes/train.md) with reCamera, follow the steps below.

Here, we'll first convert a `PyTorch` model to `ONNX` and then convert it to the `MLIR` model format. Finally, `MLIR` will be converted to `cvimodel` to run inference on-device.

<p align="center">
  <img width="80%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/recamera-toolchain-workflow.avif" alt="Seeed Studio reCamera AI development toolchain">
</p>

### Export to ONNX

Export an Ultralytics YOLO26 model to [ONNX model format](https://docs.ultralytics.com/integrations/onnx/).

#### Installation

To install the required packages, run:

!!! Tip "Installation"

    === "CLI"

        ```bash
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLO26, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

#### Usage

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to ONNX format
        model.export(format="onnx", opset=14)  # creates 'yolo26n.onnx'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to ONNX format
        yolo export model=yolo26n.pt format=onnx opset=14 # creates 'yolo26n.onnx'
        ```

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

### Export ONNX to MLIR and cvimodel

After obtaining an ONNX model, refer to [Convert and Quantize AI Models](https://wiki.seeedstudio.com/recamera_model_conversion/) page to convert the ONNX model to MLIR and then to cvimodel.

!!! note

    We're actively working on adding reCamera support directly into the Ultralytics package, and it will be available soon. In the meantime, check out our blog on [Integrating Ultralytics YOLO Models with Seeed Studio's reCamera](https://www.ultralytics.com/blog/integrating-ultralytics-yolo-models-on-seeed-studios-recamera) for more insights.

## Benchmarks

Coming soon.

## Real-World Applications of reCamera

reCamera advanced computer vision capabilities and modular design make it suitable for a wide range of real-world scenarios, helping developers and businesses tackle unique challenges with ease.

- **Fall Detection**: Designed for safety and healthcare applications, the reCamera can detect falls in real-time, making it ideal for elderly care, hospitals, and industrial settings where rapid response is critical.

- **Personal Protective Equipment Detection**: The reCamera can be used to ensure workplace safety by detecting PPE compliance in real-time. It helps identify whether workers are wearing helmets, gloves, or other safety gear, reducing risks in industrial environments.

![Personal protective equipment detection](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/personal-protective-equipment-detection.avif)

- **Fire Detection**: The reCamera's real-time processing capabilities make it an excellent choice for [fire detection](https://www.ultralytics.com/blog/computer-vision-in-fire-detection-and-prevention) in industrial and residential areas, providing early warnings to prevent potential disasters.

- **Waste Detection**: It can also be utilized for waste detection applications, making it an excellent tool for environmental monitoring and [waste management](https://www.ultralytics.com/blog/simplifying-e-waste-management-with-ai-innovations).

- **Car Parts Detection**: In manufacturing and automotive industries, it aids in detecting and analyzing car parts for quality control, assembly line monitoring, and inventory management.

![YOLO car parts detection for automotive inspection](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/carparts-detection.avif)

## FAQ

### How do I install and set up reCamera for the first time?

To set up your reCamera for the first time, follow these steps:

1. Connect the reCamera to a power source
2. Connect it to your WiFi network using the [reCamera Quick Start Guide](https://wiki.seeedstudio.com/recamera_getting_started/)
3. Access the Node-RED web UI by entering the device's IP address in a web browser (or use `192.168.42.1` if connected via USB)
4. Start using the pre-installed YOLO26 models immediately through the dashboard interface

### Can I use my custom-trained YOLO26 models with reCamera?

Yes, you can use custom-trained YOLO26 models with reCamera. The process involves:

1. Export your PyTorch model to ONNX format using `model.export(format="onnx", opset=14)`
2. Convert the ONNX model to MLIR format
3. Convert the MLIR to cvimodel format for on-device inference
4. Load the converted model onto your reCamera

For detailed instructions, refer to the [Convert and Quantize AI Models](https://wiki.seeedstudio.com/recamera_model_conversion/) guide.

### What makes reCamera different from traditional IP cameras?

Unlike traditional IP cameras that require external hardware for processing, reCamera:

- Integrates AI processing directly on the device with its RISC-V SG200X processor
- Offers 1 TOPS of computing power for real-time edge AI applications
- Features a modular design allowing for component upgrades and customization
- Supports advanced video technologies like H.264/H.265 compression, HDR imaging, and 3D noise reduction
- Comes pre-installed with Ultralytics YOLO26 models for immediate use

These features make reCamera a standalone solution for edge AI applications without requiring additional external processing hardware.
