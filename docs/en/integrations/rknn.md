---
comments: true
description: Learn how to export YOLOv8 models to RKNN format for efficient deployment on Rockchip platforms with enhanced performance.
keywords: YOLOv8, RKNN, model export, Ultralytics, Rockchip, machine learning, model deployment, computer vision, deep learning
---

# RKNN Export for YOLOv8 Models

When deploying computer vision models on embedded devices, especially those powered by Rockchip processors, having a compatible model format is essential.

Exporting [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) models to RKNN format ensures optimized performance and compatibility with Rockchip's hardware. This guide will walk you through converting your YOLOv8 models to RKNN format, enabling efficient deployment on Rockchip platforms.

## RKNN and Rockchip Platforms

The [RKNN Toolkit](https://github.com/rockchip-linux/rknn-toolkit2) is a set of tools and libraries provided by Rockchip to facilitate the deployment of deep learning models on their hardware platforms. RKNN, or Rockchip Neural Network, is the proprietary format used by these tools.

RKNN models are designed to take full advantage of the hardware acceleration provided by Rockchip's NPU (Neural Processing Unit), ensuring high performance in AI tasks on devices like RV1103, RV1106, and other Rockchip-powered systems.

<p align="center">
  <img width="100%" src="https://www.rock-chips.com/Images/web/solution/AI/chip_s.png" alt="RKNN">
</p>

## Key Features of RKNN Models

RKNN models offer several advantages for deployment on Rockchip platforms:

- **Optimized for NPU**: RKNN models are specifically optimized to run on Rockchip's NPUs, ensuring maximum performance and efficiency.
- **Low Latency**: The RKNN format minimizes inference latency, which is critical for real-time applications on edge devices.
- **Platform-Specific Customization**: RKNN models can be tailored to specific Rockchip platforms, enabling better utilization of hardware resources.

## Common Usage of RKNN

RKNN models are primarily used for deploying AI applications on Rockchip-based devices, including:

### Edge AI Deployment

RKNN models are ideal for edge AI applications, where processing needs to occur locally on devices like smart cameras, IoT devices, and robotics, without relying on cloud computing.

### Supported Deployment Options

While RKNN models are designed for Rockchip platforms, they can also be deployed in various embedded environments, making them versatile for edge computing needs.

## Exporting YOLOv8 Models to RKNN

Converting YOLOv8 models to RKNN format expands their deployment options, particularly for edge devices powered by Rockchip.

### Installation

To install the required packages, run:

!!! Tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLOv8 and RKNN
        pip install ultralytics rknn-toolkit>=1.4.0
        ```

For detailed instructions and best practices related to the installation process, check our [YOLOv8 Installation guide](../quickstart.md). If you encounter any difficulties during the installation, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

Before proceeding with the export, ensure that you have chosen the appropriate YOLOv8 model for your application. This will help you achieve the best performance on your target Rockchip platform.

!!! Example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLOv8 model
        model = YOLO("yolov8n.pt")

        # Export the model to RKNN format
        model.export(format="rknn", args={"name": "rk3588"})  # creates 'yolov8n-rk3588.rknn'

        # Run inference
        results = rknn_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLOv8n PyTorch model to RKNN format
        yolo export model=yolov8n.pt format=rknn name=rk3588  # creates 'yolov8n-rk3588.rknn'
        ```

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLOv8 RKNN Models

Once you've successfully exported your Ultralytics YOLOv8 models to RKNN format, the next step is deploying these models on Rockchip-based devices. For detailed instructions on deploying your RKNN models, refer to the following resources:

- **[RKNN Toolkit Documentation](https://github.com/rockchip-linux/rknn-toolkit)**: This guide provides essential information for loading and running RKNN models on Rockchip platforms.
- **[Deploying on Edge Devices](https://github.com/rockchip-linux/rknn-toolkit2)**: Explore examples and tutorials on deploying RKNN models on various Rockchip-based edge devices.

## Summary

In this guide, you've learned how to export Ultralytics YOLOv8 models to RKNN format to enhance their deployment on Rockchip platforms. You were also introduced to the RKNN Toolkit and the specific advantages of using RKNN models for edge AI applications.

For further details on usage, visit the [RKNN official documentation](https://github.com/rockchip-linux/rknn-toolkit).

Also, if you'd like to know more about other Ultralytics YOLOv8 integrations, visit our [integration guide page](../integrations/index.md). You'll find plenty of useful resources and insights there.
