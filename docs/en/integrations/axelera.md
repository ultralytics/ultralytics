---
comments: true
description: Export Ultralytics YOLO models to Axelera format for high-performance edge AI deployment on Metis devices. Optimize inference with up to 856 TOPS.
keywords: Ultralytics, YOLO11, YOLOv8, Axelera AI, model export, edge AI, Metis AIPU, Voyager SDK, deployment, computer vision, quantization, mix-precision quantization
---

# Axelera Export and Deployment for Ultralytics YOLO11 Models

Deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models on edge devices requires hardware acceleration to achieve real-time performance while maintaining energy efficiency. Axelera AI's dedicated hardware accelerators provide the perfect solution for running advanced computer vision tasks at the edge with exceptional throughput.

The export to the Axelera format feature allows you to optimize your [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models for Axelera's Metis® AIPU. In this guide, we'll walk you through how to convert your models to the Axelera format, enabling deployment on devices ranging from embedded systems to edge servers with up to 856 TOPS of computing power.

## Why should you export to Axelera?

<p align="center">
  <img width="100%" src="https://github.com/user-attachments/assets/c97a0297-390d-47df-bb13-ff1aa499f34a" alt="Axelera AI Ecosystem">
</p>

[Axelera AI](https://www.axelera.ai/) provides dedicated hardware acceleration for computer vision and Generative AI at the edge. Their technology leverages a proprietary dataflow architecture and [in-memory computing](https://www.ultralytics.com/glossary/edge-computing) to deliver high throughput (up to 856 TOPS) within a low power envelope, making it ideal for [Edge AI](https://www.ultralytics.com/glossary/edge-ai) deployments.

Axelera's hardware is specifically optimized for running [neural networks](https://www.ultralytics.com/glossary/neural-network-nn) efficiently on edge devices, enabling rapid deployment of [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models across various platforms including PCIe cards, M.2 modules, and integrated systems.

## Key Models' Features on Axelera Devices

Models exported with Axelera offer powerful features that enable high-performance [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) on edge devices:

- **High Performance and Efficiency**: Models exported with Axelera deliver exceptional throughput (up to 856 TOPS) while maintaining low power consumption, optimized for edge deployment with high [accuracy](https://www.ultralytics.com/glossary/accuracy) on computer vision tasks.

- **INT8 Quantization**: Axelera's mixed-precision AIPU architecture enables automatic quantization through the Voyager SDK,

- **Flexible Deployment Options**: Axelera offers various form factors (PCIe, M.2, integrated systems) to suit different deployment constraints, from embedded devices to edge servers.

- **Voyager SDK Integration**: The Voyager SDK provides seamless compilation, quantization, and runtime execution, making it easy to deploy existing YOLO models on Axelera hardware.

## Deployment Options with Axelera

Before we look at the code for exporting YOLO11 models to the Axelera format, let's understand how Axelera-exported models are typically deployed.

Axelera-exported models support multiple deployment scenarios:

- **Edge Servers and Workstations**: PCIe cards (x1 and x4) for high-density video analytics and smart city applications.

- **Embedded Systems**: M.2 modules for drones, robotics, and portable devices requiring efficient on-device inference.

- **Integrated Solutions**: Complete systems like the Metis Compute Board combining Axelera AIPU with ARM processors for standalone edge deployments.

- **Industrial Applications**: Ruggedized systems for manufacturing automation and industrial safety monitoring.

## Supported Tasks

Currently, only Object Detection Ultralytics YOLO models can be exported to the Axelera format. Other tasks including Pose Estimation, Segmentation and Oriented Bounding Boxes are currently being integrated and will become available soon.

- [Object Detection](https://docs.ultralytics.com/tasks/detect/) (supported)
- [Pose Estimation](https://docs.ultralytics.com/tasks/pose/) _(coming soon)_
- [Segmentation](https://docs.ultralytics.com/tasks/segment/) _(coming soon)_
- [Oriented Bounding Boxes](https://docs.ultralytics.com/tasks/obb/) _(coming soon)_

## Exporting to Axelera: Convert Your YOLO11 Model

Export your YOLO11 and YOLOv8 models on Axelera hardware with the following steps:

### Requirements

!!! warning "Platform Requirements"

    In order to export Ultralytics YOLO models on the Axelera format, there are+ specific platform and hardware requirements:

    - **Operating System**: Linux only (Ubuntu 22.04/24.04 recommended)
    - **Hardware**: Axelera AI accelerator required ([Metis devices](https://store.axelera.ai/))
    - **Python**: Version 3.10 (3.11 and 3.12 coming soon)

### Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"
    ```bash
    # Install Ultralytics package
    pip install ultralytics
    ```

For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLO11, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

#### Axelera Driver Installation

1. Download and add the key to the system keyring

    ```bash
    sudo sh -c "curl -fsSL https://software.axelera.ai/artifactory/api/security/keypair/axelera/public | gpg --dearmor -o /etc/apt/keyrings/axelera.gpg"
    ```

2. Add the repository to apt

    ```bash
    sudo sh -c "echo 'deb [signed-by=/etc/apt/keyrings/axelera.gpg] https://software.axelera.ai/artifactory/axelera-apt-source/ ubuntu22 main' > /etc/apt/sources.list.d/axelera.list"
    ```

3. Update packages, install and hook the driver

    ```bash
    sudo apt update
    sudo apt install -y axelera-voyager-sdk-base
    sudo modprobe metis
    yes | sudo /opt/axelera/sdk/latest/axelera_fix_groups.sh $USER
    ```

### Usage

!!! example "Usage"

    === "Python"
        ```python
        from ultralytics import YOLO

        # Load the YOLO11 model
        model = YOLO("yolo11n.pt")

        # Export the model to Axelera format
        model.export(format="axelera")  # creates 'yolo11n_axelera_model' directory

        # Load the exported Axelera model
        axelera_model = YOLO("yolo11n_axelera_model")

        # Run inference
        results = axelera_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"
        ```bash
        # Export a YOLO11n PyTorch model to Axelera format
        yolo export model=yolo11n.pt format=axelera # creates 'yolo11n_axelera_model' directory

        # Run inference with the exported model
        yolo predict model='yolo11n_axelera_model' source='https://ultralytics.com/images/bus.jpg'
        ```

### Export Arguments

| Argument   | Type             | Default        | Description                                                                                                                                                                                                                                                      |
| ---------- | ---------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'axelera'`    | Target format for the exported model, optimized for Axelera Metis AIPU hardware.                                                                                                                                                                                 |
| `imgsz`    | `int` or `tuple` | `640`          | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                                                                                                |
| `int8`     | `bool`           | `True`         | Enables quantization for optimal performance on Axelera AIPUs.                                                                                                                                                                                                   |
| `batch`    | `int`            | `1`            | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                                                                                          |
| `data`     | `str`            | `'coco128.yaml'` | Path to the [dataset](https://docs.ultralytics.com/datasets/) configuration file (default: `coco128.yaml`), essential for quantization.                                                                                                                            |
| `fraction` | `float`          | `1.0`          | Specifies the fraction of the dataset to use for INT8 quantization calibration. Allows for calibrating on a subset of the full dataset, useful for experiments or when resources are limited. If not specified with INT8 enabled, the full dataset will be used. |
| `device`   | `str`            | `None`         | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`).                                                                                                                                                                                        |

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

### Output Structure

The Axelera export creates a directory containing the model and metadata:

```text
yolo11n_axelera_model/
├── yolo11n.axm              # Axelera model file
└── metadata.yaml            # Model metadata (classes, image size, etc.)
```

## Benchmarks

Coming soon: Performance benchmarks comparing inference speed and accuracy of YOLO11 models on Axelera hardware versus other edge AI platforms.

## Deploying Axelera-Exported YOLO11 Models

After successfully exporting your Ultralytics YOLO11 models to the Axelera format, you can run them on Axelera hardware. The primary and recommended first step for running an Axelera model is to utilize the YOLO("yolo11n_axelera_model") method, as outlined in the previous usage code snippet. However, for in-depth instructions on deploying your Axelera models in various settings, take a look at the following resources:

- **[PCIe Deployment](https://www.axelera.ai/)**: Learn how to deploy models on Metis PCIe cards for high-density video analytics.

- **[M.2 Module Integration](https://www.axelera.ai/)**: Understand deployment on M.2 modules for embedded systems and drones.

- **[Voyager SDK Documentation](https://www.axelera.ai/)**: Comprehensive guide for using the Voyager SDK compiler and runtime.

- **[Industrial Systems](https://www.axelera.ai/)**: Deploy on ruggedized industrial PCs for manufacturing and safety applications.

## About This Release

This release is an experimental version demonstrating how easily you can deploy models on the Axelera Metis hardware for your existing Ultralytics projects. We anticipate full integration by February 2026, which will provide:

- Model export capabilities without requiring Axelera hardware
- Standard pip installation (not dependent on our proprietary service URL)
- Automatic compiler configuration supporting multiple VoyagerSDK versions

**Current Implementation**

This integration focuses on providing an accessible, straightforward workflow for developers to get started quickly with Axelera acceleration. The current version uses a single-core configuration to ensure compatibility and ease of setup across different environments.

**Recommended Workflow**

We recommend leveraging the powerful Ultralytics `train` capabilities in this repository to develop and `export` your models, then using the `predict` and `val` functions for quantitative and qualitative validation. This streamlined approach allows you to seamlessly experiment with hardware acceleration on your custom-trained models.

**Unlocking Maximum Performance**

The integration shown here prioritizes ease of use and quick deployment. For production environments requiring maximum throughput, we recommend you to explore the [Axelera Voyager SDK](https://github.com/axelera-ai-hub/voyager-sdk) or [reach out to our team](https://axelera.ai/contact-us). The Voyager SDK offers advanced optimizations including:

- Multi-core utilization (quad-core Metis AIPU)
- Streaming inference pipelines
- Tiled inferencing for higher-resolution cameras
- Enhanced performance configurations

Visit our [model-zoo page](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.5/docs/reference/model_zoo.md) for reference FPS benchmarks, and stay tuned for upcoming examples demonstrating advanced pipeline configurations within the Ultralytics repository.

**Known Issues**

When using M.2 accelerators, you may encounter runtime errors with large or extra-large models due to power supply limitations.

If you encounter unexpected Axelera device usage or API issues, please visit the [Axelera Community](https://community.axelera.ai/) for solutions and support.

## Summary

In this guide, we've covered how to export Ultralytics YOLO11 models to the Axelera format. This conversion enables deployment on Axelera's high-performance edge AI hardware, achieving up to 856 TOPS while maintaining energy efficiency, making it ideal for demanding edge computing applications.

For detailed instructions on usage, please refer to the [official Axelera documentation](https://www.axelera.ai/).

If you're interested in exploring other integration options for Ultralytics YOLO11, be sure to visit our [integration guide page](index.md) for further insights and information.

## FAQ

### Which Axelera hardware should I choose for my YOLO11 deployment?

Axelera offers various hardware options depending on your needs:

- **Metis PCIe x4**: For maximum throughput (856 TOPS) in edge servers handling 30+ video streams.
- **Metis PCIe x1**: For standard PCs requiring 214 TOPS in a low-profile form factor.
- **Metis M.2**: For embedded systems like drones and robotics.
- **Metis Compute Board**: For standalone edge deployments with integrated ARM CPU.

Refer to the hardware selection chart in the documentation for detailed guidance.

### What platforms are supported for Axelera [model deployment](https://www.ultralytics.com/glossary/model-deployment)?

Axelera models can be deployed across various platforms:

- **Edge Servers**: High-density video analytics using PCIe cards.
- **Embedded Systems**: Drones, robotics, and IoT devices using M.2 modules.
- **Industrial PCs**: Ruggedized systems for manufacturing automation.
- **Integrated Solutions**: Standalone systems combining AIPU with ARM processors.

Each platform offers specific advantages for different edge AI applications.

### How does INT8 quantization affect YOLO11 model accuracy on Axelera?

Axelera's Voyager SDK uses advanced calibration techniques to automatically quantize models for our mixed-precision AIPU architecture. Our accuracy-preserving hardware-software co-design delivers best-in-class performance while maintaining model accuracy. The SDK intelligently determines the optimal quantization strategy to maximize hardware throughput.
For most [object detection](https://www.ultralytics.com/glossary/object-detection) tasks, the performance gains (higher FPS, lower power) significantly outweigh the negligible impact on mAP. The quantization process is fully automatic—no manual tuning required. Quantization takes from seconds to several hours depending on model size and configuration. Once complete, you get optimal inference performance and a portable AXM package ready for deployment. Simply run yolo val to validate and discover the remarkably minimal accuracy loss.


