---
comments: true
description: Export Ultralytics YOLO models to Axelera format for high-performance edge AI deployment on Metis AIPU and Europa platforms. Optimize inference with up to 856 TOPS.
keywords: Ultralytics, YOLO11, Axelera AI, model export, edge AI, Metis AIPU, Europa, Voyager SDK, deployment, computer vision, INT8 quantization
---

# Axelera Export and Deployment for Ultralytics YOLO11 Models

Deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models on edge devices requires hardware acceleration to achieve real-time performance while maintaining energy efficiency. Axelera AI's dedicated hardware accelerators provide the perfect solution for running advanced computer vision tasks at the edge with exceptional throughput.

The export to Axelera format feature allows you to optimize your [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models for Axelera's Metis® AIPU. In this guide, we'll walk you through how to convert your models to the Axelera format, enabling deployment on devices ranging from embedded systems to edge servers with up to 856 TOPS of computing power.

## Why should you export to Axelera?

<p align="center">
  <img width="100%" src="https://github.com/user-attachments/assets/c97a0297-390d-47df-bb13-ff1aa499f34a" alt="Axelera AI Ecosystem">
</p>

[Axelera AI](https://www.axelera.ai/) provides dedicated hardware acceleration for computer vision and Generative AI at the edge. Their technology leverages a proprietary dataflow architecture and [in-memory computing](https://www.ultralytics.com/glossary/edge-computing) to deliver high throughput (up to 856 TOPS) within a low power envelope, making it ideal for [Edge AI](https://www.ultralytics.com/glossary/edge-ai) deployments.

Axelera's hardware is specifically optimized for running [neural networks](https://www.ultralytics.com/glossary/neural-network-nn) efficiently on edge devices, enabling rapid deployment of [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models across various platforms including PCIe cards, M.2 modules, and integrated systems.

## Key Features of Axelera Models

Axelera models offer powerful features that enable high-performance [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) on edge devices:

- **High Performance and Efficiency**: Axelera models deliver exceptional throughput (up to 856 TOPS) while maintaining low power consumption, optimized for edge deployment with high [accuracy](https://www.ultralytics.com/glossary/accuracy) on computer vision tasks.

- **INT8 Quantization**: Axelera models leverage automatic INT8 quantization through the Voyager SDK, reducing model size and improving inference speed while maintaining model [precision](https://www.ultralytics.com/glossary/precision).

- **Flexible Deployment Options**: Axelera offers various form factors (PCIe, M.2, integrated systems) to suit different deployment constraints, from embedded devices to edge servers.

- **Voyager SDK Integration**: The Voyager SDK provides seamless compilation, quantization, and runtime execution, making it easy to deploy existing YOLO models on Axelera hardware.

## Deployment Options with Axelera

Before we look at the code for exporting YOLO11 models to the Axelera format, let's understand how Axelera models are typically deployed.

Axelera models support multiple deployment scenarios:

- **Edge Servers and Workstations**: PCIe cards (x1 and x4) for high-density video analytics and smart city applications.

- **Embedded Systems**: M.2 modules for drones, robotics, and portable devices requiring efficient on-device inference.

- **Integrated Solutions**: Complete systems like the Metis Compute Board combining Axelera AIPU with ARM processors for standalone edge deployments.

- **Industrial Applications**: Ruggedized systems for manufacturing automation and industrial safety monitoring.

## Supported Tasks

Currently, you can only export models that include the following tasks to Axelera format.

- [Object Detection](https://docs.ultralytics.com/tasks/detect/)

## Export to Axelera: Converting Your YOLO11 Model

You can optimize your YOLO11 models for Axelera hardware by converting them to the Axelera format.

### Requirements

!!! warning "Platform Requirements"

    Export to Axelera format and model inference have specific platform and hardware requirements:

    - **Operating System**: Linux only (Ubuntu 20.04/22.04 recommended)
    - **Hardware**: Axelera AI accelerator required (Metis AIPU or Europa platform)
    - **Python**: Version 3.10 or higher

### Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"
    ```bash
    # Install Ultralytics package
    pip install ultralytics
    ```

For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLO11, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

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
| `int8`     | `bool`           | `True`         | Enables INT8 quantization for optimal performance on Axelera NPUs.                                                                                                                                                                                               |
| `batch`    | `int`            | `1`            | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                                                                                          |
| `data`     | `str`            | `'coco8.yaml'` | Path to the [dataset](https://docs.ultralytics.com/datasets/) configuration file (default: `coco8.yaml`), essential for quantization.                                                                                                                            |
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

## Deploying Exported YOLO11 Axelera Models

After successfully exporting your Ultralytics YOLO11 models to Axelera format, you can now deploy them. The primary and recommended first step for running an Axelera model is to utilize the YOLO("yolo11n_axelera_model") method, as outlined in the previous usage code snippet. However, for in-depth instructions on deploying your Axelera models in various settings, take a look at the following resources:

- **[PCIe Deployment](https://www.axelera.ai/)**: Learn how to deploy models on Metis PCIe cards for high-density video analytics.

- **[M.2 Module Integration](https://www.axelera.ai/)**: Understand deployment on M.2 modules for embedded systems and drones.

- **[Voyager SDK Documentation](https://www.axelera.ai/)**: Comprehensive guide for using the Voyager SDK compiler and runtime.

- **[Industrial Systems](https://www.axelera.ai/)**: Deploy on ruggedized industrial PCs for manufacturing and safety applications.

## Summary

In this guide, we've covered exporting Ultralytics YOLO11 models to the Axelera format. This conversion enables deployment on Axelera's high-performance edge AI hardware, achieving up to 856 TOPS while maintaining energy efficiency, making it ideal for demanding edge computing applications.

For detailed instructions on usage, please refer to the [official Axelera documentation](https://www.axelera.ai/).

Also, if you're interested in exploring other integration options for Ultralytics YOLO11, be sure to visit our [integration guide page](index.md) for further insights and information.

## FAQ

### How do I export Ultralytics YOLO11 models to Axelera format?

To export your Ultralytics YOLO11 model to Axelera format (available Q1 2026), follow these steps:

- **Python**: Use the `export` function from the YOLO class.

```python
    from ultralytics import YOLO

    # Load the YOLO11 model
    model = YOLO("yolo11n.pt")

    # Export to Axelera format with INT8 quantization
    model.export(format="axelera", int8=True)
```

- **CLI**: Use the `yolo` command with the `export` argument.

```bash
yolo export model=yolo11n.pt format=axelera int8=True
```

For detailed export options, check the [Export](../modes/export.md) page in the documentation.

### What are the advantages of exporting YOLO11 models to Axelera?

Exporting your Ultralytics YOLO11 models to Axelera offers several benefits:

- **High Performance**: Up to 856 TOPS throughput for real-time edge AI applications.
- **Energy Efficiency**: Optimized dataflow architecture for low power consumption.
- **Flexible Deployment**: Multiple form factors (PCIe, M.2, integrated systems) for various use cases.
- **Automatic Quantization**: INT8 quantization through Voyager SDK for improved speed and reduced memory usage.

For more details, see the [Why should you export to Axelera](#why-should-you-export-to-axelera) section in the documentation.

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

Axelera's Voyager SDK uses advanced calibration techniques to minimize accuracy loss during INT8 quantization. For most [object detection](https://www.ultralytics.com/glossary/object-detection) tasks, the performance gains (higher FPS, lower power) significantly outweigh the negligible impact on mAP. The quantization process is automatic and optimized for maintaining model quality while maximizing hardware efficiency.

For more advanced deployment scenarios and optimization techniques, visit the [Ultralytics documentation page](../guides/model-deployment-options.md).
