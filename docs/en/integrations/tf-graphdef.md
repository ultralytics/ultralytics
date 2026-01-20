---
comments: true
description: Learn how to export YOLO26 models to the TF GraphDef format for seamless deployment on various platforms, including mobile and web.
keywords: YOLO26, export, TensorFlow, GraphDef, model deployment, TensorFlow Serving, TensorFlow Lite, TensorFlow.js, machine learning, AI, computer vision
---

# How to Export to TF GraphDef from YOLO26 for Deployment

When you are deploying cutting-edge [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models, like YOLO26, in different environments, you might run into compatibility issues. Google's [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) GraphDef, or TF GraphDef, offers a solution by providing a serialized, platform-independent representation of your model. Using the TF GraphDef model format, you can deploy your YOLO26 model in environments where the complete TensorFlow ecosystem may not be available, such as mobile devices or specialized hardware.

In this guide, we'll walk you step by step through how to export your [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) models to the TF GraphDef model format. By converting your model, you can streamline deployment and use YOLO26's computer vision capabilities in a broader range of applications and platforms.

<p align="center">
  <img width="640" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/tensorflow-graphdef.avif" alt="TensorFlow GraphDef model serialization format">
</p>

## Why Should You Export to TF GraphDef?

TF GraphDef is a powerful component of the TensorFlow ecosystem that was developed by Google. It can be used to optimize and deploy models like YOLO26. Exporting to TF GraphDef lets you move models from research to real-world applications. It allows models to run in environments without the full TensorFlow framework.

The GraphDef format represents the model as a serialized computation graph. This enables various optimization techniques like constant folding, quantization, and graph transformations. These optimizations ensure efficient execution, reduced memory usage, and faster inference speeds.

GraphDef models can use hardware accelerators such as GPUs, TPUs, and AI chips, unlocking significant performance gains for the YOLO26 inference pipeline. The TF GraphDef format creates a self-contained package with the model and its dependencies, simplifying deployment and integration into diverse systems.

## Key Features of TF GraphDef Models

TF GraphDef offers distinct features for streamlining [model deployment](https://www.ultralytics.com/glossary/model-deployment) and optimization.

Here's a look at its key characteristics:

- **Model Serialization**: TF GraphDef provides a way to serialize and store TensorFlow models in a platform-independent format. This serialized representation allows you to load and execute your models without the original Python codebase, making deployment easier.

- **Graph Optimization**: TF GraphDef enables the optimization of computational graphs. These optimizations can boost performance by streamlining execution flow, reducing redundancies, and tailoring operations to suit specific hardware.

- **Deployment Flexibility**: Models exported to the GraphDef format can be used in various environments, including resource-constrained devices, web browsers, and systems with specialized hardware. This opens up possibilities for wider deployment of your TensorFlow models.

- **Production Focus**: GraphDef is designed for production deployment. It supports efficient execution, serialization features, and optimizations that align with real-world use cases.

## Deployment Options with TF GraphDef

Before we dive into the process of exporting YOLO26 models to TF GraphDef, let's take a look at some typical deployment situations where this format is used.

Here's how you can deploy with TF GraphDef efficiently across various platforms.

- **TensorFlow Serving:** This framework is designed to deploy TensorFlow models in production environments. TensorFlow Serving offers model management, versioning, and the infrastructure for efficient model serving at scale. It's a seamless way to integrate your GraphDef-based models into production web services or APIs.

- **Mobile and Embedded Devices:** With tools like [TensorFlow Lite](../integrations/tflite.md), you can convert TF GraphDef models into formats optimized for smartphones, tablets, and various embedded devices. Your models can then be used for on-device inference, where execution is done locally, often providing performance gains and offline capabilities.

- **Web Browsers:** [TensorFlow.js](../integrations/tfjs.md) enables the deployment of TF GraphDef models directly within web browsers. It paves the way for real-time object detection applications running on the client side, using the capabilities of YOLO26 through JavaScript.

- **Specialized Hardware:** TF GraphDef's platform-agnostic nature allows it to target custom hardware, such as accelerators and TPUs (Tensor Processing Units). These devices can provide performance advantages for computationally intensive models.

## Exporting YOLO26 Models to TF GraphDef

You can convert your YOLO26 object detection model to the TF GraphDef format, which is compatible with various systems, to improve its performance across platforms.

### Installation

To install the required package, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO26
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLO26, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

All [Ultralytics YOLO26 models](../models/index.md) are designed to support export out of the box, making it easy to integrate them into your preferred deployment workflow. You can [view the full list of supported export formats and configuration options](../modes/export.md) to choose the best setup for your application.

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to TF GraphDef format
        model.export(format="pb")  # creates 'yolo26n.pb'

        # Load the exported TF GraphDef model
        tf_graphdef_model = YOLO("yolo26n.pb")

        # Run inference
        results = tf_graphdef_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to TF GraphDef format
        yolo export model=yolo26n.pt format=pb # creates 'yolo26n.pb'

        # Run inference with the exported model
        yolo predict model='yolo26n.pb' source='https://ultralytics.com/images/bus.jpg'
        ```

### Export Arguments

| Argument | Type             | Default | Description                                                                                                                             |
| -------- | ---------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'pb'`  | Target format for the exported model, defining compatibility with various deployment environments.                                      |
| `imgsz`  | `int` or `tuple` | `640`   | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.       |
| `batch`  | `int`            | `1`     | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode. |
| `device` | `str`            | `None`  | Specifies the device for exporting: CPU (`device=cpu`), MPS for Apple silicon (`device=mps`).                                           |

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLO26 TF GraphDef Models

Once you've exported your YOLO26 model to the TF GraphDef format, the next step is deployment. The primary and recommended first step for running a TF GraphDef model is to use the YOLO("model.pb") method, as previously shown in the usage code snippet.

However, for more information on deploying your TF GraphDef models, take a look at the following resources:

- **[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)**: A guide on TensorFlow Serving that teaches how to deploy and serve [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models efficiently in production environments.

- **[TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter)**: This page describes how to convert machine learning models into a format optimized for on-device inference with TensorFlow Lite.

- **[TensorFlow.js](https://www.tensorflow.org/js/guide/conversion)**: A guide on model conversion that teaches how to convert TensorFlow or Keras models into TensorFlow.js format for use in web applications.

## Summary

In this guide, we explored how to export Ultralytics YOLO26 models to the TF GraphDef format. By doing this, you can flexibly deploy your optimized YOLO26 models in different environments.

For further details on usage, visit the [TF GraphDef official documentation](https://www.tensorflow.org/api_docs/python/tf/Graph).

For more information on integrating Ultralytics YOLO26 with other platforms and frameworks, see our [integration guide page](index.md).

## FAQ

### How do I export a YOLO26 model to TF GraphDef format?

Ultralytics YOLO26 models can be exported to TensorFlow GraphDef (TF GraphDef) format seamlessly. This format provides a serialized, platform-independent representation of the model, ideal for deploying in varied environments like mobile and web. To export a YOLO26 model to TF GraphDef, follow these steps:

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to TF GraphDef format
        model.export(format="pb")  # creates 'yolo26n.pb'

        # Load the exported TF GraphDef model
        tf_graphdef_model = YOLO("yolo26n.pb")

        # Run inference
        results = tf_graphdef_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to TF GraphDef format
        yolo export model="yolo26n.pt" format="pb" # creates 'yolo26n.pb'

        # Run inference with the exported model
        yolo predict model="yolo26n.pb" source="https://ultralytics.com/images/bus.jpg"
        ```

For more information on different export options, visit the [Ultralytics documentation on model export](../modes/export.md).

### What are the benefits of using TF GraphDef for YOLO26 model deployment?

Exporting YOLO26 models to the TF GraphDef format offers multiple advantages, including:

1. **Platform Independence**: TF GraphDef provides a platform-independent format, allowing models to be deployed across various environments including mobile and web browsers.
2. **Optimizations**: The format enables several optimizations, such as constant folding, quantization, and graph transformations, which enhance execution efficiency and reduce memory usage.
3. **Hardware Acceleration**: Models in TF GraphDef format can leverage hardware accelerators like GPUs, TPUs, and AI chips for performance gains.

Read more about the benefits in the [TF GraphDef section](#why-should-you-export-to-tf-graphdef) of our documentation.

### Why should I use Ultralytics YOLO26 over other [object detection](https://www.ultralytics.com/glossary/object-detection) models?

Ultralytics YOLO26 offers numerous advantages compared to other models like YOLOv5 and YOLOv7. Some key benefits include:

1. **State-of-the-Art Performance**: YOLO26 provides exceptional speed and [accuracy](https://www.ultralytics.com/glossary/accuracy) for real-time object detection, segmentation, and classification.
2. **Ease of Use**: Features a user-friendly API for model training, validation, prediction, and export, making it accessible for both beginners and experts.
3. **Broad Compatibility**: Supports multiple export formats including ONNX, TensorRT, CoreML, and TensorFlow, for versatile deployment options.

Explore further details in our [introduction to YOLO26](../models/yolo26.md).

### How can I deploy a YOLO26 model on specialized hardware using TF GraphDef?

Once a YOLO26 model is exported to TF GraphDef format, you can deploy it across various specialized hardware platforms. Typical deployment scenarios include:

- **TensorFlow Serving**: Use TensorFlow Serving for scalable model deployment in production environments. It supports model management and efficient serving.
- **Mobile Devices**: Convert TF GraphDef models to TensorFlow Lite, optimized for mobile and embedded devices, enabling on-device inference.
- **Web Browsers**: Deploy models using TensorFlow.js for client-side inference in web applications.
- **AI Accelerators**: Leverage TPUs and custom AI chips for accelerated inference.

Check the [deployment options](#deployment-options-with-tf-graphdef) section for detailed information.

### Where can I find solutions for common issues while exporting YOLO26 models?

For troubleshooting common issues with exporting YOLO26 models, Ultralytics provides comprehensive guides and resources. If you encounter problems during installation or model export, refer to:

- **[Common Issues Guide](../guides/yolo-common-issues.md)**: Offers solutions to frequently faced problems.
- **[Installation Guide](../quickstart.md)**: Step-by-step instructions for setting up the required packages.

These resources should help you resolve most issues related to YOLO26 model export and deployment.
