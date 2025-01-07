---
comments: true
description: Learn how to export YOLO11 models to the TF GraphDef format for seamless deployment on various platforms, including mobile and web.
keywords: YOLO11, export, TensorFlow, GraphDef, model deployment, TensorFlow Serving, TensorFlow Lite, TensorFlow.js, machine learning, AI, computer vision
---

# How to Export to TF GraphDef from YOLO11 for Deployment

When you are deploying cutting-edge [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models, like YOLO11, in different environments, you might run into compatibility issues. Google's [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) GraphDef, or TF GraphDef, offers a solution by providing a serialized, platform-independent representation of your model. Using the TF GraphDef model format, you can deploy your YOLO11 model in environments where the complete TensorFlow ecosystem may not be available, such as mobile devices or specialized hardware.

In this guide, we'll walk you step by step through how to export your [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models to the TF GraphDef model format. By converting your model, you can streamline deployment and use YOLO11's computer vision capabilities in a broader range of applications and platforms.

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/tensorflow-graphdef.avif" alt="TensorFlow GraphDef">
</p>

## Why Should You Export to TF GraphDef?

TF GraphDef is a powerful component of the TensorFlow ecosystem that was developed by Google. It can be used to optimize and deploy models like YOLO11. Exporting to TF GraphDef lets us move models from research to real-world applications. It allows models to run in environments without the full TensorFlow framework.

The GraphDef format represents the model as a serialized computation graph. This enables various optimization techniques like constant folding, quantization, and graph transformations. These optimizations ensure efficient execution, reduced memory usage, and faster inference speeds.

GraphDef models can use hardware accelerators such as GPUs, TPUs, and AI chips, unlocking significant performance gains for the YOLO11 inference pipeline. The TF GraphDef format creates a self-contained package with the model and its dependencies, simplifying deployment and integration into diverse systems.

## Key Features of TF GraphDef Models

TF GraphDef offers distinct features for streamlining [model deployment](https://www.ultralytics.com/glossary/model-deployment) and optimization.

Here's a look at its key characteristics:

- **Model Serialization**: TF GraphDef provides a way to serialize and store TensorFlow models in a platform-independent format. This serialized representation allows you to load and execute your models without the original Python codebase, making deployment easier.

- **Graph Optimization**: TF GraphDef enables the optimization of computational graphs. These optimizations can boost performance by streamlining execution flow, reducing redundancies, and tailoring operations to suit specific hardware.

- **Deployment Flexibility**: Models exported to the GraphDef format can be used in various environments, including resource-constrained devices, web browsers, and systems with specialized hardware. This opens up possibilities for wider deployment of your TensorFlow models.

- **Production Focus**: GraphDef is designed for production deployment. It supports efficient execution, serialization features, and optimizations that align with real-world use cases.

## Deployment Options with TF GraphDef

Before we dive into the process of exporting YOLO11 models to TF GraphDef, let's take a look at some typical deployment situations where this format is used.

Here's how you can deploy with TF GraphDef efficiently across various platforms.

- **TensorFlow Serving:** This framework is designed to deploy TensorFlow models in production environments. TensorFlow Serving offers model management, versioning, and the infrastructure for efficient model serving at scale. It's a seamless way to integrate your GraphDef-based models into production web services or APIs.

- **Mobile and Embedded Devices:** With tools like TensorFlow Lite, you can convert TF GraphDef models into formats optimized for smartphones, tablets, and various embedded devices. Your models can then be used for on-device inference, where execution is done locally, often providing performance gains and offline capabilities.

- **Web Browsers:** TensorFlow.js enables the deployment of TF GraphDef models directly within web browsers. It paves the way for real-time object detection applications running on the client side, using the capabilities of YOLO11 through JavaScript.

- **Specialized Hardware:** TF GraphDef's platform-agnostic nature allows it to target custom hardware, such as accelerators and TPUs (Tensor Processing Units). These devices can provide performance advantages for computationally intensive models.

## Exporting YOLO11 Models to TF GraphDef

You can convert your YOLO11 object detection model to the TF GraphDef format, which is compatible with various systems, to improve its performance across platforms.

### Installation

To install the required package, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO11
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLO11, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

Before diving into the usage instructions, it's important to note that while all [Ultralytics YOLO11 models](../models/index.md) are available for exporting, you can ensure that the model you select supports export functionality [here](../modes/export.md).

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO11 model
        model = YOLO("yolo11n.pt")

        # Export the model to TF GraphDef format
        model.export(format="pb")  # creates 'yolo11n.pb'

        # Load the exported TF GraphDef model
        tf_graphdef_model = YOLO("yolo11n.pb")

        # Run inference
        results = tf_graphdef_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to TF GraphDef format
        yolo export model=yolo11n.pt format=pb  # creates 'yolo11n.pb'

        # Run inference with the exported model
        yolo predict model='yolo11n.pb' source='https://ultralytics.com/images/bus.jpg'
        ```

For more details about supported export options, visit the [Ultralytics documentation page on deployment options](../guides/model-deployment-options.md).

## Deploying Exported YOLO11 TF GraphDef Models

Once you've exported your YOLO11 model to the TF GraphDef format, the next step is deployment. The primary and recommended first step for running a TF GraphDef model is to use the YOLO("model.pb") method, as previously shown in the usage code snippet.

However, for more information on deploying your TF GraphDef models, take a look at the following resources:

- **[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)**: A guide on TensorFlow Serving that teaches how to deploy and serve [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models efficiently in production environments.

- **[TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter)**: This page describes how to convert machine learning models into a format optimized for on-device inference with TensorFlow Lite.

- **[TensorFlow.js](https://www.tensorflow.org/js/guide/conversion)**: A guide on model conversion that teaches how to convert TensorFlow or Keras models into TensorFlow.js format for use in web applications.

## Summary

In this guide, we explored how to export Ultralytics YOLO11 models to the TF GraphDef format. By doing this, you can flexibly deploy your optimized YOLO11 models in different environments.

For further details on usage, visit the [TF GraphDef official documentation](https://www.tensorflow.org/api_docs/python/tf/Graph).

For more information on integrating Ultralytics YOLO11 with other platforms and frameworks, don't forget to check out our [integration guide page](index.md). It has great resources and insights to help you make the most of YOLO11 in your projects.

## FAQ

### How do I export a YOLO11 model to TF GraphDef format?

Ultralytics YOLO11 models can be exported to TensorFlow GraphDef (TF GraphDef) format seamlessly. This format provides a serialized, platform-independent representation of the model, ideal for deploying in varied environments like mobile and web. To export a YOLO11 model to TF GraphDef, follow these steps:

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO11 model
        model = YOLO("yolo11n.pt")

        # Export the model to TF GraphDef format
        model.export(format="pb")  # creates 'yolo11n.pb'

        # Load the exported TF GraphDef model
        tf_graphdef_model = YOLO("yolo11n.pb")

        # Run inference
        results = tf_graphdef_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to TF GraphDef format
        yolo export model="yolo11n.pt" format="pb"  # creates 'yolo11n.pb'

        # Run inference with the exported model
        yolo predict model="yolo11n.pb" source="https://ultralytics.com/images/bus.jpg"
        ```

For more information on different export options, visit the [Ultralytics documentation on model export](../modes/export.md).

### What are the benefits of using TF GraphDef for YOLO11 model deployment?

Exporting YOLO11 models to the TF GraphDef format offers multiple advantages, including:

1. **Platform Independence**: TF GraphDef provides a platform-independent format, allowing models to be deployed across various environments including mobile and web browsers.
2. **Optimizations**: The format enables several optimizations, such as constant folding, quantization, and graph transformations, which enhance execution efficiency and reduce memory usage.
3. **Hardware Acceleration**: Models in TF GraphDef format can leverage hardware accelerators like GPUs, TPUs, and AI chips for performance gains.

Read more about the benefits in the [TF GraphDef section](#why-should-you-export-to-tf-graphdef) of our documentation.

### Why should I use Ultralytics YOLO11 over other [object detection](https://www.ultralytics.com/glossary/object-detection) models?

Ultralytics YOLO11 offers numerous advantages compared to other models like YOLOv5 and YOLOv7. Some key benefits include:

1. **State-of-the-Art Performance**: YOLO11 provides exceptional speed and [accuracy](https://www.ultralytics.com/glossary/accuracy) for real-time object detection, segmentation, and classification.
2. **Ease of Use**: Features a user-friendly API for model training, validation, prediction, and export, making it accessible for both beginners and experts.
3. **Broad Compatibility**: Supports multiple export formats including ONNX, TensorRT, CoreML, and TensorFlow, for versatile deployment options.

Explore further details in our [introduction to YOLO11](https://docs.ultralytics.com/models/yolov8/).

### How can I deploy a YOLO11 model on specialized hardware using TF GraphDef?

Once a YOLO11 model is exported to TF GraphDef format, you can deploy it across various specialized hardware platforms. Typical deployment scenarios include:

- **TensorFlow Serving**: Use TensorFlow Serving for scalable model deployment in production environments. It supports model management and efficient serving.
- **Mobile Devices**: Convert TF GraphDef models to TensorFlow Lite, optimized for mobile and embedded devices, enabling on-device inference.
- **Web Browsers**: Deploy models using TensorFlow.js for client-side inference in web applications.
- **AI Accelerators**: Leverage TPUs and custom AI chips for accelerated inference.

Check the [deployment options](#deployment-options-with-tf-graphdef) section for detailed information.

### Where can I find solutions for common issues while exporting YOLO11 models?

For troubleshooting common issues with exporting YOLO11 models, Ultralytics provides comprehensive guides and resources. If you encounter problems during installation or model export, refer to:

- **[Common Issues Guide](../guides/yolo-common-issues.md)**: Offers solutions to frequently faced problems.
- **[Installation Guide](../quickstart.md)**: Step-by-step instructions for setting up the required packages.

These resources should help you resolve most issues related to YOLO11 model export and deployment.
