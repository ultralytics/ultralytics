---
comments: true
description: Convert your Ultralytics YOLO11 models to TensorFlow.js for high-speed, local object detection. Learn how to optimize ML models for browser and Node.js apps.
keywords: YOLO11, TensorFlow.js, TF.js, model export, machine learning, object detection, browser ML, Node.js, Ultralytics, YOLO, export models
---

# Export to TF.js Model Format From a YOLO11 Model Format

Deploying [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models directly in the browser or on Node.js can be tricky. You'll need to make sure your model format is optimized for faster performance so that the model can be used to run interactive applications locally on the user's device. The TensorFlow.js, or TF.js, model format is designed to use minimal power while delivering fast performance.

The 'export to TF.js model format' feature allows you to optimize your [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models for high-speed and locally-run [object detection](https://www.ultralytics.com/glossary/object-detection) inference. In this guide, we'll walk you through converting your models to the TF.js format, making it easier for your models to perform well on various local browsers and Node.js applications.

## Why Should You Export to TF.js?

Exporting your machine learning models to TensorFlow.js, developed by the TensorFlow team as part of the broader TensorFlow ecosystem, offers numerous advantages for deploying machine learning applications. It helps enhance user privacy and security by keeping sensitive data on the device. The image below shows the TensorFlow.js architecture, and how machine learning models are converted and deployed on both web browsers and Node.js.

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/tfjs-architecture.avif" alt="TF.js Architecture">
</p>

Running models locally also reduces latency and provides a more responsive user experience. [TensorFlow.js](https://www.ultralytics.com/glossary/tensorflow) also comes with offline capabilities, allowing users to use your application even without an internet connection. TF.js is designed for efficient execution of complex models on devices with limited resources as it is engineered for scalability, with GPU acceleration support.

## Key Features of TF.js

Here are the key features that make TF.js a powerful tool for developers:

- **Cross-Platform Support:** TensorFlow.js can be used in both browser and Node.js environments, providing flexibility in deployment across different platforms. It lets developers build and deploy applications more easily.

- **Support for Multiple Backends:** TensorFlow.js supports various backends for computation including CPU, WebGL for GPU acceleration, WebAssembly (WASM) for near-native execution speed, and WebGPU for advanced browser-based machine learning capabilities.

- **Offline Capabilities:** With TensorFlow.js, models can run in the browser without the need for an internet connection, making it possible to develop applications that are functional offline.

## Deployment Options with TensorFlow.js

Before we dive into the process of exporting YOLO11 models to the TF.js format, let's explore some typical deployment scenarios where this format is used.

TF.js provides a range of options to deploy your machine learning models:

- **In-Browser ML Applications:** You can build web applications that run machine learning models directly in the browser. The need for server-side computation is eliminated and the server load is reduced.

- **Node.js Applications:** TensorFlow.js also supports deployment in Node.js environments, enabling the development of server-side machine learning applications. It is particularly useful for applications that require the processing power of a server or access to server-side data.

- **Chrome Extensions:** An interesting deployment scenario is the creation of Chrome extensions with TensorFlow.js. For instance, you can develop an extension that allows users to right-click on an image within any webpage to classify it using a pre-trained ML model. TensorFlow.js can be integrated into everyday web browsing experiences to provide immediate insights or augmentations based on machine learning.

## Exporting YOLO11 Models to TensorFlow.js

You can expand model compatibility and deployment flexibility by converting YOLO11 models to TF.js.

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

All [Ultralytics YOLO11 models](../models/index.md) are designed to support export out of the box, making it easy to integrate them into your preferred deployment workflow. You can [view the full list of supported export formats and configuration options](../modes/export.md) to choose the best setup for your application.

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO11 model
        model = YOLO("yolo11n.pt")

        # Export the model to TF.js format
        model.export(format="tfjs")  # creates '/yolo11n_web_model'

        # Load the exported TF.js model
        tfjs_model = YOLO("./yolo11n_web_model")

        # Run inference
        results = tfjs_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to TF.js format
        yolo export model=yolo11n.pt format=tfjs # creates '/yolo11n_web_model'

        # Run inference with the exported model
        yolo predict model='./yolo11n_web_model' source='https://ultralytics.com/images/bus.jpg'
        ```

### Export Arguments

| Argument | Type             | Default  | Description                                                                                                                                                                                   |
| -------- | ---------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'tfjs'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                            |
| `imgsz`  | `int` or `tuple` | `640`    | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                             |
| `half`   | `bool`           | `False`  | Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.                                                                  |
| `int8`   | `bool`           | `False`  | Activates INT8 quantization, further compressing the model and speeding up inference with minimal [accuracy](https://www.ultralytics.com/glossary/accuracy) loss, primarily for edge devices. |
| `nms`    | `bool`           | `False`  | Adds Non-Maximum Suppression (NMS), essential for accurate and efficient detection post-processing.                                                                                           |
| `batch`  | `int`            | `1`      | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                       |
| `device` | `str`            | `None`   | Specifies the device for exporting: CPU (`device=cpu`), MPS for Apple silicon (`device=mps`).                                                                                                 |

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLO11 TensorFlow.js Models

Now that you have exported your YOLO11 model to the TF.js format, the next step is to deploy it. The primary and recommended first step for running a TF.js model is to use the `YOLO("./yolo11n_web_model")` method, as previously shown in the usage code snippet.

However, for in-depth instructions on deploying your TF.js models, take a look at the following resources:

- **[Chrome Extension](https://www.tensorflow.org/js/tutorials/deployment/web_ml_in_chrome)**: Here's the developer documentation for how to deploy your TF.js models to a Chrome extension.

- **[Run TensorFlow.js in Node.js](https://www.tensorflow.org/js/guide/nodejs)**: A TensorFlow blog post on running TensorFlow.js in Node.js directly.

- **[Deploying TensorFlow.js - Node Project on Cloud Platform](https://www.tensorflow.org/js/guide/node_in_cloud)**: A TensorFlow blog post on deploying a TensorFlow.js model on a Cloud Platform.

## Summary

In this guide, we learned how to export Ultralytics YOLO11 models to the TensorFlow.js format. By exporting to TF.js, you gain the flexibility to optimize, deploy, and scale your YOLO11 models on a wide range of platforms.

For further details on usage, visit the [TensorFlow.js official documentation](https://www.tensorflow.org/js/guide).

For more information on integrating Ultralytics YOLO11 with other platforms and frameworks, don't forget to check out our [integration guide page](index.md). It's packed with great resources to help you make the most of YOLO11 in your projects.

## FAQ

### How do I export Ultralytics YOLO11 models to TensorFlow.js format?

Exporting Ultralytics YOLO11 models to TensorFlow.js (TF.js) format is straightforward. You can follow these steps:

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO11 model
        model = YOLO("yolo11n.pt")

        # Export the model to TF.js format
        model.export(format="tfjs")  # creates '/yolo11n_web_model'

        # Load the exported TF.js model
        tfjs_model = YOLO("./yolo11n_web_model")

        # Run inference
        results = tfjs_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to TF.js format
        yolo export model=yolo11n.pt format=tfjs # creates '/yolo11n_web_model'

        # Run inference with the exported model
        yolo predict model='./yolo11n_web_model' source='https://ultralytics.com/images/bus.jpg'
        ```

For more details about supported export options, visit the [Ultralytics documentation page on deployment options](../guides/model-deployment-options.md).

### Why should I export my YOLO11 models to TensorFlow.js?

Exporting YOLO11 models to TensorFlow.js offers several advantages, including:

1. **Local Execution:** Models can run directly in the browser or Node.js, reducing latency and enhancing user experience.
2. **Cross-Platform Support:** TF.js supports multiple environments, allowing flexibility in deployment.
3. **Offline Capabilities:** Enables applications to function without an internet connection, ensuring reliability and privacy.
4. **GPU Acceleration:** Leverages WebGL for GPU acceleration, optimizing performance on devices with limited resources.

For a comprehensive overview, see our [Integrations with TensorFlow.js](../integrations/tf-graphdef.md).

### How does TensorFlow.js benefit browser-based machine learning applications?

TensorFlow.js is specifically designed for efficient execution of ML models in browsers and Node.js environments. Here's how it benefits browser-based applications:

- **Reduces Latency:** Runs machine learning models locally, providing immediate results without relying on server-side computations.
- **Improves Privacy:** Keeps sensitive data on the user's device, minimizing security risks.
- **Enables Offline Use:** Models can operate without an internet connection, ensuring consistent functionality.
- **Supports Multiple Backends:** Offers flexibility with backends like CPU, WebGL, WebAssembly (WASM), and WebGPU for varying computational needs.

Interested in learning more about TF.js? Check out the [official TensorFlow.js guide](https://www.tensorflow.org/js/guide).

### What are the key features of TensorFlow.js for deploying YOLO11 models?

Key features of TensorFlow.js include:

- **Cross-Platform Support:** TF.js can be used in both web browsers and Node.js, providing extensive deployment flexibility.
- **Multiple Backends:** Supports CPU, WebGL for GPU acceleration, WebAssembly (WASM), and WebGPU for advanced operations.
- **Offline Capabilities:** Models can run directly in the browser without internet connectivity, making it ideal for developing responsive web applications.

For deployment scenarios and more in-depth information, see our section on [Deployment Options with TensorFlow.js](#deployment-options-with-tensorflowjs).

### Can I deploy a YOLO11 model on server-side Node.js applications using TensorFlow.js?

Yes, TensorFlow.js allows the deployment of YOLO11 models on Node.js environments. This enables server-side machine learning applications that benefit from the processing power of a server and access to server-side data. Typical use cases include real-time data processing and machine learning pipelines on backend servers.

To get started with Node.js deployment, refer to the [Run TensorFlow.js in Node.js](https://www.tensorflow.org/js/guide/nodejs) guide from TensorFlow.
