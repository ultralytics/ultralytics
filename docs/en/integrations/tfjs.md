---
comments: true
description: A guide that showcases how to export from an Ultralytics YOLOv8 model to TF.js model format for streamlined browser deployments and optimized model performance.
keywords: Ultralytics YOLOv8, TensorFlow.js, TF.js, Model Deployment, Node.js, Model Format, Export Format, Model Conversion
---

# Export to TF.js Model Format From a YOLOv8 Model Format

Deploying machine learning models directly in the browser or on Node.js can be tricky. You’ll need to make sure your model format is optimized for faster performance so that the model can be used to run interactive applications locally on the user’s device. The TensorFlow.js, or TF.js, model format is designed to use minimal power while delivering fast performance.

The ‘export to TF.js model format’ feature allows you to optimize your [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) models for high-speed and locally-run object detection inference. In this guide, we'll walk you through converting your models to the TF.js format, making it easier for your models to perform well on various local browsers and Node.js applications.

## Why Should You Export to TF.js?

Exporting your machine learning models to TensorFlow.js, developed by the TensorFlow team as part of the broader TensorFlow ecosystem, offers numerous advantages for deploying machine learning applications. It helps enhance user privacy and security by keeping sensitive data on the device. The image below shows the TensorFlow.js architecture, and how machine learning models are converted and deployed on both web browsers and Node.js.

<p align="center">
  <img width="100%" src="https://res.cloudinary.com/practicaldev/image/fetch/s--oepXBlvm--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://dev-to-uploads.s3.amazonaws.com/uploads/articles/m7r4grt0zkrgyx62xxx3.png" alt="TF.js Architecture">
</p>

Running models locally also reduces latency and provides a more responsive user experience. TensorFlow.js also comes with offline capabilities, allowing users to use your application even without an internet connection. TF.js is designed for efficient execution of complex models on devices with limited resources as it is engineered for scalability, with GPU acceleration support.

## Key Features of TF.js

Here are the key features that make TF.js a powerful tool for developers:

- **Cross-Platform Support:** TensorFlow.js can be used in both browser and Node.js environments, providing flexibility in deployment across different platforms. It lets developers build and deploy applications more easily.

- **Support for Multiple Backends:** TensorFlow.js supports various backends for computation including CPU, WebGL for GPU acceleration, WebAssembly (WASM) for near-native execution speed, and WebGPU for advanced browser-based machine learning capabilities.

- **Offline Capabilities:** With TensorFlow.js, models can run in the browser without the need for an internet connection, making it possible to develop applications that are functional offline.

## Deployment Options with TensorFlow.js

Before we dive into the process of exporting YOLOv8 models to the TF.js format, let's explore some typical deployment scenarios where this format is used.

TF.js provides a range of options to deploy your machine learning models:

- **In-Browser ML Applications:** You can build web applications that run machine learning models directly in the browser. The need for server-side computation is eliminated and the server load is reduced.

- **Node.js Applications::** TensorFlow.js also supports deployment in Node.js environments, enabling the development of server-side machine learning applications. It is particularly useful for applications that require the processing power of a server or access to server-side data​

- **Chrome Extensions:** An interesting deployment scenario is the creation of Chrome extensions with TensorFlow.js. For instance, you can develop an extension that allows users to right-click on an image within any webpage to classify it using a pre-trained ML model. TensorFlow.js can be integrated into everyday web browsing experiences to provide immediate insights or augmentations based on machine learning​.

## Exporting YOLOv8 Models to TensorFlow.js

You can expand model compatibility and deployment flexibility by converting YOLOv8 models to TF.js.

### Installation

To install the required package, run:

!!! Tip "Installation"

    === "CLI"
    
        ```bash
        # Install the required package for YOLOv8
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLOv8, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

Before diving into the usage instructions, it's important to note that while all [Ultralytics YOLOv8 models](../models/index.md) are available for exporting, you can ensure that the model you select supports export functionality [here](../modes/export.md).

!!! Example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLOv8 model
        model = YOLO("yolov8n.pt")

        # Export the model to TF.js format
        model.export(format="tfjs")  # creates '/yolov8n_web_model'

        # Load the exported TF.js model
        tfjs_model = YOLO("./yolov8n_web_model")

        # Run inference
        results = tfjs_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLOv8n PyTorch model to TF.js format
        yolo export model=yolov8n.pt format=tfjs  # creates '/yolov8n_web_model'

        # Run inference with the exported model
        yolo predict model='./yolov8n_web_model' source='https://ultralytics.com/images/bus.jpg'
        ```

For more details about supported export options, visit the [Ultralytics documentation page on deployment options](../guides/model-deployment-options.md).

## Deploying Exported YOLOv8 TensorFlow.js Models

Now that you have exported your YOLOv8 model to the TF.js format, the next step is to deploy it. The primary and recommended first step for running a TF.js is to use the YOLO("./yolov8n_web_model") method, as previously shown in the usage code snippet.

However, for in-depth instructions on deploying your TF.js models, take a look at the following resources:

- **[Chrome Extension](https://www.tensorflow.org/js/tutorials/deployment/web_ml_in_chrome)**: Here’s the developer documentation for how to deploy your TF.js models to a Chrome extension.

- **[Run TensorFlow.js in Node.js](https://www.tensorflow.org/js/guide/nodejs)**: A TensorFlow blog post on running TensorFlow.js in Node.js directly.

- **[Deploying TensorFlow.js - Node Project on Cloud Platform](https://www.tensorflow.org/js/guide/node_in_cloud)**: A TensorFlow blog post on deploying a TensorFlow.js model on a Cloud Platform.

## Summary

In this guide, we learned how to export Ultralytics YOLOv8 models to the TensorFlow.js format. By exporting to TF.js, you gain the flexibility to optimize, deploy, and scale your YOLOv8 models on a wide range of platforms.

For further details on usage, visit the [TensorFlow.js official documentation](https://www.tensorflow.org/js/guide).

For more information on integrating Ultralytics YOLOv8 with other platforms and frameworks, don't forget to check out our [integration guide page](index.md). It's packed with great resources to help you make the most of YOLOv8 in your projects.
