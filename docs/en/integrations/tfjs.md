---
title: TensorFlow.js (TF.js) Export (Deprecated)
comments: true
description: Legacy guide to YOLO TensorFlow.js export, deprecated in Ultralytics 8.4.83. format="tfjs" now exports LiteRT; use LiteRT for browser and edge deployment.
keywords: YOLO26, TensorFlow.js, TF.js, model export, machine learning, object detection, browser ML, Node.js, Ultralytics, YOLO, export models
---

# Export to TF.js Model Format From a YOLO26 Model Format (Deprecated)

!!! warning "Deprecated — replaced by LiteRT"

    As of **Ultralytics 8.4.83**, the `tfjs` (TensorFlow.js) export format has been deprecated and replaced by the unified **Google LiteRT** format. LiteRT runs in the browser via [LiteRT.js](https://developers.google.com/edge/litert/web) (with WebGPU acceleration) — covering the in-browser and Node.js use cases TF.js handled, alongside mobile, embedded, and edge, from a single `.tflite` model.

    `format="tfjs"` still works but emits a deprecation warning and exports a LiteRT model instead. **Use `format="litert"`** going forward; for current export instructions and the browser deployment path, see the **[LiteRT export guide](litert.md)**.

Deploying [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models directly in the browser or on Node.js can be tricky. You'll need to make sure your model format is optimized for faster performance so that the model can be used to run interactive applications locally on the user's device. The TensorFlow.js, or TF.js, model format is designed to use minimal power while delivering fast performance.

## Why Should You Export to TF.js?

Exporting your machine learning models to TensorFlow.js, developed by the TensorFlow team as part of the broader TensorFlow ecosystem, offers numerous advantages for deploying machine learning applications. It helps enhance user privacy and security by keeping sensitive data on the device. The image below shows the TensorFlow.js architecture, and how machine learning models are converted and deployed on both web browsers and Node.js.

<p align="center">
  <img width="100%" src="https://cdn.ul.run/i/deb6049be0dd044a4eb34b7c680977f0.avif" alt="TensorFlow.js browser ML inference architecture">
</p>

Running models locally also reduces [latency](https://www.ultralytics.com/glossary/inference-latency) and provides a more responsive user experience. [TensorFlow.js](https://www.tensorflow.org/js) also comes with offline capabilities, allowing users to use your application even without an internet connection. TF.js is designed for efficient execution of complex models on devices with limited resources as it is engineered for scalability, with GPU acceleration support.

## Key Features of TF.js

Here are the key features that make TF.js useful for developers:

- **Cross-Platform Support:** TensorFlow.js can be used in both browser and Node.js environments, providing flexibility in deployment across different platforms. It lets developers build and deploy applications more easily.

- **Support for Multiple Backends:** TensorFlow.js supports various backends for computation including CPU, WebGL for GPU acceleration, WebAssembly (WASM) for near-native execution speed, and WebGPU for advanced browser-based machine learning capabilities.

- **Offline Capabilities:** With TensorFlow.js, models can run in the browser without the need for an internet connection, making it possible to develop applications that are functional offline.

## Deployment Options with TensorFlow.js

TF.js provides a range of options to deploy your machine learning models:

- **In-Browser ML Applications:** You can build web applications that run machine learning models directly in the browser. The need for server-side computation is eliminated and the server load is reduced.

- **Node.js Applications:** TensorFlow.js also supports deployment in Node.js environments, enabling the development of server-side machine learning applications. It is particularly useful for applications that require the processing power of a server or access to server-side data.

- **Chrome Extensions:** An interesting deployment scenario is the creation of Chrome extensions with TensorFlow.js. For instance, you can develop an extension that allows users to right-click on an image within any webpage to classify it using a pretrained ML model. TensorFlow.js can be integrated into everyday web browsing experiences to provide immediate insights or augmentations based on machine learning.

## Replace TF.js Export with LiteRT

For new exports, convert your model to LiteRT.

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

All [Ultralytics YOLO26 models](https://platform.ultralytics.com/ultralytics/yolo26) are designed to support export out of the box, making it easy to integrate them into your preferred deployment workflow. You can [view the full list of supported export formats and configuration options](../modes/export.md) to choose the best setup for your application.

The `tfjs` argument now exports a LiteRT `.tflite` model, which supports the [Export](../modes/export.md), [Predict](../modes/predict.md), and [Validate](../modes/val.md) modes locally and runs in the browser via LiteRT.js. See the [LiteRT export guide](litert.md) for the current browser and on-device deployment path.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to LiteRT format (replaces TF.js)
        model.export(format="litert", imgsz=640)  # use imgsz=224 for classification
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to LiteRT format (replaces TF.js)
        yolo export model=yolo26n.pt format=litert imgsz=640 # use imgsz=224 for classification
        ```

!!! note "Predict and Validate"

    The exported LiteRT `.tflite` model loads directly with `yolo predict` and `yolo val`, and runs in the browser via LiteRT.js.

### Export Arguments

See the [LiteRT export arguments](litert.md#export-arguments). For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLO26 TensorFlow.js Models

Now that you have exported your YOLO26 model, the next step is to deploy it. The export produces a LiteRT `.tflite` model that runs on-device and in the browser via LiteRT.js, as described in the LiteRT export guide linked above.

For reference, the legacy TensorFlow.js runtime resources are below:

- **[Chrome Extension](https://www.tensorflow.org/js/tutorials/deployment/web_ml_in_chrome)**: Developer documentation for deploying your TF.js models to a Chrome extension.

- **[Run TensorFlow.js in Node.js](https://www.tensorflow.org/js/guide/nodejs)**: A TensorFlow guide on running TensorFlow.js in Node.js directly.

- **[Deploying TensorFlow.js - Node Project on Cloud Platform](https://www.tensorflow.org/js/guide/node_in_cloud)**: A TensorFlow guide on deploying a TensorFlow.js Node project on a cloud platform.

## Summary

This guide preserves the legacy TF.js deployment context. Use LiteRT for new exports; for a walkthrough, see [LiteRT export for YOLO edge, mobile, and browser inference](https://www.ultralytics.com/blog/export-ultralytics-yolo-with-litert-for-edge-and-browser-deployment).

For further details on usage, visit the [TensorFlow.js official documentation](https://www.tensorflow.org/js/guide).

For more information on integrating Ultralytics YOLO26 with other platforms and frameworks, see the [integrations overview](index.md).

## FAQ

### How do I replace a TF.js export with LiteRT?

Export to LiteRT instead. `format="tfjs"` still works but emits a deprecation warning and exports the same LiteRT model:

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to LiteRT format (replaces TF.js)
        model.export(format="litert", imgsz=640)  # use imgsz=224 for classification

        # The exported '.tflite' model runs in the browser via LiteRT.js or locally with yolo predict/val.
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to LiteRT format (replaces TF.js)
        yolo export model=yolo26n.pt format=litert imgsz=640 # use imgsz=224 for classification

        # The exported '.tflite' model runs in the browser via LiteRT.js or locally with yolo predict/val.
        ```

For more details about supported export options, visit the [Ultralytics documentation page on deployment options](../guides/model-deployment-options.md).

### How does TensorFlow.js benefit browser-based machine learning applications?

TensorFlow.js is specifically designed for efficient execution of ML models in browsers and Node.js environments. Here's how it benefits browser-based applications:

- **Reduces Latency:** Runs machine learning models locally, providing immediate results without relying on server-side computations.
- **Improves Privacy:** Keeps sensitive data on the user's device, minimizing security risks.
- **Enables Offline Use:** Models can operate without an internet connection, ensuring consistent functionality.
- **Supports Multiple Backends:** Offers flexibility with backends like CPU, WebGL, WebAssembly (WASM), and WebGPU for varying computational needs.

Interested in learning more about TF.js? Check out the [official TensorFlow.js guide](https://www.tensorflow.org/js/guide).
