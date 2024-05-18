---
comments: true
description: Explore the process of exporting Ultralytics YOLOv8 models to CoreML format, enabling efficient object detection capabilities for iOS and macOS applications on Apple devices.
keywords: Ultralytics, YOLOv8, CoreML Export, Model Deployment, Apple Devices, Object Detection, Machine Learning
---

# CoreML Export for YOLOv8 Models

Deploying computer vision models on Apple devices like iPhones and Macs requires a format that ensures seamless performance.

The CoreML export format allows you to optimize your [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) models for efficient object detection in iOS and macOS applications. In this guide, we'll walk you through the steps for converting your models to the CoreML format, making it easier for your models to perform well on Apple devices.

## CoreML

<p align="center">
  <img width="100%" src="https://github.com/RizwanMunawar/ultralytics/assets/62513924/0c757e32-3a9f-422e-9526-efde5f663ccd" alt="CoreML Overview">
</p>

[CoreML](https://developer.apple.com/documentation/coreml) is Apple's foundational machine learning framework that builds upon Accelerate, BNNS, and Metal Performance Shaders. It provides a machine-learning model format that seamlessly integrates into iOS applications and supports tasks such as image analysis, natural language processing, audio-to-text conversion, and sound analysis.

Applications can take advantage of Core ML without the need to have a network connection or API calls because the Core ML framework works using on-device computing. This means model inference can be performed locally on the user's device.

## Key Features of CoreML Models

Apple's CoreML framework offers robust features for on-device machine learning. Here are the key features that make CoreML a powerful tool for developers:

- **Comprehensive Model Support**: Converts and runs models from popular frameworks like TensorFlow, PyTorch, scikit-learn, XGBoost, and LibSVM.

<p align="center">
  <img width="100%" src="https://apple.github.io/coremltools/docs-guides/_images/introduction-coremltools.png" alt="CoreML Supported Models">
</p>

- **On-device Machine Learning**: Ensures data privacy and swift processing by executing models directly on the user's device, eliminating the need for network connectivity.

- **Performance and Optimization**: Uses the device's CPU, GPU, and Neural Engine for optimal performance with minimal power and memory usage. Offers tools for model compression and optimization while maintaining accuracy.

- **Ease of Integration**: Provides a unified format for various model types and a user-friendly API for seamless integration into apps. Supports domain-specific tasks through frameworks like Vision and Natural Language.

- **Advanced Features**: Includes on-device training capabilities for personalized experiences, asynchronous predictions for interactive ML experiences, and model inspection and validation tools.

## CoreML Deployment Options

Before we look at the code for exporting YOLOv8 models to the CoreML format, let’s understand where CoreML models are usually used.

CoreML offers various deployment options for machine learning models, including:

- **On-Device Deployment**: This method directly integrates CoreML models into your iOS app. It's particularly advantageous for ensuring low latency, enhanced privacy (since data remains on the device), and offline functionality. This approach, however, may be limited by the device's hardware capabilities, especially for larger and more complex models. On-device deployment can be executed in the following two ways.

    - **Embedded Models**: These models are included in the app bundle and are immediately accessible. They are ideal for small models that do not require frequent updates.

    - **Downloaded Models**: These models are fetched from a server as needed. This approach is suitable for larger models or those needing regular updates. It helps keep the app bundle size smaller.

- **Cloud-Based Deployment**: CoreML models are hosted on servers and accessed by the iOS app through API requests. This scalable and flexible option enables easy model updates without app revisions. It’s ideal for complex models or large-scale apps requiring regular updates. However, it does require an internet connection and may pose latency and security issues​.

## Exporting YOLOv8 Models to CoreML

Exporting YOLOv8 to CoreML enables optimized, on-device machine learning performance within Apple's ecosystem, offering benefits in terms of efficiency, security, and seamless integration with iOS, macOS, watchOS, and tvOS platforms.

### Installation

To install the required package, run:

!!! Tip "Installation"

    === "CLI"
    
        ```bash
        # Install the required package for YOLOv8
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [YOLOv8 Installation guide](../quickstart.md). While installing the required packages for YOLOv8, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

Before diving into the usage instructions, be sure to check out the range of [YOLOv8 models offered by Ultralytics](../models/index.md). This will help you choose the most appropriate model for your project requirements.

!!! Example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLOv8 model
        model = YOLO("yolov8n.pt")

        # Export the model to CoreML format
        model.export(format="coreml")  # creates 'yolov8n.mlpackage'

        # Load the exported CoreML model
        coreml_model = YOLO("yolov8n.mlpackage")

        # Run inference
        results = coreml_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLOv8n PyTorch model to CoreML format
        yolo export model=yolov8n.pt format=coreml  # creates 'yolov8n.mlpackage''

        # Run inference with the exported model
        yolo predict model=yolov8n.mlpackage source='https://ultralytics.com/images/bus.jpg'
        ```

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLOv8 CoreML Models

Having successfully exported your Ultralytics YOLOv8 models to CoreML, the next critical phase is deploying these models effectively. For detailed guidance on deploying CoreML models in various environments, check out these resources:

- **[CoreML Tools](https://apple.github.io/coremltools/docs-guides/)**: This guide includes instructions and examples to convert models from TensorFlow, PyTorch, and other libraries to Core ML.

- **[ML and Vision](https://developer.apple.com/videos/ml-vision)**: A collection of comprehensive videos that cover various aspects of using and implementing CoreML models.

- **[Integrating a Core ML Model into Your App](https://developer.apple.com/documentation/coreml/integrating_a_core_ml_model_into_your_app)**: A comprehensive guide on integrating a CoreML model into an iOS application, detailing steps from preparing the model to implementing it in the app for various functionalities.

## Summary

In this guide, we went over how to export Ultralytics YOLOv8 models to CoreML format. By following the steps outlined in this guide, you can ensure maximum compatibility and performance when exporting YOLOv8 models to CoreML.

For further details on usage, visit the [CoreML official documentation](https://developer.apple.com/documentation/coreml).

Also, if you’d like to know more about other Ultralytics YOLOv8 integrations, visit our [integration guide page](../integrations/index.md). You'll find plenty of valuable resources and insights there.
