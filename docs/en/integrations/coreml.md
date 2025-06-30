---
comments: true
description: Learn how to export YOLO11 models to CoreML for optimized, on-device machine learning on iOS and macOS. Follow step-by-step instructions.
keywords: CoreML export, YOLO11 models, CoreML conversion, Ultralytics, iOS object detection, macOS machine learning, AI deployment, machine learning integration
---

# CoreML Export for YOLO11 Models

Deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models on Apple devices like iPhones and Macs requires a format that ensures seamless performance.

The CoreML export format allows you to optimize your [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models for efficient [object detection](https://www.ultralytics.com/glossary/object-detection) in iOS and macOS applications. In this guide, we'll walk you through the steps for converting your models to the CoreML format, making it easier for your models to perform well on Apple devices.

## CoreML

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/coreml-overview.avif" alt="CoreML Overview">
</p>

[CoreML](https://developer.apple.com/documentation/coreml) is Apple's foundational machine learning framework that builds upon Accelerate, BNNS, and Metal Performance Shaders. It provides a machine-learning model format that seamlessly integrates into iOS applications and supports tasks such as image analysis, [natural language processing](https://www.ultralytics.com/glossary/natural-language-processing-nlp), audio-to-text conversion, and sound analysis.

Applications can take advantage of Core ML without the need to have a network connection or API calls because the Core ML framework works using on-device computing. This means model inference can be performed locally on the user's device.

## Key Features of CoreML Models

Apple's CoreML framework offers robust features for on-device machine learning. Here are the key features that make CoreML a powerful tool for developers:

- **Comprehensive Model Support**: Converts and runs models from popular frameworks like TensorFlow, [PyTorch](https://www.ultralytics.com/glossary/pytorch), scikit-learn, XGBoost, and LibSVM.

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/coreml-supported-models.avif" alt="CoreML Supported Models">
</p>

- **On-device [Machine Learning](https://www.ultralytics.com/glossary/machine-learning-ml)**: Ensures data privacy and swift processing by executing models directly on the user's device, eliminating the need for network connectivity.

- **Performance and Optimization**: Uses the device's CPU, GPU, and Neural Engine for optimal performance with minimal power and memory usage. Offers tools for model compression and optimization while maintaining [accuracy](https://www.ultralytics.com/glossary/accuracy).

- **Ease of Integration**: Provides a unified format for various model types and a user-friendly API for seamless integration into apps. Supports domain-specific tasks through frameworks like Vision and Natural Language.

- **Advanced Features**: Includes on-device training capabilities for personalized experiences, asynchronous predictions for interactive ML experiences, and model inspection and validation tools.

## CoreML Deployment Options

Before we look at the code for exporting YOLO11 models to the CoreML format, let's understand where CoreML models are usually used.

CoreML offers various deployment options for machine learning models, including:

- **On-Device Deployment**: This method directly integrates CoreML models into your iOS app. It's particularly advantageous for ensuring low latency, enhanced privacy (since data remains on the device), and offline functionality. This approach, however, may be limited by the device's hardware capabilities, especially for larger and more complex models. On-device deployment can be executed in the following two ways.

    - **Embedded Models**: These models are included in the app bundle and are immediately accessible. They are ideal for small models that do not require frequent updates.

    - **Downloaded Models**: These models are fetched from a server as needed. This approach is suitable for larger models or those needing regular updates. It helps keep the app bundle size smaller.

- **Cloud-Based Deployment**: CoreML models are hosted on servers and accessed by the iOS app through API requests. This scalable and flexible option enables easy model updates without app revisions. It's ideal for complex models or large-scale apps requiring regular updates. However, it does require an internet connection and may pose latency and security issues.

## Exporting YOLO11 Models to CoreML

Exporting YOLO11 to CoreML enables optimized, on-device machine learning performance within Apple's ecosystem, offering benefits in terms of efficiency, security, and seamless integration with iOS, macOS, watchOS, and tvOS platforms.

### Installation

To install the required package, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO11
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [YOLO11 Installation guide](../quickstart.md). While installing the required packages for YOLO11, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

Before diving into the usage instructions, be sure to check out the range of [YOLO11 models offered by Ultralytics](../models/index.md). This will help you choose the most appropriate model for your project requirements.

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO11 model
        model = YOLO("yolo11n.pt")

        # Export the model to CoreML format
        model.export(format="coreml")  # creates 'yolo11n.mlpackage'

        # Load the exported CoreML model
        coreml_model = YOLO("yolo11n.mlpackage")

        # Run inference
        results = coreml_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to CoreML format
        yolo export model=yolo11n.pt format=coreml # creates 'yolo11n.mlpackage''

        # Run inference with the exported model
        yolo predict model=yolo11n.mlpackage source='https://ultralytics.com/images/bus.jpg'
        ```

### Export Arguments

| Argument | Type             | Default    | Description                                                                                                                                                                                   |
| -------- | ---------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'coreml'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                            |
| `imgsz`  | `int` or `tuple` | `640`      | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                             |
| `half`   | `bool`           | `False`    | Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.                                                                  |
| `int8`   | `bool`           | `False`    | Activates INT8 quantization, further compressing the model and speeding up inference with minimal [accuracy](https://www.ultralytics.com/glossary/accuracy) loss, primarily for edge devices. |
| `nms`    | `bool`           | `False`    | Adds Non-Maximum Suppression (NMS), essential for accurate and efficient detection post-processing.                                                                                           |
| `batch`  | `int`            | `1`        | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                       |
| `device` | `str`            | `None`     | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`).                                                                               |

!!! tip

    Please make sure to use a macOS or x86 Linux machine when exporting to CoreML.

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLO11 CoreML Models

Having successfully exported your Ultralytics YOLO11 models to CoreML, the next critical phase is deploying these models effectively. For detailed guidance on deploying CoreML models in various environments, check out these resources:

- **[CoreML Tools](https://apple.github.io/coremltools/docs-guides/)**: This guide includes instructions and examples to convert models from [TensorFlow](https://www.ultralytics.com/glossary/tensorflow), PyTorch, and other libraries to Core ML.

- **[ML and Vision](https://developer.apple.com/videos/)**: A collection of comprehensive videos that cover various aspects of using and implementing CoreML models.

- **[Integrating a Core ML Model into Your App](https://developer.apple.com/documentation/coreml/integrating-a-core-ml-model-into-your-app)**: A comprehensive guide on integrating a CoreML model into an iOS application, detailing steps from preparing the model to implementing it in the app for various functionalities.

## Summary

In this guide, we went over how to export Ultralytics YOLO11 models to CoreML format. By following the steps outlined in this guide, you can ensure maximum compatibility and performance when exporting YOLO11 models to CoreML.

For further details on usage, visit the [CoreML official documentation](https://developer.apple.com/documentation/coreml).

Also, if you'd like to know more about other Ultralytics YOLO11 integrations, visit our [integration guide page](../integrations/index.md). You'll find plenty of valuable resources and insights there.

## FAQ

### How do I export YOLO11 models to CoreML format?

To export your [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models to CoreML format, you'll first need to ensure you have the `ultralytics` package installed. You can install it using:

!!! example "Installation"

    === "CLI"

        ```bash
        pip install ultralytics
        ```

Next, you can export the model using the following Python or CLI commands:

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        model.export(format="coreml")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=coreml
        ```

For further details, refer to the [Exporting YOLO11 Models to CoreML](../modes/export.md) section of our documentation.

### What are the benefits of using CoreML for deploying YOLO11 models?

CoreML provides numerous advantages for deploying [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models on Apple devices:

- **On-device Processing**: Enables local model inference on devices, ensuring [data privacy](https://www.ultralytics.com/glossary/data-privacy) and minimizing latency.
- **Performance Optimization**: Leverages the full potential of the device's CPU, GPU, and Neural Engine, optimizing both speed and efficiency.
- **Ease of Integration**: Offers a seamless integration experience with Apple's ecosystems, including iOS, macOS, watchOS, and tvOS.
- **Versatility**: Supports a wide range of machine learning tasks such as image analysis, audio processing, and natural language processing using the CoreML framework.

For more details on integrating your CoreML model into an iOS app, check out the guide on [Integrating a Core ML Model into Your App](https://developer.apple.com/documentation/coreml/integrating-a-core-ml-model-into-your-app).

### What are the deployment options for YOLO11 models exported to CoreML?

Once you export your YOLO11 model to CoreML format, you have multiple deployment options:

1. **On-Device Deployment**: Directly integrate CoreML models into your app for enhanced privacy and offline functionality. This can be done as:

    - **Embedded Models**: Included in the app bundle, accessible immediately.
    - **Downloaded Models**: Fetched from a server as needed, keeping the app bundle size smaller.

2. **Cloud-Based Deployment**: Host CoreML models on servers and access them via API requests. This approach supports easier updates and can handle more complex models.

For detailed guidance on deploying CoreML models, refer to [CoreML Deployment Options](#coreml-deployment-options).

### How does CoreML ensure optimized performance for YOLO11 models?

CoreML ensures optimized performance for [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models by utilizing various optimization techniques:

- **Hardware Acceleration**: Uses the device's CPU, GPU, and Neural Engine for efficient computation.
- **Model Compression**: Provides tools for compressing models to reduce their footprint without compromising accuracy.
- **Adaptive Inference**: Adjusts inference based on the device's capabilities to maintain a balance between speed and performance.

For more information on performance optimization, visit the [CoreML official documentation](https://developer.apple.com/documentation/coreml).

### Can I run inference directly with the exported CoreML model?

Yes, you can run inference directly using the exported CoreML model. Below are the commands for Python and CLI:

!!! example "Running Inference"

    === "Python"

        ```python
        from ultralytics import YOLO

        coreml_model = YOLO("yolo11n.mlpackage")
        results = coreml_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo11n.mlpackage source='https://ultralytics.com/images/bus.jpg'
        ```

For additional information, refer to the [Usage section](#usage) of the CoreML export guide.
