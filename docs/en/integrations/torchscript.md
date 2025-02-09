---
comments: true
description: Learn how to export Ultralytics YOLO11 models to TorchScript for flexible, cross-platform deployment. Boost performance and utilize in various environments.
keywords: YOLO11, TorchScript, model export, Ultralytics, PyTorch, deep learning, AI deployment, cross-platform, performance optimization
---

# YOLO11 Model Export to TorchScript for Quick Deployment

Deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models across different environments, including embedded systems, web browsers, or platforms with limited Python support, requires a flexible and portable solution. TorchScript focuses on portability and the ability to run models in environments where the entire Python framework is unavailable. This makes it ideal for scenarios where you need to deploy your computer vision capabilities across various devices or platforms.

Export to Torchscript to serialize your [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models for cross-platform compatibility and streamlined deployment. In this guide, we'll show you how to export your YOLO11 models to the TorchScript format, making it easier for you to use them across a wider range of applications.

## Why should you export to TorchScript?

![Torchscript Overview](https://github.com/ultralytics/docs/releases/download/0/torchscript-overview.avif)

Developed by the creators of PyTorch, TorchScript is a powerful tool for optimizing and deploying PyTorch models across a variety of platforms. Exporting YOLO11 models to [TorchScript](https://pytorch.org/docs/stable/jit.html) is crucial for moving from research to real-world applications. TorchScript, part of the PyTorch framework, helps make this transition smoother by allowing PyTorch models to be used in environments that don't support Python.

The process involves two techniques: tracing and scripting. Tracing records operations during model execution, while scripting allows for the definition of models using a subset of Python. These techniques ensure that models like YOLO11 can still work their magic even outside their usual Python environment.

![TorchScript Script and Trace](https://github.com/ultralytics/docs/releases/download/0/torchscript-script-and-trace.avif)

TorchScript models can also be optimized through techniques such as operator fusion and refinements in memory usage, ensuring efficient execution. Another advantage of exporting to TorchScript is its potential to accelerate model execution across various hardware platforms. It creates a standalone, production-ready representation of your PyTorch model that can be integrated into C++ environments, embedded systems, or deployed in web or mobile applications.

## Key Features of TorchScript Models

TorchScript, a key part of the PyTorch ecosystem, provides powerful features for optimizing and deploying [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models.

![TorchScript Features](https://github.com/ultralytics/docs/releases/download/0/torchscript-features.avif)

Here are the key features that make TorchScript a valuable tool for developers:

- **Static Graph Execution**: TorchScript uses a static graph representation of the model's computation, which is different from PyTorch's dynamic graph execution. In static graph execution, the computational graph is defined and compiled once before the actual execution, resulting in improved performance during inference.

- **Model Serialization**: TorchScript allows you to serialize PyTorch models into a platform-independent format. Serialized models can be loaded without requiring the original Python code, enabling deployment in different runtime environments.

- **JIT Compilation**: TorchScript uses Just-In-Time (JIT) compilation to convert PyTorch models into an optimized intermediate representation. JIT compiles the model's computational graph, enabling efficient execution on target devices.

- **Cross-Language Integration**: With TorchScript, you can export PyTorch models to other languages such as C++, Java, and JavaScript. This makes it easier to integrate PyTorch models into existing software systems written in different languages.

- **Gradual Conversion**: TorchScript provides a gradual conversion approach, allowing you to incrementally convert parts of your PyTorch model into TorchScript. This flexibility is particularly useful when dealing with complex models or when you want to optimize specific portions of the code.

## Deployment Options in TorchScript

Before we look at the code for exporting YOLO11 models to the TorchScript format, let's understand where TorchScript models are normally used.

TorchScript offers various deployment options for [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models, such as:

- **C++ API**: The most common use case for TorchScript is its C++ API, which allows you to load and execute optimized TorchScript models directly within C++ applications. This is ideal for production environments where Python may not be suitable or available. The C++ API offers low-overhead and efficient execution of TorchScript models, maximizing performance potential.

- **Mobile Deployment**: TorchScript offers tools for converting models into formats readily deployable on mobile devices. PyTorch Mobile provides a runtime for executing these models within iOS and Android apps. This enables low-latency, offline inference capabilities, enhancing user experience and [data privacy](https://www.ultralytics.com/glossary/data-privacy).

- **Cloud Deployment**: TorchScript models can be deployed to cloud-based servers using solutions like TorchServe. It provides features like model versioning, batching, and metrics monitoring for scalable deployment in production environments. Cloud deployment with TorchScript can make your models accessible via APIs or other web services.

## Export to TorchScript: Converting Your YOLO11 Model

Exporting YOLO11 models to TorchScript makes it easier to use them in different places and helps them run faster and more efficiently. This is great for anyone looking to use deep learning models more effectively in real-world applications.

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

        # Export the model to TorchScript format
        model.export(format="torchscript")  # creates 'yolo11n.torchscript'

        # Load the exported TorchScript model
        torchscript_model = YOLO("yolo11n.torchscript")

        # Run inference
        results = torchscript_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to TorchScript format
        yolo export model=yolo11n.pt format=torchscript  # creates 'yolo11n.torchscript'

        # Run inference with the exported model
        yolo predict model=yolo11n.torchscript source='https://ultralytics.com/images/bus.jpg'
        ```

### Export Arguments

| Argument   | Type             | Default       | Description                                                                                                                             |
| ---------- | ---------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `torchscript` | Target format for the exported model, defining compatibility with various deployment environments.                                      |
| `imgsz`    | `int` or `tuple` | `640`         | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.       |
| `optimize` | `bool`           | `False`       | Applies optimization for mobile devices, potentially reducing model size and improving performance.                                     |
| `nms`      | `bool`           | `False`       | Adds Non-Maximum Suppression (NMS), essential for accurate and efficient detection post-processing.                                     |
| `batch`    | `int`            | `1`           | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode. |

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLO11 TorchScript Models

After successfully exporting your Ultralytics YOLO11 models to TorchScript format, you can now deploy them. The primary and recommended first step for running a TorchScript model is to utilize the YOLO("model.torchscript") method, as outlined in the previous usage code snippet. However, for in-depth instructions on deploying your TorchScript models in various other settings, take a look at the following resources:

- **[Explore Mobile Deployment](https://pytorch.org/mobile/home/)**: The [PyTorch](https://www.ultralytics.com/glossary/pytorch) Mobile Documentation provides comprehensive guidelines for deploying models on mobile devices, ensuring your applications are efficient and responsive.

- **[Master Server-Side Deployment](https://pytorch.org/serve/getting_started.html)**: Learn how to deploy models server-side with TorchServe, offering a step-by-step tutorial for scalable, efficient model serving.

- **[Implement C++ Deployment](https://pytorch.org/tutorials/advanced/cpp_export.html)**: Dive into the Tutorial on Loading a TorchScript Model in C++, facilitating the integration of your TorchScript models into C++ applications for enhanced performance and versatility.

## Summary

In this guide, we explored the process of exporting Ultralytics YOLO11 models to the TorchScript format. By following the provided instructions, you can optimize YOLO11 models for performance and gain the flexibility to deploy them across various platforms and environments.

For further details on usage, visit [TorchScript's official documentation](https://pytorch.org/docs/stable/jit.html).

Also, if you'd like to know more about other Ultralytics YOLO11 integrations, visit our [integration guide page](../integrations/index.md). You'll find plenty of useful resources and insights there.

## FAQ

### What is Ultralytics YOLO11 model export to TorchScript?

Exporting an Ultralytics YOLO11 model to TorchScript allows for flexible, cross-platform deployment. TorchScript, a part of the PyTorch ecosystem, facilitates the serialization of models, which can then be executed in environments that lack Python support. This makes it ideal for deploying models on embedded systems, C++ environments, mobile applications, and even web browsers. Exporting to TorchScript enables efficient performance and wider applicability of your YOLO11 models across diverse platforms.

### How can I export my YOLO11 model to TorchScript using Ultralytics?

To export a YOLO11 model to TorchScript, you can use the following example code:

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO11 model
        model = YOLO("yolo11n.pt")

        # Export the model to TorchScript format
        model.export(format="torchscript")  # creates 'yolo11n.torchscript'

        # Load the exported TorchScript model
        torchscript_model = YOLO("yolo11n.torchscript")

        # Run inference
        results = torchscript_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to TorchScript format
        yolo export model=yolo11n.pt format=torchscript  # creates 'yolo11n.torchscript'

        # Run inference with the exported model
        yolo predict model=yolo11n.torchscript source='https://ultralytics.com/images/bus.jpg'
        ```

For more details about the export process, refer to the [Ultralytics documentation on exporting](../modes/export.md).

### Why should I use TorchScript for deploying YOLO11 models?

Using TorchScript for deploying YOLO11 models offers several advantages:

- **Portability**: Exported models can run in environments without the need for Python, such as C++ applications, embedded systems, or mobile devices.
- **Optimization**: TorchScript supports static graph execution and Just-In-Time (JIT) compilation, which can optimize model performance.
- **Cross-Language Integration**: TorchScript models can be integrated into other programming languages, enhancing flexibility and expandability.
- **Serialization**: Models can be serialized, allowing for platform-independent loading and inference.

For more insights into deployment, visit the [PyTorch Mobile Documentation](https://pytorch.org/mobile/home/), [TorchServe Documentation](https://pytorch.org/serve/getting_started.html), and [C++ Deployment Guide](https://pytorch.org/tutorials/advanced/cpp_export.html).

### What are the installation steps for exporting YOLO11 models to TorchScript?

To install the required package for exporting YOLO11 models, use the following command:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO11
        pip install ultralytics
        ```

For detailed instructions, visit the [Ultralytics Installation guide](../quickstart.md). If any issues arise during installation, consult the [Common Issues guide](../guides/yolo-common-issues.md).

### How do I deploy my exported TorchScript YOLO11 models?

After exporting YOLO11 models to the TorchScript format, you can deploy them across a variety of platforms:

- **C++ API**: Ideal for low-overhead, highly efficient production environments.
- **Mobile Deployment**: Use [PyTorch Mobile](https://pytorch.org/mobile/home/) for iOS and Android applications.
- **Cloud Deployment**: Utilize services like [TorchServe](https://pytorch.org/serve/getting_started.html) for scalable server-side deployment.

Explore comprehensive guidelines for deploying models in these settings to take full advantage of TorchScript's capabilities.
