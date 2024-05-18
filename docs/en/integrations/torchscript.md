---
comments: true
description: Learn to export your Ultralytics YOLOv8 models to TorchScript format for deployment through platforms like embedded systems, web browsers, and C++ applications.
keywords: Ultralytics, YOLOv8, Export to Torchscript, Model Optimization, Deployment, PyTorch, C++, Faster Inference
---

# YOLOv8 Model Export to TorchScript for Quick Deployment

Deploying computer vision models across different environments, including embedded systems, web browsers, or platforms with limited Python support, requires a flexible and portable solution. TorchScript focuses on portability and the ability to run models in environments where the entire Python framework is unavailable. This makes it ideal for scenarios where you need to deploy your computer vision capabilities across various devices or platforms.

Export to Torchscript to serialize your [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) models for cross-platform compatibility and streamlined deployment. In this guide, we'll show you how to export your YOLOv8 models to the TorchScript format, making it easier for you to use them across a wider range of applications.

## Why should you export to TorchScript?

![Torchscript Overview](https://github.com/ultralytics/ultralytics/assets/26833433/6873349d-c2f6-4620-b3cc-7b26b0698d0b)

Developed by the creators of PyTorch, TorchScript is a powerful tool for optimizing and deploying PyTorch models across a variety of platforms. Exporting YOLOv8 models to [TorchScript](https://pytorch.org/docs/stable/jit.html) is crucial for moving from research to real-world applications. TorchScript, part of the PyTorch framework, helps make this transition smoother by allowing PyTorch models to be used in environments that don't support Python.

The process involves two techniques: tracing and scripting. Tracing records operations during model execution, while scripting allows for the definition of models using a subset of Python. These techniques ensure that models like YOLOv8 can still work their magic even outside their usual Python environment.

![TorchScript Script and Trace](https://github.com/ultralytics/ultralytics/assets/26833433/ea9ea24f-a3a9-44bb-aca7-9c358d7490d7)

TorchScript models can also be optimized through techniques such as operator fusion and refinements in memory usage, ensuring efficient execution. Another advantage of exporting to TorchScript is its potential to accelerate model execution across various hardware platforms. It creates a standalone, production-ready representation of your PyTorch model that can be integrated into C++ environments, embedded systems, or deployed in web or mobile applications.

## Key Features of TorchScript Models

TorchScript, a key part of the PyTorch ecosystem, provides powerful features for optimizing and deploying deep learning models.

![TorchScript Features](https://github.com/ultralytics/ultralytics/assets/26833433/44c7c5e3-1146-42db-952a-9060f070fead)

Here are the key features that make TorchScript a valuable tool for developers:

- **Static Graph Execution**: TorchScript uses a static graph representation of the model’s computation, which is different from PyTorch’s dynamic graph execution. In static graph execution, the computational graph is defined and compiled once before the actual execution, resulting in improved performance during inference.

- **Model Serialization**: TorchScript allows you to serialize PyTorch models into a platform-independent format. Serialized models can be loaded without requiring the original Python code, enabling deployment in different runtime environments.

- **JIT Compilation**: TorchScript uses Just-In-Time (JIT) compilation to convert PyTorch models into an optimized intermediate representation. JIT compiles the model’s computational graph, enabling efficient execution on target devices.

- **Cross-Language Integration**: With TorchScript, you can export PyTorch models to other languages such as C++, Java, and JavaScript. This makes it easier to integrate PyTorch models into existing software systems written in different languages.

- **Gradual Conversion**: TorchScript provides a gradual conversion approach, allowing you to incrementally convert parts of your PyTorch model into TorchScript. This flexibility is particularly useful when dealing with complex models or when you want to optimize specific portions of the code.

## Deployment Options in TorchScript

Before we look at the code for exporting YOLOv8 models to the TorchScript format, let’s understand where TorchScript models are normally used.

TorchScript offers various deployment options for machine learning models, such as:

- **C++ API**: The most common use case for TorchScript is its C++ API, which allows you to load and execute optimized TorchScript models directly within C++ applications. This is ideal for production environments where Python may not be suitable or available. The C++ API offers low-overhead and efficient execution of TorchScript models, maximizing performance potential.

- **Mobile Deployment**: TorchScript offers tools for converting models into formats readily deployable on mobile devices. PyTorch Mobile provides a runtime for executing these models within iOS and Android apps. This enables low-latency, offline inference capabilities, enhancing user experience and data privacy.

- **Cloud Deployment**: TorchScript models can be deployed to cloud-based servers using solutions like TorchServe. It provides features like model versioning, batching, and metrics monitoring for scalable deployment in production environments. Cloud deployment with TorchScript can make your models accessible via APIs or other web services.

## Export to TorchScript: Converting Your YOLOv8 Model

Exporting YOLOv8 models to TorchScript makes it easier to use them in different places and helps them run faster and more efficiently. This is great for anyone looking to use deep learning models more effectively in real-world applications.

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

        # Export the model to TorchScript format
        model.export(format="torchscript")  # creates 'yolov8n.torchscript'

        # Load the exported TorchScript model
        torchscript_model = YOLO("yolov8n.torchscript")

        # Run inference
        results = torchscript_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLOv8n PyTorch model to TorchScript format
        yolo export model=yolov8n.pt format=torchscript  # creates 'yolov8n.torchscript'

        # Run inference with the exported model
        yolo predict model=yolov8n.torchscript source='https://ultralytics.com/images/bus.jpg'
        ```

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLOv8 TorchScript Models

After successfully exporting your Ultralytics YOLOv8 models to TorchScript format, you can now deploy them. The primary and recommended first step for running a TorchScript model is to utilize the YOLO("model.torchscript") method, as outlined in the previous usage code snippet. However, for in-depth instructions on deploying your TorchScript models in various other settings, take a look at the following resources:

- **[Explore Mobile Deployment](https://pytorch.org/mobile/home/)**: The PyTorch Mobile Documentation provides comprehensive guidelines for deploying models on mobile devices, ensuring your applications are efficient and responsive.

- **[Master Server-Side Deployment](https://pytorch.org/serve/getting_started.html)**: Learn how to deploy models server-side with TorchServe, offering a step-by-step tutorial for scalable, efficient model serving.

- **[Implement C++ Deployment](https://pytorch.org/tutorials/advanced/cpp_export.html)**: Dive into the Tutorial on Loading a TorchScript Model in C++, facilitating the integration of your TorchScript models into C++ applications for enhanced performance and versatility.

## Summary

In this guide, we explored the process of exporting Ultralytics YOLOv8 models to the TorchScript format. By following the provided instructions, you can optimize YOLOv8 models for performance and gain the flexibility to deploy them across various platforms and environments.

For further details on usage, visit [TorchScript’s official documentation](https://pytorch.org/docs/stable/jit.html).

Also, if you’d like to know more about other Ultralytics YOLOv8 integrations, visit our [integration guide page](../integrations/index.md). You'll find plenty of useful resources and insights there.
