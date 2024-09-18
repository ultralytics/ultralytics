---
comments: true
description: Learn how to export Ultralytics YOLOv8 models to TensorFlow SavedModel format for easy deployment across various platforms and environments.
keywords: YOLOv8, TF SavedModel, Ultralytics, TensorFlow, model export, model deployment, machine learning, AI
---

# Understand How to Export to TF SavedModel Format From YOLOv8

Deploying machine learning models can be challenging. However, using an efficient and flexible model format can make your job easier. TF SavedModel is an open-source machine-learning framework used by TensorFlow to load machine-learning models in a consistent way. It is like a suitcase for TensorFlow models, making them easy to carry and use on different devices and systems.

Learning how to export to TF SavedModel from [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) models can help you deploy models easily across different platforms and environments. In this guide, we'll walk through how to convert your models to the TF SavedModel format, simplifying the process of running inferences with your models on different devices.

## Why Should You Export to TF SavedModel?

The TensorFlow SavedModel format is a part of the TensorFlow ecosystem developed by Google as shown below. It is designed to save and serialize TensorFlow models seamlessly. It encapsulates the complete details of models like the architecture, weights, and even compilation information. This makes it straightforward to share, deploy, and continue training across different environments.

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/tf-savedmodel-overview.avif" alt="TF SavedModel">
</p>

The TF SavedModel has a key advantage: its compatibility. It works well with TensorFlow Serving, TensorFlow Lite, and TensorFlow.js. This compatibility makes it easier to share and deploy models across various platforms, including web and mobile applications. The TF SavedModel format is useful both for research and production. It provides a unified way to manage your models, ensuring they are ready for any application.

## Key Features of TF SavedModels

Here are the key features that make TF SavedModel a great option for AI developers:

- **Portability**: TF SavedModel provides a language-neutral, recoverable, hermetic serialization format. They enable higher-level systems and tools to produce, consume, and transform TensorFlow models. SavedModels can be easily shared and deployed across different platforms and environments.

- **Ease of Deployment**: TF SavedModel bundles the computational graph, trained parameters, and necessary metadata into a single package. They can be easily loaded and used for inference without requiring the original code that built the model. This makes the deployment of TensorFlow models straightforward and efficient in various production environments.

- **Asset Management**: TF SavedModel supports the inclusion of external assets such as vocabularies, embeddings, or lookup tables. These assets are stored alongside the graph definition and variables, ensuring they are available when the model is loaded. This feature simplifies the management and distribution of models that rely on external resources.

## Deployment Options with TF SavedModel

Before we dive into the process of exporting YOLOv8 models to the TF SavedModel format, let's explore some typical deployment scenarios where this format is used.

TF SavedModel provides a range of options to deploy your machine learning models:

- **TensorFlow Serving:** TensorFlow Serving is a flexible, high-performance serving system designed for production environments. It natively supports TF SavedModels, making it easy to deploy and serve your models on cloud platforms, on-premises servers, or edge devices.

- **Cloud Platforms:** Major cloud providers like Google Cloud Platform (GCP), Amazon Web Services (AWS), and Microsoft Azure offer services for deploying and running TensorFlow models, including TF SavedModels. These services provide scalable and managed infrastructure, allowing you to deploy and scale your models easily.

- **Mobile and Embedded Devices:** TensorFlow Lite, a lightweight solution for running machine learning models on mobile, embedded, and IoT devices, supports converting TF SavedModels to the TensorFlow Lite format. This allows you to deploy your models on a wide range of devices, from smartphones and tablets to microcontrollers and edge devices.

- **TensorFlow Runtime:** TensorFlow Runtime (`tfrt`) is a high-performance runtime for executing TensorFlow graphs. It provides lower-level APIs for loading and running TF SavedModels in C++ environments. TensorFlow Runtime offers better performance compared to the standard TensorFlow runtime. It is suitable for deployment scenarios that require low-latency inference and tight integration with existing C++ codebases.

## Exporting YOLOv8 Models to TF SavedModel

By exporting YOLOv8 models to the TF SavedModel format, you enhance their adaptability and ease of deployment across various platforms.

### Installation

To install the required package, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLOv8
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLOv8, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

Before diving into the usage instructions, it's important to note that while all [Ultralytics YOLOv8 models](../models/index.md) are available for exporting, you can ensure that the model you select supports export functionality [here](../modes/export.md).

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLOv8 model
        model = YOLO("yolov8n.pt")

        # Export the model to TF SavedModel format
        model.export(format="saved_model")  # creates '/yolov8n_saved_model'

        # Load the exported TF SavedModel model
        tf_savedmodel_model = YOLO("./yolov8n_saved_model")

        # Run inference
        results = tf_savedmodel_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLOv8n PyTorch model to TF SavedModel format
        yolo export model=yolov8n.pt format=saved_model  # creates '/yolov8n_saved_model'

        # Run inference with the exported model
        yolo predict model='./yolov8n_saved_model' source='https://ultralytics.com/images/bus.jpg'
        ```

For more details about supported export options, visit the [Ultralytics documentation page on deployment options](../guides/model-deployment-options.md).

## Deploying Exported YOLOv8 TF SavedModel Models

Now that you have exported your YOLOv8 model to the TF SavedModel format, the next step is to deploy it. The primary and recommended first step for running a TF GraphDef model is to use the YOLO("./yolov8n_saved_model") method, as previously shown in the usage code snippet.

However, for in-depth instructions on deploying your TF SavedModel models, take a look at the following resources:

- **[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)**: Here's the developer documentation for how to deploy your TF SavedModel models using TensorFlow Serving.

- **[Run a TensorFlow SavedModel in Node.js](https://blog.tensorflow.org/2020/01/run-tensorflow-savedmodel-in-nodejs-directly-without-conversion.html)**: A TensorFlow blog post on running a TensorFlow SavedModel in Node.js directly without conversion.

- **[Deploying on Cloud](https://blog.tensorflow.org/2020/04/how-to-deploy-tensorflow-2-models-on-cloud-ai-platform.html)**: A TensorFlow blog post on deploying a TensorFlow SavedModel model on the Cloud AI Platform.

## Summary

In this guide, we explored how to export Ultralytics YOLOv8 models to the TF SavedModel format. By exporting to TF SavedModel, you gain the flexibility to optimize, deploy, and scale your YOLOv8 models on a wide range of platforms.

For further details on usage, visit the [TF SavedModel official documentation](https://www.tensorflow.org/guide/saved_model).

For more information on integrating Ultralytics YOLOv8 with other platforms and frameworks, don't forget to check out our [integration guide page](index.md). It's packed with great resources to help you make the most of YOLOv8 in your projects.

## FAQ

### How do I export an Ultralytics YOLO model to TensorFlow SavedModel format?

Exporting an Ultralytics YOLO model to the TensorFlow SavedModel format is straightforward. You can use either Python or CLI to achieve this:

!!! example "Exporting YOLOv8 to TF SavedModel"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLOv8 model
        model = YOLO("yolov8n.pt")

        # Export the model to TF SavedModel format
        model.export(format="saved_model")  # creates '/yolov8n_saved_model'

        # Load the exported TF SavedModel for inference
        tf_savedmodel_model = YOLO("./yolov8n_saved_model")
        results = tf_savedmodel_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export the YOLOv8 model to TF SavedModel format
        yolo export model=yolov8n.pt format=saved_model  # creates '/yolov8n_saved_model'

        # Run inference with the exported model
        yolo predict model='./yolov8n_saved_model' source='https://ultralytics.com/images/bus.jpg'
        ```

Refer to the [Ultralytics Export documentation](../modes/export.md) for more details.

### Why should I use the TensorFlow SavedModel format?

The TensorFlow SavedModel format offers several advantages for model deployment:

- **Portability:** It provides a language-neutral format, making it easy to share and deploy models across different environments.
- **Compatibility:** Integrates seamlessly with tools like TensorFlow Serving, TensorFlow Lite, and TensorFlow.js, which are essential for deploying models on various platforms, including web and mobile applications.
- **Complete encapsulation:** Encodes the model architecture, weights, and compilation information, allowing for straightforward sharing and training continuation.

For more benefits and deployment options, check out the [Ultralytics YOLO model deployment options](../guides/model-deployment-options.md).

### What are the typical deployment scenarios for TF SavedModel?

TF SavedModel can be deployed in various environments, including:

- **TensorFlow Serving:** Ideal for production environments requiring scalable and high-performance model serving.
- **Cloud Platforms:** Supports major cloud services like Google Cloud Platform (GCP), Amazon Web Services (AWS), and Microsoft Azure for scalable model deployment.
- **Mobile and Embedded Devices:** Using TensorFlow Lite to convert TF SavedModels allows for deployment on mobile devices, IoT devices, and microcontrollers.
- **TensorFlow Runtime:** For C++ environments needing low-latency inference with better performance.

For detailed deployment options, visit the official guides on [deploying TensorFlow models](https://www.tensorflow.org/tfx/guide/serving).

### How can I install the necessary packages to export YOLOv8 models?

To export YOLOv8 models, you need to install the `ultralytics` package. Run the following command in your terminal:

```bash
pip install ultralytics
```

For more detailed installation instructions and best practices, refer to our [Ultralytics Installation guide](../quickstart.md). If you encounter any issues, consult our [Common Issues guide](../guides/yolo-common-issues.md).

### What are the key features of the TensorFlow SavedModel format?

TF SavedModel format is beneficial for AI developers due to the following features:

- **Portability:** Allows sharing and deployment across various environments effortlessly.
- **Ease of Deployment:** Encapsulates the computational graph, trained parameters, and metadata into a single package, which simplifies loading and inference.
- **Asset Management:** Supports external assets like vocabularies, ensuring they are available when the model loads.

For further details, explore the [official TensorFlow documentation](https://www.tensorflow.org/guide/saved_model).
