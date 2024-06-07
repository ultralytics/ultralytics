---
comments: true
description: Learn how to export YOLOv8 models to PaddlePaddle format for enhanced performance, flexibility, and deployment across various platforms and devices.
keywords: YOLOv8, PaddlePaddle, export models, computer vision, deep learning, model deployment, performance optimization
---

# How to Export to PaddlePaddle Format from YOLOv8 Models

Bridging the gap between developing and deploying computer vision models in real-world scenarios with varying conditions can be difficult. PaddlePaddle makes this process easier with its focus on flexibility, performance, and its capability for parallel processing in distributed environments. This means you can use your YOLOv8 computer vision models on a wide variety of devices and platforms, from smartphones to cloud-based servers.

The ability to export to PaddlePaddle model format allows you to optimize your [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) models for use within the PaddlePaddle framework. PaddlePaddle is known for facilitating industrial deployments and is a good choice for deploying computer vision applications in real-world settings across various domains.

## Why should you export to PaddlePaddle?

<p align="center">
  <img width="75%" src="https://github.com/PaddlePaddle/Paddle/blob/develop/doc/imgs/logo.png?raw=true" alt="PaddlePaddle Logo">
</p>

Developed by Baidu, [PaddlePaddle](https://www.paddlepaddle.org.cn/en) (**PArallel **D**istributed **D**eep **LE**arning) is China's first open-source deep learning platform. Unlike some frameworks built mainly for research, PaddlePaddle prioritizes ease of use and smooth integration across industries.

It offers tools and resources similar to popular frameworks like TensorFlow and PyTorch, making it accessible for developers of all experience levels. From farming and factories to service businesses, PaddlePaddle's large developer community of over 4.77 million is helping create and deploy AI applications.

By exporting your Ultralytics YOLOv8 models to PaddlePaddle format, you can tap into PaddlePaddle's strengths in performance optimization. PaddlePaddle prioritizes efficient model execution and reduced memory usage. As a result, your YOLOv8 models can potentially achieve even better performance, delivering top-notch results in practical scenarios.

## Key Features of PaddlePaddle Models

PaddlePaddle models offer a range of key features that contribute to their flexibility, performance, and scalability across diverse deployment scenarios:

- **Dynamic-to-Static Graph**: PaddlePaddle supports [dynamic-to-static compilation](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/jit/index_en.html), where models can be translated into a static computational graph. This enables optimizations that reduce runtime overhead and boost inference performance.

- **Operator Fusion**: PaddlePaddle, like TensorRT, uses [operator fusion](https://developer.nvidia.com/gtc/2020/video/s21436-vid) to streamline computation and reduce overhead. The framework minimizes memory transfers and computational steps by merging compatible operations, resulting in faster inference.

- **Quantization**: PaddlePaddle supports [quantization techniques](https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/quantization/PTQ_en.html), including post-training quantization and quantization-aware training. These techniques allow for the use of lower-precision data representations, effectively boosting performance and reducing model size.

## Deployment Options in PaddlePaddle

Before diving into the code for exporting YOLOv8 models to PaddlePaddle, let's take a look at the different deployment scenarios in which PaddlePaddle models excel.

PaddlePaddle provides a range of options, each offering a distinct balance of ease of use, flexibility, and performance:

- **Paddle Serving**: This framework simplifies the deployment of PaddlePaddle models as high-performance RESTful APIs. Paddle Serving is ideal for production environments, providing features like model versioning, online A/B testing, and scalability for handling large volumes of requests.

- **Paddle Inference API**: The Paddle Inference API gives you low-level control over model execution. This option is well-suited for scenarios where you need to integrate the model tightly within a custom application or optimize performance for specific hardware.

- **Paddle Lite**: Paddle Lite is designed for deployment on mobile and embedded devices where resources are limited. It optimizes models for smaller sizes and faster inference on ARM CPUs, GPUs, and other specialized hardware.

- **Paddle.js**:  Paddle.js enables you to deploy PaddlePaddle models directly within web browsers. Paddle.js can either load a pre-trained model or transform a model from [paddle-hub](https://github.com/PaddlePaddle/PaddleHub) with model transforming tools provided by Paddle.js. It can run in browsers that support WebGL/WebGPU/WebAssembly.

## Export to PaddlePaddle: Converting Your YOLOv8 Model

Converting YOLOv8 models to the PaddlePaddle format can improve execution flexibility and optimize performance for various deployment scenarios.

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

        # Export the model to PaddlePaddle format
        model.export(format="paddle")  # creates '/yolov8n_paddle_model'

        # Load the exported PaddlePaddle model
        paddle_model = YOLO("./yolov8n_paddle_model")

        # Run inference
        results = paddle_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLOv8n PyTorch model to PaddlePaddle format
        yolo export model=yolov8n.pt format=paddle  # creates '/yolov8n_paddle_model'

        # Run inference with the exported model
        yolo predict model='./yolov8n_paddle_model' source='https://ultralytics.com/images/bus.jpg'
        ```

For more details about supported export options, visit the [Ultralytics documentation page on deployment options](../guides/model-deployment-options.md).

## Deploying Exported YOLOv8 PaddlePaddle Models

After successfully exporting your Ultralytics YOLOv8 models to PaddlePaddle format, you can now deploy them. The primary and recommended first step for running a PaddlePaddle model is to use the YOLO("./model_paddle_model") method, as outlined in the previous usage code snippet.

However, for in-depth instructions on deploying your PaddlePaddle models in various other settings, take a look at the following resources:

- **[Paddle Serving](https://github.com/PaddlePaddle/Serving/blob/v0.9.0/README_CN.md)**: Learn how to deploy your PaddlePaddle models as performant services using Paddle Serving.

- **[Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/README_en.md)**: Explore how to optimize and deploy models on mobile and embedded devices using Paddle Lite.

- **[Paddle.js](https://github.com/PaddlePaddle/Paddle.js)**: Discover how to run PaddlePaddle models in web browsers for client-side AI using Paddle.js.

## Summary

In this guide, we explored the process of exporting Ultralytics YOLOv8 models to the PaddlePaddle format. By following these steps, you can leverage PaddlePaddle's strengths in diverse deployment scenarios, optimizing your models for different hardware and software environments.

For further details on usage, visit the [PaddlePaddle official documentation](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)

Want to explore more ways to integrate your Ultralytics YOLOv8 models? Our [integration guide page](index.md) explores various options, equipping you with valuable resources and insights.
