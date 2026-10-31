---
comments: true
description: Explore Ultralytics YOLO integrations for training, experiment tracking, dataset management, and deployment to ONNX, TensorRT, OpenVINO, CoreML, and edge NPUs.
keywords: Ultralytics, machine learning, ML workflows, dataset management, model training, model deployment, Roboflow, ClearML, Comet ML, DVC, MLflow, Ultralytics Platform, Ray Tune, TensorBoard, Weights & Biases, Amazon SageMaker, Paperspace Gradient, Google Colab, Neural Magic, Gradio, TorchScript, ONNX, OpenVINO, TensorRT, CoreML, TF SavedModel, TF GraphDef, TFLite Edge TPU, LiteRT, Luxonis, PaddlePaddle, NCNN, Hailo, Hailo HEF, Ambarella, CVflow, Qualcomm QNN, RKNN, edge AI
---

# Ultralytics Integrations

This page lists the tools and platforms that Ultralytics integrates with to support your [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) workflows, from dataset management and model training to experiment tracking and deployment.

<img width="1024" src="https://cdn.ul.run/i/f874ab850f33f361d01a01e9a8c98655.avif" alt="Ultralytics YOLO ecosystem and integrations">

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ZzUSXQkLbNw"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics YOLO Deployment and Integrations
</p>

## Training Integrations

- [Agent Skills](agent-skills.md): Teach AI coding agents like Claude Code, Codex, Cursor, and Gemini CLI the complete Ultralytics YOLO workflow with official skills covering datasets, training, inference, and export.

- [Albumentations](albumentations.md): Apply additional image [augmentations](https://www.ultralytics.com/glossary/data-augmentation) during Ultralytics training to improve model robustness and generalization.

- [Amazon SageMaker](amazon-sagemaker.md): Use Amazon SageMaker to build, train, and deploy Ultralytics models on one platform for the ML lifecycle.

- [ClearML](clearml.md): Automate your Ultralytics ML workflows, [track experiments](https://www.ultralytics.com/glossary/experiment-tracking), and collaborate as a team.

- [Comet ML](comet.md): Enhance your model development with Ultralytics by tracking, comparing, and optimizing your machine learning experiments.

- [DVC](dvc.md): Implement version control for your Ultralytics machine learning projects, synchronizing data, code, and models effectively.

- [Google Colab](google-colab.md): Use Google Colab to train and evaluate Ultralytics models in a cloud-based environment that supports collaboration and sharing.

- [IBM Watsonx](ibm-watsonx.md): Train and evaluate Ultralytics models in IBM watsonx.ai notebooks, alongside IBM tools for data management and model governance.

- [JupyterLab](jupyterlab.md): Use JupyterLab's interactive, customizable environment to train and evaluate Ultralytics models.

- [Kaggle](kaggle.md): Explore how you can use Kaggle to train and evaluate Ultralytics models in a cloud-based environment with pre-installed libraries, GPU support, and an active community for collaboration and sharing.

- [MLflow](mlflow.md): Streamline the entire ML lifecycle of Ultralytics models, from experimentation and reproducibility to deployment.

- [Modal](../guides/modal-quickstart.md): Run Ultralytics models on Modal's serverless cloud platform with automatic GPU provisioning, pay-per-second pricing, and automatic scaling for inference and training workloads.

- [Paperspace Gradient](paperspace.md): Use Paperspace Gradient cloud GPUs and notebooks to train, test, and deploy YOLO26 models.

- [Ray Tune](ray-tune.md): Run [hyperparameter tuning](https://www.ultralytics.com/glossary/hyperparameter-tuning) for your Ultralytics models at any scale.

- [TensorBoard](tensorboard.md): Visualize your Ultralytics training runs and monitor model metrics.

- [Ultralytics Platform](https://platform.ultralytics.com): Train, manage, and deploy Ultralytics models in the browser, and download [official pretrained models](https://platform.ultralytics.com/ultralytics/yolo26) and datasets.

- [VS Code](vscode.md): An extension for VS Code that provides code snippets to accelerate Ultralytics development workflows and offers examples to help anyone learn or get started.

- [Weights & Biases (W&B)](weights-biases.md): Monitor experiments, visualize metrics, and foster reproducibility and collaboration on Ultralytics projects.

- [YAML2ModelGraph](yaml2modelgraph.md): Generate publication-ready SVG architecture diagrams directly from Ultralytics YOLO YAML configuration files.

## Deployment Integrations

- [Ambarella](ambarella.md): Train, compress, and export Ultralytics YOLO models for Ambarella CVflow® SoCs such as the CV72 with Ambarella's Ultralytics fork, using SpongeTorch compression-aware training and the offline CVflow toolchain.

- [AMD](amd.md): Export Ultralytics YOLO to ONNX and run GPU-accelerated inference on AMD Instinct and Radeon GPUs through the ONNX Runtime MIGraphX execution provider on ROCm, with native PyTorch ROCm training also supported.

- [AMD Xilinx](xilinx.md): Export Ultralytics YOLO with `format="xilinx"` for AMD Versal AI Edge Series Gen 2 NPUs using AMD Quark and Vitis AI, with a guide to Zynq, Kria and Versal devices, DPU vs NPU accelerators and YOLO operator support.

- [Axelera](axelera.md): Explore Metis accelerators and the Voyager SDK for running Ultralytics models with efficient edge inference.

- [Core AI](coreai.md): Learn about Apple's new `.aimodel` format, how Core AI compares with Core ML, its current limitations, and how to export YOLO26 models with `format="coreai"` for the iOS 27 and macOS 27 generation.

- [CoreML](coreml.md): CoreML, developed by [Apple](https://www.apple.com/), is a framework designed for efficiently integrating machine learning models into applications across iOS, macOS, watchOS, and tvOS, using Apple's hardware for effective and secure [model deployment](https://www.ultralytics.com/glossary/model-deployment).

- [DEEPX](deepx.md): Export Ultralytics YOLO models to the DEEPX `.dxnn` format for power-efficient INT8 inference on DEEPX NPU hardware, targeting embedded and edge AI applications.

- [ExecuTorch](executorch.md): Developed by [Meta](https://about.meta.com/), ExecuTorch is PyTorch's unified solution for deploying Ultralytics YOLO models on edge devices.

- [Gradio](gradio.md): Deploy Ultralytics models with Gradio for real-time, interactive object detection demos.

- [Hailo](hailo.md): Export Ultralytics YOLO detection, segmentation, semantic segmentation, depth estimation, classification, pose, and OBB models directly to Hailo HEF; see the integration guide for validated models and targets.

- [Huawei Ascend](ascend.md): Compile Ultralytics YOLO models to the Huawei Ascend `.om` offline format with the CANN ATC compiler for FP16 inference on Atlas and OrangePi AIPro devices.

- [LiteRT](litert.md): Developed by [Google](https://www.google.com/), LiteRT (formerly TensorFlow Lite) is the on-device runtime for the `.tflite` format, deploying a single model across mobile, embedded, edge, and the browser (via LiteRT.js).

- [Luxonis OAK](luxonis.md): Convert Ultralytics YOLO models with Luxonis Hub or local Luxonis tooling and run them on-device on Luxonis OAK `RVC2` and `RVC4` cameras.

- [MNN](mnn.md): Developed by [Alibaba](https://www.alibabagroup.com/), MNN is a lightweight deep learning framework for on-device inference and training on mobile and embedded hardware.

- [NCNN](ncnn.md): Developed by [Tencent](https://www.tencent.com/), NCNN is a [neural network](https://www.ultralytics.com/glossary/neural-network-nn) inference framework optimized for mobile devices. It runs AI models directly inside apps across various mobile platforms.

- [Neural Magic](neural-magic.md): Historical guide to sparse CPU inference with the DeepSparse engine, whose community libraries are deprecated.

- [ONNX](onnx.md): An open-source format originally developed by [Microsoft](https://www.microsoft.com/) and Facebook for transferring AI models between frameworks, making Ultralytics models portable across runtimes and hardware.

- [OpenVINO](openvino.md): Intel's toolkit for optimizing and deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models on Intel CPUs, GPUs, and NPUs.

- [PaddlePaddle](paddlepaddle.md): An open-source deep learning platform by [Baidu](https://www.baidu.com/), PaddlePaddle supports AI model deployment with a focus on scalable industrial applications.

- [Qualcomm QNN](qnn.md): Compile Ultralytics YOLO models locally into the QNN (AI Engine Direct) context-binary format with the [ONNX Runtime](https://onnxruntime.ai/) QNN Execution Provider for accelerated inference on Snapdragon CPU, Adreno GPU, and Hexagon NPU hardware in mobile and edge devices.

- [Rockchip RKNN](rockchip-rknn.md): Developed by [Rockchip](https://www.rock-chips.com/), RKNN is a neural network inference framework optimized for Rockchip's hardware platforms, particularly their NPUs, for real-time AI inference on edge devices.

- [Seeed Studio reCamera](seeedstudio-recamera.md): Developed by [Seeed Studio](https://www.seeedstudio.com/), the reCamera is an edge AI device for real-time computer vision applications. Powered by the RISC-V-based SG200X processor, it runs energy-efficient AI inference, and its modular design and video processing capabilities suit use cases such as safety monitoring, environmental applications, and manufacturing.

- [SONY IMX500](sony-imx500.md): Optimize and deploy Ultralytics YOLOv8n and YOLO11n models on Raspberry Pi AI Cameras with the IMX500 sensor for fast, low-power performance.

- [TensorRT](tensorrt.md): Developed by [NVIDIA](https://www.nvidia.com/), this high-performance [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) inference framework and model format optimizes AI models for speed and efficiency on NVIDIA GPUs.

- [TF GraphDef](tf-graphdef.md): Developed by [Google](https://www.google.com/), GraphDef is TensorFlow's format for representing computation graphs, enabling optimized execution of machine learning models across diverse hardware.

- [TF SavedModel](tf-savedmodel.md): Developed by [Google](https://www.google.com/), TF SavedModel is a serialization format for [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) models, enabling sharing and deployment across a wide range of platforms, from servers to edge devices.

- [TFLite Edge TPU](edge-tpu.md): Developed by [Google](https://www.google.com/) for optimizing TensorFlow Lite models on Edge TPUs, this model format enables fast, efficient [edge computing](https://www.ultralytics.com/glossary/edge-computing).

- [TorchScript](torchscript.md): Developed as part of the [PyTorch](https://pytorch.org/) framework, TorchScript enables efficient execution and deployment of machine learning models in various production environments without the need for Python dependencies.

## Datasets Integrations

- [Roboflow](roboflow.md): Facilitate dataset labeling and management for Ultralytics models, offering annotation tools to label images.

## Export Formats

We also support a variety of model export formats for deployment in different environments. Here are the available formats:

{% include "macros/export-table.md" %}

Explore the links to learn more about each integration and how to get the most out of them with Ultralytics. See full `export` details in the [Export](../modes/export.md) page.

## Contribute to Our Integrations

We welcome community guides on integrating Ultralytics YOLO with other technologies, tools, and platforms. If you have successfully integrated YOLO with a new system or have valuable insights to share, consider [contributing to our Integrations Docs](../help/contributing.md).

By writing a guide or tutorial, you can help expand our documentation and provide real-world examples that benefit the community.

To contribute, follow the instructions in our contributing guide on how to submit a Pull Request (PR).

## FAQ

### What is Ultralytics Platform, and how does it streamline the ML workflow?

[Ultralytics Platform](https://platform.ultralytics.com) is a cloud-based platform for end-to-end machine learning workflows with Ultralytics models. You can upload and annotate datasets, train models on cloud GPUs, track experiments, and deploy YOLO models without extensive coding. The platform serves as a centralized workspace for your entire ML pipeline, from data preparation to deployment. Explore the key features in the [Ultralytics Platform docs](../platform/index.md) and get started with the [Quickstart](../platform/quickstart.md) guide.

### Can I track the performance of my Ultralytics models using MLflow?

Yes, you can. Integrating [MLflow](https://mlflow.org/) with Ultralytics models allows you to track experiments, improve reproducibility, and streamline the entire ML lifecycle. Detailed instructions for setting up this integration can be found on the [MLflow](mlflow.md) integration page. This integration is particularly useful for monitoring model metrics, comparing different training runs, and managing the ML workflow efficiently. MLflow provides a centralized platform to log parameters, metrics, and artifacts, making it easier to understand model behavior and make data-driven improvements.

### How do I deploy Ultralytics YOLO models with Gradio for interactive demos?

To deploy Ultralytics YOLO models with [Gradio](https://gradio.app/) for interactive [object detection](https://www.ultralytics.com/glossary/object-detection) demos, you can follow the steps outlined on the [Gradio](gradio.md) integration page. Gradio creates web interfaces for real-time model inference, which makes it useful for showcasing your YOLO model to both developers and end users. With a few lines of code, you can build interactive applications that demonstrate your model's performance on custom inputs.
