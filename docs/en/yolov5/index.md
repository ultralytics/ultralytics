---
comments: true
description: Deep dive into Ultralytics' YOLOv5. Learn about object detection model - YOLOv5, how to train it on custom data, multi-GPU training and more.
keywords: YOLOv5, object detection, computer vision, CUDA, PyTorch tutorial, multi-GPU training, custom dataset, model export, deployment, CI tests
---

# Comprehensive Guide to Ultralytics YOLOv5

<div align="center">
  <p>
    <a href="https://ultralytics.com/yolov5" target="_blank">
    <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov5/v70/splash.png" alt="Ultralytics YOLOv5 v7.0 banner"></a>
  </p>

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>
<a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv5 Citation"></a>
<a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>
<br>
<a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a>
<a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
<a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
<br>
<br>

Welcome to the Ultralytics' <a href="https://github.com/ultralytics/yolov5">YOLOv5</a>🚀 Documentation! YOLOv5, the fifth iteration of the revolutionary "You Only Look Once" object detection model, is designed to deliver high-speed, high-accuracy results in real-time.
<br><br>
Built on PyTorch, this powerful deep learning framework has garnered immense popularity for its versatility, ease of use, and high performance. Our documentation guides you through the installation process, explains the architectural nuances of the model, showcases various use-cases, and provides a series of detailed tutorials. These resources will help you harness the full potential of YOLOv5 for your computer vision projects. Let's get started!

</div>

## Explore and Learn

Buckle up as we embark on this learning journey through YOLOv5's extensive functionalities.

* [Train Custom Data](tutorials/train_custom_data.md) 🚀 **STRONGLY RECOMMENDED**: Elevate your models with custom dataset training.
* [Tips for Best Training Results](tutorials/tips_for_best_training_results.md) ☘️: Extract the secret sauce for peak training performance.
* [Multi-GPU Training](tutorials/multi_gpu_training.md): Harness the power of multiple GPUs to supercharge your training runs.
* [PyTorch Hub](tutorials/pytorch_hub_model_loading.md) 🌟 **NEW**: Effortless loading of pre-trained models with PyTorch Hub.
* [TFLite, ONNX, CoreML, TensorRT Export](tutorials/model_export.md) 🚀: Seamlessly transition between model formats for deployment versatility.
* [NVIDIA Jetson platform Deployment](tutorials/running_on_jetson_nano.md) 🌟 **NEW**: Deploy models on edge with NVIDIA Jetson.
* [Test-Time Augmentation (TTA)](tutorials/test_time_augmentation.md): Amplify prediction robustness with TTA.
* [Model Ensembling](tutorials/model_ensembling.md): Stack models for statistician-approved performance boosts.
* [Model Pruning/Sparsity](tutorials/model_pruning_and_sparsity.md): Craft streamlined models without compromising performance.
* [Hyperparameter Evolution](tutorials/hyperparameter_evolution.md): Fine-tune your approach with automated hyperparameter optimization.
* [Transfer Learning with Frozen Layers](tutorials/transfer_learning_with_frozen_layers.md): Accelerate training with learned knowledge and frozen layers.
* [Architecture Summary](tutorials/architecture_description.md) 🌟: Dive deep into the architectural intricacies of YOLOv5.
* [Roboflow for Datasets](tutorials/roboflow_datasets_integration.md): Enhance dataset management and augmentation with Roboflow.
* [ClearML Logging](tutorials/clearml_logging_integration.md) 🌟: Integrate your training workflow with ClearML for streamlined logging.
* [YOLOv5 with Neural Magic](tutorials/neural_magic_pruning_quantization.md): Optimize YOLOv5 with Neural Magic's pruning and quantization for performance efficiency.
* [Comet Logging](tutorials/comet_logging_integration.md) 🌟 **NEW**: Employ Comet for comprehensive training logs and analytics.

## Supported Environments

Ultralytics provides a range of ready-to-use environments, each pre-installed with essential dependencies such as [CUDA](https://developer.nvidia.com/cuda), [CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/), and [PyTorch](https://pytorch.org/), to kickstart your projects.

- **Free GPU Notebooks**: <a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
- **Google Cloud**: [GCP Quickstart Guide](environments/google_cloud_quickstart_tutorial.md)
- **Amazon**: [AWS Quickstart Guide](environments/aws_quickstart_tutorial.md)
- **Azure**: [AzureML Quickstart Guide](environments/azureml_quickstart_tutorial.md)
- **Docker**: [Docker Quickstart Guide](environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

## Project Status

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>

This badge indicates that all [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) Continuous Integration (CI) tests are successfully passing. These CI tests rigorously check the functionality and performance of YOLOv5 across various key aspects: [training](https://github.com/ultralytics/yolov5/blob/master/train.py), [validation](https://github.com/ultralytics/yolov5/blob/master/val.py), [inference](https://github.com/ultralytics/yolov5/blob/master/detect.py), [export](https://github.com/ultralytics/yolov5/blob/master/export.py), and [benchmarks](https://github.com/ultralytics/yolov5/blob/master/benchmarks.py). They ensure consistent and reliable operation on macOS, Windows, and Ubuntu, with tests conducted every 24 hours and upon each new commit.

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.instagram.com/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="3%" alt="Ultralytics Instagram"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/discord"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>

## Connect and Contribute

Your journey with YOLOv5 doesn't have to be a solitary one. Join our vibrant community on [GitHub](https://github.com/ultralytics/yolov5), connect with professionals on [LinkedIn](https://www.linkedin.com/company/ultralytics/), share your results on [Twitter](https://twitter.com/ultralytics), and find educational resources on [YouTube](https://youtube.com/ultralytics). Follow us on [TikTok](https://www.tiktok.com/@ultralytics) and [Instagram](https://www.instagram.com/ultralytics/) for more engaging content.

Interested in contributing? We welcome contributions of all forms; from code improvements and bug reports to documentation updates. Check out our [contributing guidelines](https://github.com/ultralytics/yolov5/blob/master/CONTRIBUTING.md) for more information.

We're excited to see the innovative ways you'll use YOLOv5. Dive in, experiment, and revolutionize your computer vision projects! 🚀
