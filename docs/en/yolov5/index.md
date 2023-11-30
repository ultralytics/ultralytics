---
comments: true
description: Deep dive into Ultralytics' YOLOv5. Learn about object detection model - YOLOv5, how to train it on custom data, multi-GPU training and more.
keywords: Ultralytics, YOLOv5, Deep Learning, Object detection, PyTorch, Tutorial, Multi-GPU training, Custom data training
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

## Tutorials

Here's a compilation of comprehensive tutorials that will guide you through different aspects of YOLOv5.

* [Train Custom Data](tutorials/train_custom_data.md) 🚀 RECOMMENDED: Learn how to train the YOLOv5 model on your custom dataset.
* [Tips for Best Training Results](tutorials/tips_for_best_training_results.md) ☘️: Uncover practical tips to optimize your model training process.
* [Multi-GPU Training](tutorials/multi_gpu_training.md): Understand how to leverage multiple GPUs to expedite your training.
* [PyTorch Hub](tutorials/pytorch_hub_model_loading.md) 🌟 NEW: Learn to load pre-trained models via PyTorch Hub.
* [TFLite, ONNX, CoreML, TensorRT Export](tutorials/model_export.md) 🚀: Understand how to export your model to different formats.
* [NVIDIA Jetson platform Deployment](tutorials/running_on_jetson_nano.md) 🌟 NEW: Learn how to deploy your YOLOv5 model on NVIDIA Jetson platform.
* [Test-Time Augmentation (TTA)](tutorials/test_time_augmentation.md): Explore how to use TTA to improve your model's prediction accuracy.
* [Model Ensembling](tutorials/model_ensembling.md): Learn the strategy of combining multiple models for improved performance.
* [Model Pruning/Sparsity](tutorials/model_pruning_and_sparsity.md): Understand pruning and sparsity concepts, and how to create a more efficient model.
* [Hyperparameter Evolution](tutorials/hyperparameter_evolution.md): Discover the process of automated hyperparameter tuning for better model performance.
* [Transfer Learning with Frozen Layers](tutorials/transfer_learning_with_frozen_layers.md): Learn how to implement transfer learning by freezing layers in YOLOv5.
* [Architecture Summary](tutorials/architecture_description.md) 🌟 Delve into the structural details of the YOLOv5 model.
* [Roboflow for Datasets](tutorials/roboflow_datasets_integration.md): Understand how to utilize Roboflow for dataset management, labeling, and active learning.
* [ClearML Logging](tutorials/clearml_logging_integration.md) 🌟 Learn how to integrate ClearML for efficient logging during your model training.
* [YOLOv5 with Neural Magic](tutorials/neural_magic_pruning_quantization.md) Discover how to use Neural Magic's Deepsparse to prune and quantize your YOLOv5 model.
* [Comet Logging](tutorials/comet_logging_integration.md) 🌟 NEW: Explore how to utilize Comet for improved model training logging.

## Environments

YOLOv5 is designed to be run in the following up-to-date, verified environments, with all dependencies (including [CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/), and [PyTorch](https://pytorch.org/)) pre-installed:

- **Notebooks** with free GPU: <a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
- **Google Cloud** Deep Learning VM. See [GCP Quickstart Guide](environments/google_cloud_quickstart_tutorial.md)
- **Amazon** Deep Learning AMI. See [AWS Quickstart Guide](environments/aws_quickstart_tutorial.md)
- **Azure** Azure Machine Learning. See [AzureML Quickstart Guide](environments/azureml_quickstart_tutorial.md)
- **Docker Image**. See [Docker Quickstart Guide](environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

## Status

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>

This badge signifies that all [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) Continuous Integration (CI) tests are currently passing. CI tests verify the correct operation of YOLOv5 [training](https://github.com/ultralytics/yolov5/blob/master/train.py), [validation](https://github.com/ultralytics/yolov5/blob/master/val.py), [inference](https://github.com/ultralytics/yolov5/blob/master/detect.py), [export](https://github.com/ultralytics/yolov5/blob/master/export.py) and [benchmarks](https://github.com/ultralytics/yolov5/blob/master/benchmarks.py) on macOS, Windows, and Ubuntu every 24 hours and with every new commit.

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
