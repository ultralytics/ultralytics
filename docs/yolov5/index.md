---
comments: true
---

# Ultralytics YOLOv5

<div align="center">
  <p>
    <a href="https://ultralytics.com/yolov5" target="_blank">
    <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov5/v70/splash.png"></a>
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

Welcome to the Ultralytics YOLOv5 🚀 Docs! YOLOv5, or You Only Look Once version 5, is an Ultralytics object detection model designed to deliver fast and accurate real-time results.
<br><br>
This powerful deep learning framework is built on the PyTorch platform and has gained immense popularity due to its ease of use, high performance, and versatility. In this documentation, we will guide you through the installation process, explain the model's architecture, showcase various use-cases, and provide detailed tutorials to help you harness the full potential of YOLOv5 for your computer vision projects. Let's dive in!

</div>

## Tutorials

* [Train Custom Data](tutorials/train_custom_data.md) 🚀 RECOMMENDED
* [Tips for Best Training Results](tutorials/tips_for_best_training_results.md) ☘️
* [Multi-GPU Training](tutorials/multi_gpu_training.md)
* [PyTorch Hub](tutorials/pytorch_hub_model_loading.md) 🌟 NEW
* [TFLite, ONNX, CoreML, TensorRT Export](tutorials/model_export.md) 🚀
* [NVIDIA Jetson platform Deployment](tutorials/running_on_jetson_nano.md) 🌟 NEW
* [Test-Time Augmentation (TTA)](tutorials/test_time_augmentation.md)
* [Model Ensembling](tutorials/model_ensembling.md)
* [Model Pruning/Sparsity](tutorials/model_pruning_and_sparsity.md)
* [Hyperparameter Evolution](tutorials/hyperparameter_evolution.md)
* [Transfer Learning with Frozen Layers](tutorials/transfer_learning_with_frozen_layers.md)
* [Architecture Summary](tutorials/architecture_description.md) 🌟 NEW
* [Roboflow for Datasets, Labeling, and Active Learning](tutorials/roboflow_datasets_integration.md)
* [ClearML Logging](tutorials/clearml_logging_integration.md) 🌟 NEW
* [YOLOv5 with Neural Magic's Deepsparse](tutorials/neural_magic_pruning_quantization.md) 🌟 NEW
* [Comet Logging](tutorials/comet_logging_integration.md) 🌟 NEW

## Environments

YOLOv5 may be run in any of the following up-to-date verified environments (with all dependencies
including [CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/)
and [PyTorch](https://pytorch.org/) preinstalled):

- **Notebooks** with free
  GPU: <a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
- **Google Cloud** Deep Learning VM.
  See [GCP Quickstart Guide](environments/google_cloud_quickstart_tutorial.md)
- **Amazon** Deep Learning AMI. See [AWS Quickstart Guide](environments/aws_quickstart_tutorial.md)
- **Docker Image**.
  See [Docker Quickstart Guide](environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

## Status

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>

If this badge is green, all [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) Continuous
Integration (CI) tests are currently passing. CI tests verify correct operation of
YOLOv5 [training](https://github.com/ultralytics/yolov5/blob/master/train.py), [validation](https://github.com/ultralytics/yolov5/blob/master/val.py), [inference](https://github.com/ultralytics/yolov5/blob/master/detect.py), [export](https://github.com/ultralytics/yolov5/blob/master/export.py)
and [benchmarks](https://github.com/ultralytics/yolov5/blob/master/benchmarks.py) on macOS, Windows, and Ubuntu every 24
hours and on every commit.

<br>
<div align="center">
  <a href="https://github.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.linkedin.com/company/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://twitter.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://youtube.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.tiktok.com/@ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.instagram.com/ultralytics/" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://discord.gg/n6cFeSPZdD" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/blob/main/social/logo-social-discord.png" width="3%" alt="" /></a>
</div>