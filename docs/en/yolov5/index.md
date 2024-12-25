---
comments: true
description: Explore comprehensive YOLOv5 documentation with step-by-step tutorials on training, deployment, and model optimization. Empower your vision projects today!.
keywords: YOLOv5, Ultralytics, object detection, computer vision, deep learning, AI, tutorials, PyTorch, model optimization, machine learning, neural networks
---

# Comprehensive Guide to Ultralytics YOLOv5

<div align="center">
  <p>
    <a href="https://www.ultralytics.com/yolo" target="_blank">
    <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov5-splash.avif" alt="Ultralytics YOLOv5 v7.0 banner"></a>
  </p>

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>
<a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv5 Citation"></a>
<a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>
<br>
<a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a>
<a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
<a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
<br>
<br>

Welcome to the Ultralytics' <a href="https://github.com/ultralytics/yolov5">YOLOv5</a>🚀 Documentation! YOLOv5, the fifth iteration of the revolutionary "You Only Look Once" <a href="https://www.ultralytics.com/glossary/object-detection">object detection</a> model, is designed to deliver high-speed, high-accuracy results in real-time.

<br><br>

Built on PyTorch, this powerful <a href="https://www.ultralytics.com/glossary/deep-learning-dl">deep learning</a> framework has garnered immense popularity for its versatility, ease of use, and high performance. Our documentation guides you through the installation process, explains the architectural nuances of the model, showcases various use-cases, and provides a series of detailed tutorials. These resources will help you harness the full potential of YOLOv5 for your <a href="https://www.ultralytics.com/glossary/computer-vision-cv">computer vision</a> projects. Let's get started!

</div>

## Explore and Learn

Here's a compilation of comprehensive tutorials that will guide you through different aspects of YOLOv5.

- [Train Custom Data](tutorials/train_custom_data.md) 🚀 RECOMMENDED: Learn how to train the YOLOv5 model on your custom dataset.
- [Tips for Best Training Results](tutorials/tips_for_best_training_results.md) ☘️: Uncover practical tips to optimize your model training process.
- [Multi-GPU Training](tutorials/multi_gpu_training.md): Understand how to leverage multiple GPUs to expedite your training.
- [PyTorch Hub](tutorials/pytorch_hub_model_loading.md) 🌟 NEW: Learn to load pre-trained models via PyTorch Hub.
- [TFLite, ONNX, CoreML, TensorRT Export](tutorials/model_export.md) 🚀: Understand how to export your model to different formats.
- [Test-Time Augmentation (TTA)](tutorials/test_time_augmentation.md): Explore how to use TTA to improve your model's prediction accuracy.
- [Model Ensembling](tutorials/model_ensembling.md): Learn the strategy of combining multiple models for improved performance.
- [Model Pruning/Sparsity](tutorials/model_pruning_and_sparsity.md): Understand pruning and sparsity concepts, and how to create a more efficient model.
- [Hyperparameter Evolution](tutorials/hyperparameter_evolution.md): Discover the process of automated [hyperparameter tuning](https://www.ultralytics.com/glossary/hyperparameter-tuning) for better model performance.
- [Transfer Learning with Frozen Layers](tutorials/transfer_learning_with_frozen_layers.md): Learn how to implement [transfer learning](https://www.ultralytics.com/glossary/transfer-learning) by freezing layers in YOLOv5.
- [Architecture Summary](tutorials/architecture_description.md) 🌟 Delve into the structural details of the YOLOv5 model.
- [Roboflow for Datasets](tutorials/roboflow_datasets_integration.md): Understand how to utilize Roboflow for dataset management, labeling, and [active learning](https://www.ultralytics.com/glossary/active-learning).
- [ClearML Logging](tutorials/clearml_logging_integration.md) 🌟 Learn how to integrate ClearML for efficient logging during your model training.
- [YOLOv5 with Neural Magic](tutorials/neural_magic_pruning_quantization.md) Discover how to use Neural Magic's Deepsparse to prune and quantize your YOLOv5 model.
- [Comet Logging](tutorials/comet_logging_integration.md) 🌟 NEW: Explore how to utilize Comet for improved model training logging.

## Supported Environments

Ultralytics provides a range of ready-to-use environments, each pre-installed with essential dependencies such as [CUDA](https://developer.nvidia.com/cuda-zone), [CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/), and [PyTorch](https://pytorch.org/), to kickstart your projects.

- **Free GPU Notebooks**: <a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
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
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>

## Connect and Contribute

Your journey with YOLOv5 doesn't have to be a solitary one. Join our vibrant community on [GitHub](https://github.com/ultralytics/yolov5), connect with professionals on [LinkedIn](https://www.linkedin.com/company/ultralytics/), share your results on [Twitter](https://twitter.com/ultralytics), and find educational resources on [YouTube](https://www.youtube.com/ultralytics?sub_confirmation=1). Follow us on [TikTok](https://www.tiktok.com/@ultralytics) and [BiliBili](https://ultralytics.com/bilibili) for more engaging content.

Interested in contributing? We welcome contributions of all forms; from code improvements and bug reports to documentation updates. Check out our [contributing guidelines](../help/contributing.md) for more information.

We're excited to see the innovative ways you'll use YOLOv5. Dive in, experiment, and revolutionize your computer vision projects! 🚀

## FAQ

### What are the key features of Ultralytics YOLOv5?

Ultralytics YOLOv5 is renowned for its high-speed and high-[accuracy](https://www.ultralytics.com/glossary/accuracy) object detection capabilities. Built on [PyTorch](https://www.ultralytics.com/glossary/pytorch), it is versatile and user-friendly, making it suitable for various computer vision projects. Key features include real-time inference, support for multiple training tricks like Test-Time Augmentation (TTA) and Model Ensembling, and compatibility with export formats such as TFLite, ONNX, CoreML, and TensorRT. To delve deeper into how Ultralytics YOLOv5 can elevate your project, explore our [TFLite, ONNX, CoreML, TensorRT Export guide](tutorials/model_export.md).

### How can I train a custom YOLOv5 model on my dataset?

Training a custom YOLOv5 model on your dataset involves a few key steps. First, prepare your dataset in the required format, annotated with labels. Then, configure the YOLOv5 training parameters and start the training process using the `train.py` script. For an in-depth tutorial on this process, consult our [Train Custom Data guide](tutorials/train_custom_data.md). It provides step-by-step instructions to ensure optimal results for your specific use case.

### Why should I use Ultralytics YOLOv5 over other object detection models like RCNN?

Ultralytics YOLOv5 is preferred over models like RCNN due to its superior speed and accuracy in real-time object detection. YOLOv5 processes the entire image in one go, making it significantly faster compared to the region-based approach of RCNN, which involves multiple passes. Additionally, YOLOv5's seamless integration with various export formats and extensive documentation make it an excellent choice for both beginners and professionals. Learn more about the architectural advantages in our [Architecture Summary](tutorials/architecture_description.md).

### How can I optimize YOLOv5 model performance during training?

Optimizing YOLOv5 model performance involves tuning various hyperparameters and incorporating techniques like [data augmentation](https://www.ultralytics.com/glossary/data-augmentation) and transfer learning. Ultralytics provides comprehensive resources on hyperparameter evolution and pruning/sparsity to improve model efficiency. You can discover practical tips in our [Tips for Best Training Results guide](tutorials/tips_for_best_training_results.md), which offers actionable insights for achieving optimal performance during training.

### What environments are supported for running YOLOv5 applications?

Ultralytics YOLOv5 supports a variety of environments, including free GPU notebooks on Gradient, Google Colab, Kaggle, as well as major cloud platforms like Google Cloud, Amazon AWS, and Azure. Docker images are also available for convenient setup. For a detailed guide on setting up these environments, check our [Supported Environments](tutorials/roboflow_datasets_integration.md) section, which includes step-by-step instructions for each platform.
