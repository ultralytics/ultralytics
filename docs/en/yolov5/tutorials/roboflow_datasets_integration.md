---
comments: true
description: Learn how to use Roboflow for organizing, labeling, and versioning datasets to train YOLOv5 models. Free for public workspaces.
keywords: Roboflow, YOLOv5, data management, dataset labeling, dataset versioning, Ultralytics, machine learning, AI training
---

# Roboflow Datasets

You can now use Roboflow to organize, label, prepare, version, and host your datasets for training YOLOv5 🚀 models. Roboflow is free to use with YOLOv5 if you make your workspace public.

!!! question "Licensing"

    Ultralytics offers two licensing options:

    - The [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), an [OSI-approved](https://opensource.org/license) open-source license ideal for students and enthusiasts.
    - The [Enterprise License](https://www.ultralytics.com/license) for businesses seeking to incorporate our AI models into their products and services.

    For more details see [Ultralytics Licensing](https://www.ultralytics.com/license).

## Upload

You can upload your data to Roboflow via [web UI](https://docs.roboflow.com/adding-data?ref=ultralytics), [REST API](https://docs.roboflow.com/adding-data/upload-api?ref=ultralytics), or [Python](https://docs.roboflow.com/python?ref=ultralytics).

## Labeling

After uploading data to Roboflow, you can label your data and review previous labels.

[![Roboflow Annotate](https://github.com/ultralytics/docs/releases/download/0/roboflow-annotate-1.avif)](https://roboflow.com/annotate)

## Versioning

You can make versions of your dataset with different preprocessing and offline augmentation options. YOLOv5 does online augmentations natively, so be intentional when layering Roboflow offline augmentations on top.

![Roboflow Preprocessing](https://github.com/ultralytics/docs/releases/download/0/roboflow-preprocessing.avif)

## Exporting Data

You can download your data in YOLOv5 format to quickly begin training.

```
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR API KEY HERE")
project = rf.workspace().project("YOUR PROJECT")
dataset = project.version("YOUR VERSION").download("yolov5")
```

## Custom Training

We have released a custom training tutorial demonstrating all of the above capabilities. You can access the code here:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/yolov5-custom-training-tutorial/blob/main/yolov5-custom-training.ipynb)

## Active Learning

The real world is messy and your model will invariably encounter situations your dataset didn't anticipate. Using [active learning](https://blog.roboflow.com/what-is-active-learning/?ref=ultralytics) is an important strategy to iteratively improve your dataset and model. With the Roboflow and YOLOv5 integration, you can quickly make improvements on your [model deployments](https://www.ultralytics.com/glossary/model-deployment) by using a battle tested [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) pipeline.

<p align=""><a href="https://roboflow.com/?ref=ultralytics"><img width="1000" src="https://github.com/ultralytics/docs/releases/download/0/roboflow-active-learning.avif" alt="Roboflow active learning"></a></p>

## Supported Environments

Ultralytics provides a range of ready-to-use environments, each pre-installed with essential dependencies such as [CUDA](https://developer.nvidia.com/cuda-zone), [CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/), and [PyTorch](https://pytorch.org/), to kickstart your projects.

- **Free GPU Notebooks**: <a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
- **Google Cloud**: [GCP Quickstart Guide](../environments/google_cloud_quickstart_tutorial.md)
- **Amazon**: [AWS Quickstart Guide](../environments/aws_quickstart_tutorial.md)
- **Azure**: [AzureML Quickstart Guide](../environments/azureml_quickstart_tutorial.md)
- **Docker**: [Docker Quickstart Guide](../environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

## Project Status

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>

This badge indicates that all [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) Continuous Integration (CI) tests are successfully passing. These CI tests rigorously check the functionality and performance of YOLOv5 across various key aspects: [training](https://github.com/ultralytics/yolov5/blob/master/train.py), [validation](https://github.com/ultralytics/yolov5/blob/master/val.py), [inference](https://github.com/ultralytics/yolov5/blob/master/detect.py), [export](https://github.com/ultralytics/yolov5/blob/master/export.py), and [benchmarks](https://github.com/ultralytics/yolov5/blob/master/benchmarks.py). They ensure consistent and reliable operation on macOS, Windows, and Ubuntu, with tests conducted every 24 hours and upon each new commit.

## FAQ

### How do I upload data to Roboflow for training YOLOv5 models?

You can upload your data to Roboflow using three different methods: via the website, the REST API, or through Python. These options offer flexibility depending on your technical preference or project requirements. Once your data is uploaded, you can organize, label, and version it to prepare for training with Ultralytics YOLOv5 models. For more details, visit the [Upload](#upload) section of the documentation.

### What are the advantages of using Roboflow for data labeling and versioning?

Roboflow provides a comprehensive platform for data organization, labeling, and versioning which is essential for efficient machine learning workflows. By using Roboflow with YOLOv5, you can streamline the process of dataset preparation, ensuring that your data is accurately annotated and consistently versioned. The platform also supports various preprocessing and offline augmentation options to enhance your dataset's quality. For a deeper dive into these features, see the [Labeling](#labeling) and [Versioning](#versioning) sections of the documentation.

### How can I export my dataset from Roboflow to YOLOv5 format?

Exporting your dataset from Roboflow to YOLOv5 format is straightforward. You can use the Python code snippet provided in the documentation:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR API KEY HERE")
project = rf.workspace().project("YOUR PROJECT")
dataset = project.version("YOUR VERSION").download("yolov5")
```

This code will download your dataset in a format compatible with YOLOv5, allowing you to quickly begin training your model. For more details, refer to the [Exporting Data](#exporting-data) section.

### What is [active learning](https://www.ultralytics.com/glossary/active-learning) and how does it work with YOLOv5 and Roboflow?

Active learning is a machine learning strategy that iteratively improves a model by intelligently selecting the most informative data points to label. With the Roboflow and YOLOv5 integration, you can implement active learning to continuously enhance your model's performance. This involves deploying a model, capturing new data, using the model to make predictions, and then manually verifying or correcting those predictions to further train the model. For more insights into active learning see the [Active Learning](#active-learning) section above.

### How can I use Ultralytics environments for training YOLOv5 models on different platforms?

Ultralytics provides ready-to-use environments with pre-installed dependencies like CUDA, CUDNN, Python, and [PyTorch](https://www.ultralytics.com/glossary/pytorch), making it easier to kickstart your training projects. These environments are available on various platforms such as Google Cloud, AWS, Azure, and Docker. You can also access free GPU notebooks via [Paperspace](https://bit.ly/yolov5-paperspace-notebook), [Google Colab](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb), and [Kaggle](https://www.kaggle.com/models/ultralytics/yolov5). For specific setup instructions, visit the [Supported Environments](#supported-environments) section of the documentation.
