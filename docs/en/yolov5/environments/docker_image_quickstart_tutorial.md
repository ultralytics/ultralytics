---
comments: true
description: Learn how to set up and run YOLOv5 in a Docker container with step-by-step instructions. Explore other quickstart options for an easy setup.
keywords: YOLOv5, Docker, Ultralytics, setup, guide, tutorial, machine learning, deep learning, AI, GPU, NVIDIA, container
---

# Get Started with YOLOv5 🚀 in Docker

This tutorial will guide you through the process of setting up and running YOLOv5 in a Docker container.

You can also explore other quickstart options for YOLOv5, such as our [Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>, [GCP Deep Learning VM](./google_cloud_quickstart_tutorial.md), and [Amazon AWS](./aws_quickstart_tutorial.md).

## Prerequisites

1. **NVIDIA Driver**: Version 455.23 or higher. Download from [Nvidia's website](https://www.nvidia.com/Download/index.aspx).
2. **NVIDIA-Docker**: Allows Docker to interact with your local GPU. Installation instructions are available on the [NVIDIA-Docker GitHub repository](https://github.com/NVIDIA/nvidia-docker).
3. **Docker Engine - CE**: Version 19.03 or higher. Download and installation instructions can be found on the [Docker website](https://docs.docker.com/get-started/get-docker/).

## Step 1: Pull the YOLOv5 Docker Image

The Ultralytics YOLOv5 DockerHub repository is available at [https://hub.docker.com/r/ultralytics/yolov5](https://hub.docker.com/r/ultralytics/yolov5). Docker Autobuild ensures that the `ultralytics/yolov5:latest` image is always in sync with the most recent repository commit. To pull the latest image, run the following command:

```bash
sudo docker pull ultralytics/yolov5:latest
```

## Step 2: Run the Docker Container

### Basic container:

Run an interactive instance of the YOLOv5 Docker image (called a "container") using the `-it` flag:

```bash
sudo docker run --ipc=host -it ultralytics/yolov5:latest
```

### Container with local file access:

To run a container with access to local files (e.g., COCO [training data](https://www.ultralytics.com/glossary/training-data) in `/datasets`), use the `-v` flag:

```bash
sudo docker run --ipc=host -it -v "$(pwd)"/datasets:/usr/src/datasets ultralytics/yolov5:latest
```

### Container with GPU access:

To run a container with GPU access, use the `--gpus all` flag:

```bash
sudo docker run --ipc=host -it --gpus all ultralytics/yolov5:latest
```

## Step 3: Use YOLOv5 🚀 within the Docker Container

Now you can train, test, detect, and export YOLOv5 models within the running Docker container:

```bash
# Train a model on your data
python train.py

# Validate the trained model for Precision, Recall, and mAP
python val.py --weights yolov5s.pt

# Run inference using the trained model on your images or videos
python detect.py --weights yolov5s.pt --source path/to/images

# Export the trained model to other formats for deployment
python export.py --weights yolov5s.pt --include onnx coreml tflite
```

<p align="center"><img width="1000" src="https://github.com/ultralytics/docs/releases/download/0/gcp-running-docker.avif" alt="GCP running Docker"></p>
