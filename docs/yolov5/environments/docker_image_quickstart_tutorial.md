---
comments: true
description: Learn how to set up and run YOLOv5 in a Docker container. This tutorial includes the prerequisites and step-by-step instructions.
keywords: YOLOv5, Docker, Ultralytics, Image Detection, YOLOv5 Docker Image, Docker Container, Machine Learning, AI
---

# Get Started with YOLOv5 🚀 in Docker

This tutorial will guide you through the process of setting up and running YOLOv5 in a Docker container.

You can also explore other quickstart options for YOLOv5, such as our [Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>, [GCP Deep Learning VM](https://docs.ultralytics.com/yolov5/environments/google_cloud_quickstart_tutorial), and [Amazon AWS](https://docs.ultralytics.com/yolov5/environments/aws_quickstart_tutorial). *Updated: 21 April 2023*.

## Prerequisites

1. **Nvidia Driver**: Version 455.23 or higher. Download from [Nvidia's website](https://www.nvidia.com/Download/index.aspx).
2. **Nvidia-Docker**: Allows Docker to interact with your local GPU. Installation instructions are available on the [Nvidia-Docker GitHub repository](https://github.com/NVIDIA/nvidia-docker).
3. **Docker Engine - CE**: Version 19.03 or higher. Download and installation instructions can be found on the [Docker website](https://docs.docker.com/install/).

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

To run a container with access to local files (e.g., COCO training data in `/datasets`), use the `-v` flag:

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
python train.py  # train a model
python val.py --weights yolov5s.pt  # validate a model for Precision, Recall, and mAP
python detect.py --weights yolov5s.pt --source path/to/images  # run inference on images and videos
python export.py --weights yolov5s.pt --include onnx coreml tflite  # export models to other formats
```

<p align="center"><img width="1000" src="https://user-images.githubusercontent.com/26833433/142224770-6e57caaf-ac01-4719-987f-c37d1b6f401f.png"></p>