---
comments: true
description: Learn how to set up and run YOLOv5 in a Docker container with step-by-step instructions for CPU and GPU environments, mounting volumes, and using display servers.
keywords: YOLOv5, Docker, Ultralytics, setup, guide, tutorial, machine learning, deep learning, AI, GPU, NVIDIA, container, X11, Wayland
---

# Get Started with YOLOv5 🚀 in Docker

This tutorial will guide you through the process of setting up and running YOLOv5 in a Docker container, providing comprehensive instructions for both CPU and GPU environments.

You can also explore other quickstart options for YOLOv5, such as our [Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>, [GCP Deep Learning VM](./google_cloud_quickstart_tutorial.md), and [Amazon AWS](./aws_quickstart_tutorial.md).

## Prerequisites

1. **Docker**: Install Docker from the [official Docker website](https://docs.docker.com/get-docker/).
2. **NVIDIA Driver** (for GPU support): Version 455.23 or higher. Download from [NVIDIA's website](https://www.nvidia.com/Download/index.aspx).
3. **NVIDIA Docker Runtime** (for GPU support): Allows Docker to interact with your local GPU. Follow the installation instructions below.

### Setting up NVIDIA Docker Runtime

Verify that your NVIDIA drivers are properly installed:

```bash
nvidia-smi
```

Install the NVIDIA Docker runtime:

```bash
# Add NVIDIA package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(lsb_release -cs)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install NVIDIA Docker runtime
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker service
sudo systemctl restart docker
```

Verify the NVIDIA runtime is available:

```bash
docker info | grep -i runtime
```

## Step 1: Pull the YOLOv5 Docker Image

The Ultralytics YOLOv5 DockerHub repository is available at [https://hub.docker.com/r/ultralytics/yolov5](https://hub.docker.com/r/ultralytics/yolov5). Docker Autobuild ensures that the `ultralytics/yolov5:latest` image is always in sync with the most recent repository commit.

```bash
# Set image name as a variable
t=ultralytics/yolov5:latest

# Pull the latest image
sudo docker pull $t
```

## Step 2: Run the Docker Container

### Using CPU Only

Run an interactive instance of the YOLOv5 Docker image (called a "container") using the `-it` flag:

```bash
# Run without GPU
sudo docker run -it --ipc=host $t
```

### Using GPU

```bash
# Run with all GPUs
sudo docker run -it --ipc=host --gpus all $t

# Run with specific GPUs
sudo docker run -it --ipc=host --gpus '"device=2,3"' $t
```

### Mounting Local Directories

To access files on your local machine within the container:

```bash
# Mount a local directory into the container
sudo docker run -it --ipc=host --gpus all -v /path/on/host:/path/in/container $t
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
