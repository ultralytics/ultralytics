---
comments: true
description: Learn how to set up and run YOLOv5 in a Docker container with step-by-step instructions for CPU and GPU environments, mounting volumes, and using display servers.
keywords: YOLOv5, Docker, Ultralytics, setup, guide, tutorial, machine learning, deep learning, AI, GPU, NVIDIA, container, X11, Wayland
---

# Get Started with YOLOv5 🚀 in Docker

Welcome to the Ultralytics YOLOv5 Docker Quickstart Guide! This tutorial provides step-by-step instructions for setting up and running [YOLOv5](../../models/yolov5.md) within a [Docker](https://www.ultralytics.com/glossary/docker) container. Using Docker enables you to run YOLOv5 in an isolated, consistent environment, simplifying deployment and dependency management across different systems. This approach leverages [containerization](https://www.ultralytics.com/glossary/containerization) to package the application and its dependencies together.

For alternative setup methods, consider our [Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>, [GCP Deep Learning VM](./google_cloud_quickstart_tutorial.md), or [Amazon AWS](./aws_quickstart_tutorial.md) guides. For a general overview of Docker usage with Ultralytics models, see the [Ultralytics Docker Quickstart Guide](../../guides/docker-quickstart.md).

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Docker**: Download and install Docker from the [official Docker website](https://docs.docker.com/get-started/get-docker/). Docker is essential for creating and managing containers.
2.  **NVIDIA Drivers** (Required for [GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit) support): Ensure you have NVIDIA drivers version 455.23 or higher installed. You can download the latest drivers from [NVIDIA's website](https://www.nvidia.com/Download/index.aspx).
3.  **NVIDIA Container Toolkit** (Required for GPU support): This toolkit allows Docker containers to access your host machine's NVIDIA GPUs. Follow the official [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for detailed instructions.

### Setting up NVIDIA Container Toolkit (GPU Users)

First, verify that your NVIDIA drivers are installed correctly by running:

```bash
nvidia-smi
```

This command should display information about your GPU(s) and the installed driver version.

Next, install the NVIDIA Container Toolkit. The commands below are typical for Debian-based systems like Ubuntu, but refer to the official guide linked above for instructions specific to your distribution:

```bash
# Add NVIDIA package repositories (refer to official guide for latest setup)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package list and install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker service to apply changes
sudo systemctl restart docker
```

Finally, verify that the NVIDIA runtime is configured and available to Docker:

```bash
docker info | grep -i runtime
```

You should see `nvidia` listed as one of the available runtimes.

## Step 1: Pull the YOLOv5 Docker Image

Ultralytics provides official YOLOv5 images on [Docker Hub](https://hub.docker.com/). The `latest` tag tracks the most recent repository commit, ensuring you always get the newest version. Pull the image using the following command:

```bash
# Define the image name with tag
t=ultralytics/yolov5:latest

# Pull the latest YOLOv5 image from Docker Hub
sudo docker pull $t
```

You can browse all available images at the [Ultralytics YOLOv5 Docker Hub repository](https://hub.docker.com/r/ultralytics/yolov5).

## Step 2: Run the Docker Container

Once the image is pulled, you can run it as a container.

### Using CPU Only

To run an interactive container instance using only the CPU, use the `-it` flag. The `--ipc=host` flag allows sharing of host IPC namespace, which is important for shared memory access.

```bash
# Run an interactive container instance using CPU
sudo docker run -it --ipc=host $t
```

### Using GPU

To enable GPU access within the container, use the `--gpus` flag. This requires the NVIDIA Container Toolkit to be installed correctly.

```bash
# Run with access to all available GPUs
sudo docker run -it --ipc=host --gpus all $t

# Run with access to specific GPUs (e.g., GPUs 2 and 3)
sudo docker run -it --ipc=host --gpus '"device=2,3"' $t
```

Refer to the [Docker run reference](https://docs.docker.com/engine/containers/run/) for more details on command options.

### Mounting Local Directories

To work with your local files (datasets, model weights, etc.) inside the container, use the `-v` flag to mount a host directory into the container:

```bash
# Mount /path/on/host (your local machine) to /path/in/container (inside the container)
sudo docker run -it --ipc=host --gpus all -v /path/on/host:/path/in/container $t
```

Replace `/path/on/host` with the actual path on your machine and `/path/in/container` with the desired path inside the Docker container (e.g., `/usr/src/datasets`).

## Step 3: Use YOLOv5 🚀 within the Docker Container

You are now inside the running YOLOv5 Docker container! From here, you can execute standard YOLOv5 commands for various [Machine Learning](https://www.ultralytics.com/glossary/machine-learning-ml) and [Deep Learning](https://www.ultralytics.com/glossary/deep-learning-dl) tasks like [Object Detection](https://www.ultralytics.com/glossary/object-detection).

```bash
# Train a YOLOv5 model on your custom dataset (ensure data is mounted or downloaded)
python train.py --data your_dataset.yaml --weights yolov5s.pt --img 640 # Start training

# Validate the trained model's performance (Precision, Recall, mAP)
python val.py --weights path/to/your/best.pt --data your_dataset.yaml # Validate accuracy

# Run inference on images or videos using a trained model
python detect.py --weights yolov5s.pt --source path/to/your/images_or_videos # Perform detection

# Export the trained model to various formats like ONNX, CoreML, or TFLite for deployment
python export.py --weights yolov5s.pt --include onnx coreml tflite # Export model
```

Explore the documentation for detailed usage of different modes:

- [Train](../../modes/train.md)
- [Validate](../../modes/val.md)
- [Predict](../../modes/predict.md)
- [Export](../../modes/export.md)

Learn more about evaluation metrics like [Precision](https://www.ultralytics.com/glossary/precision), [Recall](https://www.ultralytics.com/glossary/recall), and [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map). Understand different export formats like [ONNX](../../integrations/onnx.md), [CoreML](../../integrations/coreml.md), and [TFLite](../../integrations/tflite.md), and explore various [Model Deployment Options](../../guides/model-deployment-options.md). Remember to manage your [model weights](https://www.ultralytics.com/glossary/model-weights) effectively.

<p align="center"><img width="1000" src="https://github.com/ultralytics/docs/releases/download/0/gcp-running-docker.avif" alt="Running YOLOv5 inside a Docker container on GCP"></p>

Congratulations! You have successfully set up and run YOLOv5 within a Docker container.
