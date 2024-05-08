---
comments: true
description: Complete guide to setting up and using Ultralytics YOLO models with Docker. Learn how to install Docker, manage GPU support, and run YOLO models in isolated containers.
keywords: Ultralytics, YOLO, Docker, GPU, containerization, object detection, package installation, deep learning, machine learning, guide
---

# Docker Quickstart Guide for Ultralytics

<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/26833433/270173601-fc7011bd-e67c-452f-a31a-aa047dcd2771.png" alt="Ultralytics Docker Package Visual">
</p>

This guide serves as a comprehensive introduction to setting up a Docker environment for your Ultralytics projects. [Docker](https://docker.com/) is a platform for developing, shipping, and running applications in containers. It is particularly beneficial for ensuring that the software will always run the same, regardless of where it's deployed. For more details, visit the Ultralytics Docker repository on [Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics).

[![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker)](https://hub.docker.com/r/ultralytics/ultralytics)

## What You Will Learn

- Setting up Docker with NVIDIA support
- Installing Ultralytics Docker images
- Running Ultralytics in a Docker container
- Mounting local directories into the container

---

## Prerequisites

- Make sure Docker is installed on your system. If not, you can download and install it from [Docker's website](https://www.docker.com/products/docker-desktop).
- Ensure that your system has an NVIDIA GPU and NVIDIA drivers are installed.

---

## Setting up Docker with NVIDIA Support

First, verify that the NVIDIA drivers are properly installed by running:

```bash
nvidia-smi
```

### Installing NVIDIA Docker Runtime

Now, let's install the NVIDIA Docker runtime to enable GPU support in Docker containers:

```bash
# Add NVIDIA package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(lsb_release -cs)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install NVIDIA Docker runtime
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker service to apply changes
sudo systemctl restart docker
```

### Verify NVIDIA Runtime with Docker

Run `docker info | grep -i runtime` to ensure that `nvidia` appears in the list of runtimes:

```bash
docker info | grep -i runtime
```

---

## Installing Ultralytics Docker Images

Ultralytics offers several Docker images optimized for various platforms and use-cases:

- **Dockerfile:** GPU image, ideal for training.
- **Dockerfile-arm64:** For ARM64 architecture, suitable for devices like [Raspberry Pi](raspberry-pi.md).
- **Dockerfile-cpu:** CPU-only version for inference and non-GPU environments.
- **Dockerfile-jetson:** Optimized for NVIDIA Jetson devices.
- **Dockerfile-python:** Minimal Python environment for lightweight applications.
- **Dockerfile-conda:** Includes [Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/) and Ultralytics package installed via Conda.

To pull the latest image:

```bash
# Set image name as a variable
t=ultralytics/ultralytics:latest

# Pull the latest Ultralytics image from Docker Hub
sudo docker pull $t
```

---

## Running Ultralytics in Docker Container

Here's how to execute the Ultralytics Docker container:
### Using only the CPU
```bash
# Run with all GPUs
sudo docker run -it --ipc=host $t
```
### Using GPUs
```bash
# Run with all GPUs
sudo docker run -it --ipc=host --gpus all $t

# Run specifying which GPUs to use
sudo docker run -it --ipc=host --gpus '"device=2,3"' $t
```

The `-it` flag assigns a pseudo-TTY and keeps stdin open, allowing you to interact with the container. The `--ipc=host` flag enables sharing of host's IPC namespace, essential for sharing memory between processes. The `--gpus` flag allows the container to access the host's GPUs.

### Visualize Ultralytics' Object detection in your GNU-Linux Display Server

The following instructions are highly experimental. Sharing a X11 socket with a docker container can raise security concerns, and as such this solution should be tested only in a controlled environment. 

Firstly, there is the need to allow the docker container to access the X11 socket of your GNU-Linux Display Server. There is also the need to mount the socket in the container and have the `DISPLAY` environment variable set to point to it.

Secondly there is the need to grant access to the X11 Server. For that there is the need to allow connections from the Docker container into the host. A restrictive way to do it is :
```bash
xhost +local:docker
```
This allows connections from the `docker` group into the X11 Server. 

A simple way to validate that the docker group has access to the X11 server is to run a container with a GUI program like `xclock` or `xeyes`. Alternatively, you can also install these programs in the ultralytics docker container to test the access to the X11 Server of you GNU-Linux Display Server .

Below there are presented two examples on how to run the Ultralytics Object detection container in  in both Xorg and Wayland in interact mode `-it`. You can run the commands to start the docker container with using a [GPU](#using-gpus)
#### Xorg
1. Open a terminal and run:

```bash
docker run -e DISPLAY=$DISPLAY -e QT_DEBUG_PLUGINS=1  -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/.Xauthority:/root/.Xauthority -it $t
```
2. Open another terminal and run:
```bash
xhost +local:docker
```
3. Test the connection to the X11 Server inside the docker terminal:
```bash
yolo predict show=True
```
#### Wayland
1. Open a terminal and use the following command to run the container:
```bash
docker run --env DISPLAY=$DISPLAY --volume $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:/tmp/$WAYLAND_DISPLAY -e QT_DEBUG_PLUGINS=1 --net=host -it $t  
#Note: Wayland also supports running the command.
```

2. Open another terminal and run:
```bash
xhost +local:docker
#Note: You may need to rerun the command each time that the X11 Server session closes. 
```
3. Test the connection to the X11 Server inside the docker terminal:
```bash
yolo predict show=True
```

### Note on File Accessibility

To work with files on your local machine within the container, you can use Docker volumes:

```bash
# Mount a local directory into the container
sudo docker run -it --ipc=host --gpus all -v /path/on/host:/path/in/container $t
```

Replace `/path/on/host` with the directory path on your local machine and `/path/in/container` with the desired path inside the Docker container.

---

Congratulations! You're now set up to use Ultralytics with Docker and ready to take advantage of its powerful capabilities. For alternate installation methods, feel free to explore the [Ultralytics quickstart documentation](../quickstart.md).
