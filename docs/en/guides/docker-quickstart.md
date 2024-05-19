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
- Running Ultralytics in a Docker container with CPU or GPU support
- Using a Display Server with Docker to Show Ultralytics Detection Results
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

### Note on File Accessibility

To work with files on your local machine within the container, you can use Docker volumes:

```bash
# Mount a local directory into the container
sudo docker run -it --ipc=host --gpus all -v /path/on/host:/path/in/container $t
```

Replace `/path/on/host` with the directory path on your local machine and `/path/in/container` with the desired path inside the Docker container.

## Run graphical user interface (GUI) applications in a Docker Container

!!! danger "Highly Experimental - User Assumes All Risk"

    The following instructions are experimental. Sharing a X11 socket with a Docker container poses potential security risks. Therefore, it's recommended to test this solution only in a controlled environment. For more information, refer to these resources on how to use `xhost`<sup>[(1)](http://users.stat.umn.edu/~geyer/secure.html)[(2)](https://linux.die.net/man/1/xhost)</sup>.

Docker is primarily used to containerize background applications and CLI programs, but it can also run graphical programs. In the Linux world, two main graphic servers handle graphical display: [X11](https://www.x.org/wiki/) (also known as the X Window System) and [Wayland](https://wayland.freedesktop.org/). Before starting it's essential to determine which graphics server you're currently using. Just run this command to find out:

```bash
env | grep -E -i 'x11|xorg|wayland'
```

Setup and configuration of an X11 or Wayland display server is outside the scope of this guide. If the above command returns nothing, then you'll need to start by getting either working for your system before continuing.

### Running a Docker Container with a GUI

!!! example

    ??? info "Use GPUs"
            If you're using [GPUs](#using-gpus), you can add the `--gpus all` flag to the command.
    
    === "X11"

        If you're using X11, you can run the following command to allow the Docker container to access the X11 socket:

        ```bash
        xhost +local:docker && docker run -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v ~/.Xauthority:/root/.Xauthority \
        -it --ipc=host $t
        ```

        This command sets the `DISPLAY` environment variable to the host's display, mounts the X11 socket, and maps the `.Xauthority` file to the container. The `xhost +local:docker` command allows the Docker container to access the X11 server.


    === "Wayland"

        For Wayland, use the following command:

        ```bash
        xhost +local:docker && docker run -e DISPLAY=$DISPLAY \
        -v $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:/tmp/$WAYLAND_DISPLAY \
        --net=host -it --ipc=host $t
        ```

        This command sets the `DISPLAY` environment variable to the host's display, mounts the Wayland socket, and allows the Docker container to access the Wayland server.

### Using Docker with a GUI

Now you can display graphical applications inside your Docker container. For example, you can run the following [CLI command](../usage/cli.md) to visualize the [predictions](../modes/predict.md) from a [YOLOv8 model](../models/yolov8.md):

```bash
yolo predict model=yolov8n.pt show=True
```

??? info "Testing"

    A simple way to validate that the Docker group has access to the X11 server is to run a container with a GUI program like [`xclock`](https://www.x.org/archive/X11R6.8.1/doc/xclock.1.html) or [`xeyes`](https://www.x.org/releases/X11R7.5/doc/man/man1/xeyes.1.html). Alternatively, you can also install these programs in the Ultralytics Docker container to test the access to the X11 server of your GNU-Linux display server. If you run into any problems, consider setting the environment variable `-e QT_DEBUG_PLUGINS=1`. Setting this environment variable enables the output of debugging information, aiding in the troubleshooting process.

### When finished with Docker GUI

!!! warning "Revoke access"

    In both cases, don't forget to revoke access from the Docker group when you're done.

    ```bash
    xhost -local:docker
    ```

??? question "Want to view image results directly in the Terminal?"

    Refer to the following guide on [viewing the image results using a terminal](./view-results-in-terminal.md)

---

Congratulations! You're now set up to use Ultralytics with Docker and ready to take advantage of its powerful capabilities. For alternate installation methods, feel free to explore the [Ultralytics quickstart documentation](../quickstart.md).
