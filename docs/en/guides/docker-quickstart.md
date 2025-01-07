---
comments: true
description: Learn to effortlessly set up Ultralytics in Docker, from installation to running with CPU/GPU support. Follow our comprehensive guide for seamless container experience.
keywords: Ultralytics, Docker, Quickstart Guide, CPU support, GPU support, NVIDIA Docker, container setup, Docker environment, Docker Hub, Ultralytics projects
---

# Docker Quickstart Guide for Ultralytics

<p align="center">
  <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-docker-package-visual.avif" alt="Ultralytics Docker Package Visual">
</p>

This guide serves as a comprehensive introduction to setting up a Docker environment for your Ultralytics projects. [Docker](https://www.docker.com/) is a platform for developing, shipping, and running applications in containers. It is particularly beneficial for ensuring that the software will always run the same, regardless of where it's deployed. For more details, visit the Ultralytics Docker repository on [Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics).

[![Docker Image Version](https://img.shields.io/docker/v/ultralytics/ultralytics?sort=semver&logo=docker)](https://hub.docker.com/r/ultralytics/ultralytics)
[![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/ultralytics)](https://hub.docker.com/r/ultralytics/ultralytics)

## What You Will Learn

- Setting up Docker with NVIDIA support
- Installing Ultralytics Docker images
- Running Ultralytics in a Docker container with CPU or GPU support
- Using a Display Server with Docker to Show Ultralytics Detection Results
- Mounting local directories into the container

---

## Prerequisites

- Make sure Docker is installed on your system. If not, you can download and install it from [Docker's website](https://www.docker.com/products/docker-desktop/).
- (Optional) If you have an NVIDIA GPU, you need to install the NVIDIA Container Toolkit to use Docker with your GPU (instructions below).

---

## Setting up Docker with NVIDIA Container Toolkit

!!! tip "Skip if you don't have an NVIDIA GPU"

    If your system does not have an NVIDIA GPU, you can skip this section.

This guide will walk you through the process of configuring Docker to leverage NVIDIA GPUs using the NVIDIA Container Toolkit. This setup is essential for running GPU-accelerated applications inside Docker containers.

Before proceeding, ensure that your system meets the following prerequisites:

1. **NVIDIA GPU**: Make sure you have a compatible NVIDIA GPU installed on your system.
2. **NVIDIA Drivers**: Install the appropriate NVIDIA drivers for your GPU. You can verify the installation by executing the following command:

    ```sh
    nvidia-smi
    ```
   This command should display details about your NVIDIA GPU, such as its model, driver version, and current usage statistics. If it doesn't, check your driver installation.

### Installing NVIDIA Container Toolkit

The NVIDIA Container Toolkit provides Docker container support for NVIDIA GPUs, enabling GPU-accelerated applications to run seamlessly inside containers. Below are the steps to install and configure the toolkit.

#### Step 1: Set up the NVIDIA Container Toolkit repository

First, you'll need to add the NVIDIA Container Toolkit repository to your package manager's sources. This ensures you have access to the latest toolkit packages directly from NVIDIA. Execute the following command:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

This command does the following:
- Downloads the GPG key for the repository and stores it securely.
- Adds the NVIDIA Container Toolkit repository to your system's package sources list.

#### Step 2: Update the package list and install the NVIDIA Container Toolkit

Next, update your package list to include the newly added repository and install the NVIDIA Container Toolkit:

```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

This command performs the following actions:
- `sudo apt-get update`: Refreshes your package manager's list of available packages, including those from the NVIDIA repository.
- `sudo apt-get install -y nvidia-container-toolkit`: Installs the NVIDIA Container Toolkit on your system.

#### Step 3: Restart the Docker daemon

For the changes to take effect, you need to restart the Docker daemon. This ensures that Docker recognizes and utilizes the NVIDIA runtime:

```bash
sudo systemctl restart docker
```

Restarting Docker allows it to reload its configuration and apply any updates related to the NVIDIA runtime.

#### Step 4: Verify the NVIDIA runtime is installed

Finally, verify that the NVIDIA runtime is installed and functioning correctly by running a test Docker container that uses the GPU:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

This command starts a temporary Ubuntu container and runs the `nvidia-smi` command within it. If everything is set up correctly, you should see output similar to the following, indicating that the container can access the NVIDIA GPU:

```console
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.58.02              Driver Version: 555.58.02      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060        Off |   00000000:01:00.0  On |                  N/A |
|  0%   40C    P5             19W /  170W |     785MiB /  12288MiB |     17%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
```

This output confirms that Docker can successfully use the GPU, enabling you to run GPU-accelerated applications in your containers. If you encounter any issues, double-check the installation steps and ensure that your NVIDIA drivers are correctly installed.

---

## Official Ultralytics Docker Images

Ultralytics is a Docker Verified Publisher, providing a range of convenient, ready-to-use images directly available on Docker Hub. These images streamline your workflow, allowing you to test Ultralytics YOLO securely and efficiently without the hassle of installing dependencies. The official Ultralytics Docker images are optimized for various platforms and use-cases, including GPU, CPU, ARM64, and NVIDIA Jetson devices.

Below is a list of the available images:

| tags                                              | Dockerfile                    | GPU support  | Architecture  |
| -------                                           | ----------                    | ------------ | ------------  |
| ultralytics/ultralytics:latest                    | docker/Dockerfile             |   ✅         | amd64         |
| ultralytics/ultralytics:latest-jupyter            | docker/Dockerfile-jupyter     |   ✅         | amd64         |
| ultralytics/ultralytics:latest-cpu                | docker/Dockerfile-cpu         |   ❌         | amd64         |
| ultralytics/ultralytics:latest-arm64              | docker/Dockerfile-arm64       |   ❌         | arm64         |
| ultralytics/ultralytics:latest-jetson-jetpack4    | docker/Dockerfile-cpu         |   ❌         | arm64         |
| ultralytics/ultralytics:latest-jetson-jetpack5    | docker/Dockerfile-cpu         |   ❌         | arm64         |
| ultralytics/ultralytics:latest-jetson-jetpack6    | docker/Dockerfile-cpu         |   ❌         | arm64         |
| ultralytics/ultralytics:latest-python             | docker/Dockerfile-python      |   ❌         | amd64         |
| ultralytics/ultralytics:latest-conda              | docker/Dockerfile-conda       |   ✅         | amd64         |

The naming convention for the images is `ultralytics/ultralytics:<version>-<tag>`, where `<version>` is the image version and `<tag>` is the image tag. latest is the default tag, which pulls the latest image version. For example, ultralytics/ultralytics:latest-jupyter refers to the latest Jupyter image. You can also specify a specific version or tag to pull a particular image version from Docker Hub, example, the image version 8.3.58 with the tag jupyter would be ultralytics/ultralytics:8.3.58-jupyter. Images are pushed to docker hub automatically with the latest tag when a new release is made, and the latest tag is updated to the latest release.

In order to pull and run the ultralytics/ultralytics:latest image, run the following command:

!!! note ""

    === "CPU"

        ```bash
        docker run -it --ipc=host --gpus all ultralytics/ultralytics:latest bash
        ```

    === "NVIDIA GPU"

        ```bash
        docker run -it --ipc=host ultralytics/ultralytics:latest bash
        ```

This command starts a Docker container with the Ultralytics image, enabling you to run YOLO models with GPU support. The `--ipc=host` flag allows the container to share the host's IPC namespace, essential for sharing memory between processes. The `--gpus all` flag grants the container access to all available GPUs on the host. Finally, the `bash` command starts an interactive shell within the container, allowing you to execute commands and interact with the Ultralytics environment.


=== "`:latest`"

    This is the default image for training and inference with YOLO models. It includes all necessary dependencies and libraries pre-installed. This image is optimized for GPU usage and supports NVIDIA GPUs. The image is suitable for training and inference with YOLO models on GPU-accelerated hardware.

=== "`:latest-arm64`"

    For ARM64 architecture, suitable for devices like [Raspberry Pi](raspberry-pi.md). This image is optimized for ARM64 devices and provides a lightweight environment for running YOLO models on ARM64 hardware.

=== "`:latest-cpu`"

    CPU-only version for inference and non-GPU environments. This image is optimized for CPU usage and provides a lightweight environment for running YOLO models on CPU-only hardware.

=== "`:latest-jetson-jetpack*`"

    Optimized for NVIDIA Jetson devices. These images are designed for NVIDIA Jetson devices and provide a lightweight environment for running YOLO models on Jetson hardware.

=== "`:latest-python`"

    Minimal Python environment for lightweight applications. This image is optimized for lightweight applications and provides a minimal Python environment for running YOLO models.

=== "`:latest-conda`"

    Includes [Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/) and Ultralytics package installed via Conda. This image is useful for managing Python environments and dependencies using Conda.

=== "`:latest-jupyter`"

    Jupyter notebook environment for interactive development. Very useful for experimenting with YOLO models. This image is optimized for interactive development and provides a Jupyter notebook environment for running YOLO models interactively.


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

Docker is primarily used to containerize background applications and CLI programs, but it can also run graphical programs. In the Linux world, two main graphic servers handle graphical display: [X11](https://www.x.org/wiki/) (also known as the X Window System) and [Wayland](https://wayland.freedesktop.org/). Before starting, it's essential to determine which graphics server you are currently using. Run this command to find out:

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

Now you can display graphical applications inside your Docker container. For example, you can run the following [CLI command](../usage/cli.md) to visualize the [predictions](../modes/predict.md) from a [YOLO11 model](../models/yolo11.md):

```bash
yolo predict model=yolo11n.pt show=True
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

## FAQ

### How do I set up Ultralytics with Docker?

To set up Ultralytics with Docker, first ensure that Docker is installed on your system. If you have an NVIDIA GPU, install the NVIDIA Docker runtime to enable GPU support. Then, pull the latest Ultralytics Docker image from Docker Hub using the following command:

```bash
sudo docker pull ultralytics/ultralytics:latest
```

For detailed steps, refer to our [Docker Quickstart Guide](../quickstart.md).

### What are the benefits of using Ultralytics Docker images for [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) projects?

Using Ultralytics Docker images ensures a consistent environment across different machines, replicating the same software and dependencies. This is particularly useful for collaborating across teams, running models on various hardware, and maintaining reproducibility. For GPU-based training, Ultralytics provides optimized Docker images such as `Dockerfile` for general GPU usage and `Dockerfile-jetson` for NVIDIA Jetson devices. Explore [Ultralytics Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics) for more details.

### How can I run Ultralytics YOLO in a Docker container with GPU support?

First, ensure that the NVIDIA Docker runtime is installed and configured. Then, use the following command to run Ultralytics YOLO with GPU support:

```bash
sudo docker run -it --ipc=host --gpus all ultralytics/ultralytics:latest
```

This command sets up a Docker container with GPU access. For additional details, see the [Docker Quickstart Guide](../quickstart.md).

### How do I visualize YOLO prediction results in a Docker container with a display server?

To visualize YOLO prediction results with a GUI in a Docker container, you need to allow Docker to access your display server. For systems running X11, the command is:

```bash
xhost +local:docker && docker run -e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v ~/.Xauthority:/root/.Xauthority \
-it --ipc=host ultralytics/ultralytics:latest
```

For systems running Wayland, use:

```bash
xhost +local:docker && docker run -e DISPLAY=$DISPLAY \
-v $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:/tmp/$WAYLAND_DISPLAY \
--net=host -it --ipc=host ultralytics/ultralytics:latest
```

More information can be found in the [Run graphical user interface (GUI) applications in a Docker Container](#run-graphical-user-interface-gui-applications-in-a-docker-container) section.

### Can I mount local directories into the Ultralytics Docker container?

Yes, you can mount local directories into the Ultralytics Docker container using the `-v` flag:

```bash
sudo docker run -it --ipc=host --gpus all -v /path/on/host:/path/in/container ultralytics/ultralytics:latest
```

Replace `/path/on/host` with the directory on your local machine and `/path/in/container` with the desired path inside the container. This setup allows you to work with your local files within the container. For more information, refer to the relevant section on [mounting local directories](../usage/python.md).
