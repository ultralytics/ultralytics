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

# Official Ultralytics Docker Images

Ultralytics, a Docker Verified Publisher, offers a collection of ready-to-use Docker images available on Docker Hub. These images are designed to simplify your workflow by allowing you to test Ultralytics YOLO securely and efficiently, eliminating the need to manually install dependencies. The official Ultralytics Docker images are optimized for various platforms and use-cases, including configurations for GPU, CPU, ARM64, and NVIDIA Jetson devices.

Below is a detailed list of the available Docker images provided by Ultralytics, including their specific use-cases and compatibility:

| Tags                                             | Dockerfile                          | GPU | Architecture | Description                                                      |
| ------------------------------------------------ | ----------------------------------- | --- | ------------ | ---------------------------------------------------------------- |
| `ultralytics/ultralytics:latest`                 | `docker/Dockerfile`                 | ✅  | amd64        | Default image for training and inference with YOLO models.       |
| `ultralytics/ultralytics:latest-jupyter`         | `docker/Dockerfile-jupyter`         | ✅  | amd64        | Jupyter notebook environment for interactive development.        |
| `ultralytics/ultralytics:latest-cpu`             | `docker/Dockerfile-cpu`             | ❌  | amd64        | CPU-only version for inference and non-GPU environments.         |
| `ultralytics/ultralytics:latest-arm64`           | `docker/Dockerfile-arm64`           | ❌  | arm64        | For ARM64 architecture, suitable for devices like Raspberry Pi.  |
| `ultralytics/ultralytics:latest-jetson-jetpack4` | `docker/Dockerfile-jetson-jetpack4` | ❌  | arm64        | Optimized for NVIDIA Jetson devices.                             |
| `ultralytics/ultralytics:latest-jetson-jetpack5` | `docker/Dockerfile-jetson-jetpack5` | ❌  | arm64        | Optimized for NVIDIA Jetson devices.                             |
| `ultralytics/ultralytics:latest-jetson-jetpack6` | `docker/Dockerfile-jetson-jetpack6` | ❌  | arm64        | Optimized for NVIDIA Jetson devices.                             |
| `ultralytics/ultralytics:latest-python`          | `docker/Dockerfile-python`          | ❌  | amd64        | Minimal Python environment for lightweight applications.         |
| `ultralytics/ultralytics:latest-conda`           | `docker/Dockerfile-conda`           | ✅  | amd64        | Includes Miniconda3 and Ultralytics package installed via Conda. |

???+ note "Image Naming Convention"

    The naming convention for Ultralytics images follows the pattern:

    ```
    ultralytics/ultralytics:<version>-<tag>
    ```

    `<version>`: Represents the image version (with `latest` indicating the latest published image). `<tag>`: Specifies the image tag.

    The default tag latest refers to the most recent image version available. For instance, `ultralytics/ultralytics:latest-jupyter` fetches the latest Jupyter image. If you want to pull a specific version or tag, you can do so by specifying them. For example, an image with version `8.3.58` and tag `jupyter` would be accessed using:

    ```
    ultralytics/ultralytics:8.3.58-jupyter.
    ```

    Whenever a new release is made, the images are automatically pushed to Docker Hub, and the latest tag is updated to reflect the newest release. This ensures that you always have access to the most recent features and improvements by using the latest tag.

In order to pull and run the `ultralytics/ultralytics:latest` image, run the following command:

!!! success "Pull the latest Ultralytics image"

    === "CPU"

        ```bash
        docker run -it --ipc=host ultralytics/ultralytics:latest bash
        ```

    === "NVIDIA GPU"

        ```bash
        docker run -it --ipc=host --gpus all ultralytics/ultralytics:latest bash
        ```

    This command starts a Docker container with the Ultralytics base image, enabling you to run YOLO models. The `--ipc=host` flag allows the container to share the host's IPC namespace, essential for sharing memory between processes. The `--gpus all` flag grants the container access to all available GPUs on the host. Finally, the `bash` command starts an interactive shell within the container, allowing you to execute commands and interact with the Ultralytics environment.

=== "`:latest`"

    This is the default image for training and inference with YOLO models. It includes all necessary dependencies and libraries pre-installed including dependencies for exports and development. This image is optimized for GPU usage and supports NVIDIA GPUs. The image is suitable for training YOLO models on GPU-accelerated hardware.

    This image is the most comprehensive and feature-rich of all the available images and consequently has the largest size. It is recommended for users who require the full range of Ultralytics features and capabilities. Below a list of the most common commands to use the image.

    !!! note "Usage"

        === "Run"
            ```bash
            docker run -it --ipc=host ultralytics/ultralytics:latest bash
            ```

        === "Run with GPUs"
            ```bash
            docker run -it --ipc=host --gpus all ultralytics/ultralytics:latest bash
            ```

        === "Run with GPUs and local directory mounted"
            ```bash
            docker run -it --ipc=host --gpus all -v $PWD:/workspace ultralytics/ultralytics:latest bash
            ```

        === "Pull"
            ```bash
            docker pull ultralytics/ultralytics:latest
            ```

        === "Build"
            ```bash
            docker build -t ultralytics/ultralytics:latest -f docker/Dockerfile .
            ```

        === "Push"
            ```bash
            docker push ultralytics/ultralytics:latest
            ```

=== "`:latest-arm64`"

    For ARM64 architecture, suitable for devices like [Raspberry Pi](raspberry-pi.md). This image is optimized for ARM64 devices and provides a lightweight environment for running YOLO models on ARM64 hardware.

    !!! note "Usage"

        === "Run"
            ```bash
            docker run -it --ipc=host ultralytics/ultralytics:latest-arm64 bash
            ```

        === "Pull"
            ```bash
            docker pull ultralytics/ultralytics:latest-arm64
            ```

        === "Build"
            ```bash
            docker build -t ultralytics/ultralytics:latest-arm64 -f docker/Dockerfile-arm64 .
            ```

        === "Push"
            ```bash
            docker push ultralytics/ultralytics:latest-arm64
            ```


=== "`:latest-cpu`"

    CPU-only version for inference and non-GPU environments. This image is optimized for CPU usage and provides a lightweight environment for running YOLO models on CPU-only hardware. It is suitable for inference and non-GPU environments.

    !!! note "Usage"

        === "Run"
            ```bash
            docker run -it --ipc=host ultralytics/ultralytics:latest-cpu bash
            ```

        === "Pull"
            ```bash
            docker pull ultralytics/ultralytics:latest-cpu
            ```

        === "Build"
            ```bash
            docker build -t ultralytics/ultralytics:latest-cpu -f docker/Dockerfile-cpu .
            ```

        === "Push"
            ```bash
            docker push ultralytics/ultralytics:latest-cpu
            ```

=== "`:latest-jetson-jetpack*`"

    Optimized for NVIDIA Jetson devices. These images are designed for NVIDIA Jetson devices and provide a lightweight environment for running YOLO models on Jetson hardware.

    !!! note "Usage"

        === "Run"
            ```bash
            docker run -it --ipc=host ultralytics/ultralytics:latest-jetson-jetpack4 bash
            ```

        === "Pull"
            ```bash
            docker pull ultralytics/ultralytics:latest-jetson-jetpack4
            ```

        === "Build"
            ```bash
            docker build -t ultralytics/ultralytics:latest-jetson-jetpack4 -f docker/Dockerfile-jetson-jetpack4 .
            ```

        === "Push"
            ```bash
            docker push ultralytics/ultralytics:latest-jetson-jetpack4
            ```

=== "`:latest-python`"

    Minimal Python environment for lightweight applications. This image is optimized for lightweight applications and provides a minimal Python environment for running YOLO models. This image does not include GPU support and extra dependencies. It is suitable for lightweight applications and environments where GPU support is not required.

    !!! note "Usage"

        === "Run"
            ```bash
            docker run -it --ipc=host ultralytics/ultralytics:latest-python bash
            ```

        === "Pull"
            ```bash
            docker pull ultralytics/ultralytics:latest-python
            ```

        === "Build"
            ```bash
            docker build -t ultralytics/ultralytics:latest-python -f docker/Dockerfile-python .
            ```

        === "Push"
            ```bash
            docker push ultralytics/ultralytics:latest-python
            ```

=== "`:latest-conda`"

    Includes [Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/) and Ultralytics package installed via Conda. This image is useful for managing Python environments and dependencies using Conda, providing a lightweight environment for running YOLO models.

    !!! note "Usage"

        === "Run"
            ```bash
            docker run -it --ipc=host ultralytics/ultralytics:latest-conda bash
            ```

        === "Pull"
            ```bash
            docker pull ultralytics/ultralytics:latest-conda
            ```

        === "Build"
            ```bash
            docker build -t ultralytics/ultralytics:latest-conda -f docker/Dockerfile-conda .
            ```

        === "Push"
            ```bash
            docker push ultralytics/ultralytics:latest-conda
            ```

=== "`:latest-jupyter`"

    This Docker image is tailored for interactive development and experimentation with Ultralytics YOLO models. It provides an efficient and user-friendly Jupyter notebook environment, designed to enhance the process of developing, testing, and refining YOLO models. When you run this image, you can access the Jupyter notebook interface in your web browser to experiment with YOLO models.

    A common use case for this image is when you have a remote server and want to run YOLO models interactively in a Jupyter notebook environment on your local machine. You can connect to the remote server and start the docker container. Then you can safely access the Jupyter notebook interface in your web browser from your local machine.

    !!! note "Usage"

        === "Run"
            ```bash
            docker run -it --ipc=host ultralytics/ultralytics:latest-jupyter bash
            ```

        === "Pull"
            ```bash
            docker pull ultralytics/ultralytics:latest-jupyter
            ```

        === "Build"
            ```bash
            docker build -t ultralytics/ultralytics:latest-jupyter -f docker/Dockerfile-jupyter .
            ```

        === "Push"
            ```bash
            docker push ultralytics/ultralytics:latest-jupyter
            ```

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
