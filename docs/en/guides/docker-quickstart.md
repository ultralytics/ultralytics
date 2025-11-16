---
comments: true
description: Learn to effortlessly set up Ultralytics in Docker, from installation to running with CPU/GPU support. Follow our comprehensive guide for seamless container experience.
keywords: Ultralytics, Docker, Quickstart Guide, CPU support, GPU support, NVIDIA Docker, NVIDIA Container Toolkit, container setup, Docker environment, Docker Hub, Ultralytics projects
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
- Using a display server with Docker to show Ultralytics detection results
- Mounting local directories into the container

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/IYWQZvtOy_Q"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Get started with Docker | Usage of Ultralytics Python Package inside Docker live demo 🎉
</p>

---

## Prerequisites

- Make sure Docker is installed on your system. If not, you can download and install it from [Docker's website](https://www.docker.com/products/docker-desktop/).
- Ensure that your system has an NVIDIA GPU and NVIDIA drivers are installed.
- If you are using NVIDIA Jetson devices, ensure that you have the appropriate JetPack version installed. Refer to the [NVIDIA Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/) for more details.

---

## Setting up Docker with NVIDIA Support

First, verify that the NVIDIA drivers are properly installed by running:

```bash
nvidia-smi
```

### Installing NVIDIA Container Toolkit

Now, let's install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) to enable GPU support in Docker containers:

=== "Ubuntu/Debian"

    ```bash
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    ```
    Update the package lists and install the nvidia-container-toolkit package:

    ```bash
    sudo apt-get update
    ```

    Install the latest version of `nvidia-container-toolkit`:

    ```bash
    sudo apt-get install -y nvidia-container-toolkit \
      nvidia-container-toolkit-base libnvidia-container-tools \
      libnvidia-container1
    ```

    ??? info "Optional: Install specific version of nvidia-container-toolkit"

        Optionally, you can install a specific version of the nvidia-container-toolkit by setting the `NVIDIA_CONTAINER_TOOLKIT_VERSION` environment variable:

        ```bash
        export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
        sudo apt-get install -y \
          nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
          nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
          libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
          libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
        ```

    ```bash
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

=== "RHEL/CentOS/Fedora/Amazon Linux"

    ```bash
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
      | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
    ```

    Update the package lists and install the nvidia-container-toolkit package:

    ```bash
    sudo dnf clean expire-cache
    sudo dnf check-update
    ```

    ```bash
    sudo dnf install \
      nvidia-container-toolkit \
      nvidia-container-toolkit-base \
      libnvidia-container-tools \
      libnvidia-container1
    ```


    ??? info "Optional: Install specific version of nvidia-container-toolkit"

        Optionally, you can install a specific version of the nvidia-container-toolkit by setting the `NVIDIA_CONTAINER_TOOLKIT_VERSION` environment variable:

          ```bash
          export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
          sudo dnf install -y \
            nvidia-container-toolkit-${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
            nvidia-container-toolkit-base-${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
            libnvidia-container-tools-${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
            libnvidia-container1-${NVIDIA_CONTAINER_TOOLKIT_VERSION}
          ```

    ```bash
    sudo nvidia-ctk runtime configure --runtime=docker
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
- **Dockerfile-jetson-jetpack4:** Optimized for [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) devices running [NVIDIA JetPack 4](https://developer.nvidia.com/embedded/jetpack-sdk-461).
- **Dockerfile-jetson-jetpack5:** Optimized for [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) devices running [NVIDIA JetPack 5](https://developer.nvidia.com/embedded/jetpack-sdk-512).
- **Dockerfile-jetson-jetpack6:** Optimized for [NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/) devices running [NVIDIA JetPack 6](https://developer.nvidia.com/embedded/jetpack-sdk-61).
- **Dockerfile-jupyter:** For interactive development using JupyterLab in the browser.
- **Dockerfile-python:** Minimal Python environment for lightweight applications.
- **Dockerfile-conda:** Includes [Miniconda3](https://www.anaconda.com/docs/main) and Ultralytics package installed via Conda.

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
# Run without GPU
sudo docker run -it --ipc=host $t
```

### Using GPUs

```bash
# Run with all GPUs
sudo docker run -it --ipc=host --runtime=nvidia --gpus all $t

# Run specifying which GPUs to use
sudo docker run -it --ipc=host --runtime=nvidia --gpus '"device=2,3"' $t
```

The `-it` flag assigns a pseudo-TTY and keeps stdin open, allowing you to interact with the container. The `--ipc=host` flag enables sharing of host's IPC namespace, essential for sharing memory between processes. The `--gpus` flag allows the container to access the host's GPUs.

### Note on File Accessibility

To work with files on your local machine within the container, you can use Docker volumes:

```bash
# Mount a local directory into the container
sudo docker run -it --ipc=host --runtime=nvidia --gpus all -v /path/on/host:/path/in/container $t
```

Replace `/path/on/host` with the directory path on your local machine and `/path/in/container` with the desired path inside the Docker container.

## Run graphical user interface (GUI) applications in a Docker Container

!!! danger "Highly Experimental - User Assumes All Risk"

    The following instructions are experimental. Sharing a X11 socket with a Docker container poses potential security risks. Therefore, it's recommended to test this solution only in a controlled environment. For more information, refer to these resources on how to use `xhost`<sup>[(1)](http://users.stat.umn.edu/~geyer/secure.html)[(2)](https://linux.die.net/man/1/xhost)</sup>.

Docker is primarily used to containerize background applications and CLI programs, but it can also run graphical programs. In the Linux world, two main graphic servers handle graphical display: [X11](https://www.x.org/wiki/) (also known as the X Window System) and [Wayland](<https://en.wikipedia.org/wiki/Wayland_(protocol)>). Before starting, it's essential to determine which graphics server you are currently using. Run this command to find out:

```bash
env | grep -E -i 'x11|xorg|wayland'
```

Setup and configuration of an X11 or Wayland display server is outside the scope of this guide. If the above command returns nothing, then you'll need to start by getting either working for your system before continuing.

### Running a Docker Container with a GUI

!!! example

    ??? info "Use GPUs"
            If you're using [GPUs](#using-gpus), you can add the `--gpus all` flag to the command.

    ??? info "Docker runtime flag"
            If your Docker installation does not use the `nvidia` runtime by default, you can add the `--runtime=nvidia` flag to the command.

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

To set up Ultralytics with Docker, first ensure that Docker is installed on your system. If you have an NVIDIA GPU, install the [NVIDIA Container Toolkit](#installing-nvidia-container-toolkit) to enable GPU support. Then, pull the latest Ultralytics Docker image from Docker Hub using the following command:

```bash
sudo docker pull ultralytics/ultralytics:latest
```

For detailed steps, refer to our Docker Quickstart Guide.

### What are the benefits of using Ultralytics Docker images for machine learning projects?

Using Ultralytics Docker images ensures a consistent environment across different machines, replicating the same software and dependencies. This is particularly useful for [collaborating across teams](https://www.ultralytics.com/blog/how-ultralytics-integration-can-enhance-your-workflow), running models on various hardware, and maintaining reproducibility. For GPU-based training, Ultralytics provides optimized Docker images such as `Dockerfile` for general GPU usage and `Dockerfile-jetson` for NVIDIA Jetson devices. Explore [Ultralytics Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics) for more details.

### How can I run Ultralytics YOLO in a Docker container with GPU support?

First, ensure that the [NVIDIA Container Toolkit](#installing-nvidia-container-toolkit) is installed and configured. Then, use the following command to run Ultralytics YOLO with GPU support:

```bash
sudo docker run -it --ipc=host --runtime=nvidia --gpus all ultralytics/ultralytics:latest # all GPUs
```

This command sets up a Docker container with GPU access. For additional details, see the Docker Quickstart Guide.

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
sudo docker run -it --ipc=host --runtime=nvidia --gpus all -v /path/on/host:/path/in/container ultralytics/ultralytics:latest
```

Replace `/path/on/host` with the directory on your local machine and `/path/in/container` with the desired path inside the container. This setup allows you to work with your local files within the container. For more information, refer to the [Note on File Accessibility](#note-on-file-accessibility) section.
