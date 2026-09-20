---
title: Install YOLO with Conda
comments: true
description: Install Ultralytics YOLO with Conda. Set up an isolated conda-forge environment, add CUDA GPU support, run the Conda Docker image, and speed up installs with the libmamba solver.
keywords: Ultralytics, YOLO, Conda, conda-forge, install Ultralytics, conda environment, CUDA, GPU, pytorch-gpu, Miniconda, Anaconda, libmamba solver, Conda Docker image, machine learning, environment management
---

# How to Install Ultralytics YOLO with Conda

<p align="center">
  <img width="800" src="https://cdn.ul.run/i/59a3c831a2309a5822e4d5f5be52dbe5.avif" alt="Ultralytics Conda Package Visual">
</p>

This guide walks through setting up a Conda environment for your Ultralytics projects. Conda is an open-source package and environment management system that offers an excellent alternative to pip for installing packages and dependencies. Its isolated environments make it particularly well-suited for data science and [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) work. For more details, visit the Ultralytics Conda package on [Anaconda](https://anaconda.org/conda-forge/ultralytics) and check out the Ultralytics feedstock repository for package updates on [GitHub](https://github.com/conda-forge/ultralytics-feedstock/).

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ultralytics?logo=condaforge)](https://anaconda.org/conda-forge/ultralytics)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)
[![Conda Recipe](https://img.shields.io/badge/recipe-ultralytics-green.svg)](https://anaconda.org/conda-forge/ultralytics)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)

This guide covers how to [create an environment](#setting-up-a-conda-environment), [install Ultralytics](#installing-ultralytics), [run inference](#using-ultralytics), use the [Conda Docker image](#ultralytics-conda-docker-image), and [speed up installs with libmamba](#speeding-up-installation-with-libmamba).

## Prerequisites

You should have Anaconda or Miniconda installed on your system. If not, download and install it from [Anaconda](https://www.anaconda.com/) or [Miniconda](https://www.anaconda.com/docs/main).

## Setting up a Conda Environment

First, create a new Conda environment. Open your terminal and run the following command:

```bash
conda create --name ultralytics-env python=3.11 -y
```

Activate the new environment:

```bash
conda activate ultralytics-env
```

## Installing Ultralytics

You can install the Ultralytics package from the conda-forge channel. Execute the following command:

```bash
conda install -c conda-forge ultralytics
```

!!! note "Installing in a CUDA environment"

    If you're working in a CUDA-enabled environment, install `ultralytics` together with the conda-forge `pytorch-gpu` metapackage so the Conda package manager resolves a CUDA-enabled PyTorch build alongside it:

    ```bash
    conda install -c conda-forge ultralytics pytorch-gpu
    ```

## Using Ultralytics

With Ultralytics installed, you can now start using its robust features for [object detection](https://www.ultralytics.com/glossary/object-detection), [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation), and more. For example, to predict an image, you can run:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # initialize model
results = model("path/to/image.jpg")  # perform inference
results[0].show()  # display results for the first image
```

## Ultralytics Conda Docker Image

The [Conda Dockerfile](https://github.com/ultralytics/ultralytics/blob/main/docker/Dockerfile-conda) uses conda-forge PyTorch 2.13, torchvision 0.28, and CUDA 13.0. Conda-forge releases can lag the PyTorch pip releases used by the standard GPU image. Automated publishing of `latest-conda` is disabled, so build the current image locally:

```bash
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
t=ultralytics-conda:local
sudo docker build -f docker/Dockerfile-conda -t $t .
```

Run the image:

```bash
# Run the Ultralytics image in a container with GPU support
sudo docker run -it --ipc=host --device nvidia.com/gpu=all $t                         # all GPUs
sudo docker run -it --ipc=host --device nvidia.com/gpu=2 --device nvidia.com/gpu=3 $t # specify GPUs
```

On Linux, CDI device requests require Docker >= 28.2.0 and NVIDIA Container Toolkit >= 1.18. The legacy `--gpus all` flag can lose GPU access after host daemon reloads, so upgrade older Linux hosts and use `--device` instead. See the [Docker Quickstart Guide](docker-quickstart.md) for details.

## Speeding Up Installation with Libmamba

[`libmamba`](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) is a fast, cross-platform, dependency-aware solver that replaces Conda's classic solver. Conda 23.10 and later already use `libmamba` as the default solver, so most installations are faster out of the box.

If you're on an older Conda version, you can enable `libmamba` manually:

1. First, install the `conda-libmamba-solver` package:

    ```bash
    conda install conda-libmamba-solver
    ```

2. Next, configure Conda to use `libmamba` as the solver:

    ```bash
    conda config --set solver libmamba
    ```

You have successfully set up a Conda environment, installed the Ultralytics package, and are now ready to explore its features. For more advanced tutorials and examples, see the [Ultralytics documentation](../index.md).

## FAQ

### What is the process for setting up a Conda environment for Ultralytics projects?

Setting up a Conda environment for Ultralytics projects is straightforward and ensures smooth package management. First, create a new Conda environment using the following command:

```bash
conda create --name ultralytics-env python=3.11 -y
```

Then, activate the new environment with:

```bash
conda activate ultralytics-env
```

Finally, install Ultralytics from the conda-forge channel:

```bash
conda install -c conda-forge ultralytics
```

### Why should I use Conda over pip for managing dependencies in Ultralytics projects?

Conda is a robust package and environment management system that offers several advantages over pip. It manages dependencies efficiently and ensures that all necessary libraries are compatible. Conda's isolated environments prevent conflicts between packages, which is crucial in data science and machine learning projects. Additionally, Conda supports binary package distribution, speeding up the installation process.

### Can I use Ultralytics YOLO in a CUDA-enabled environment for faster performance?

Yes, you can enhance performance by utilizing a CUDA-enabled environment. Install `ultralytics` together with the conda-forge `pytorch-gpu` metapackage so a CUDA-enabled PyTorch build is resolved alongside it:

```bash
conda install -c conda-forge ultralytics pytorch-gpu
```

This setup enables GPU acceleration, crucial for intensive tasks like [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) model training and inference. For more information, visit the [Ultralytics installation guide](../quickstart.md).

### What are the benefits of using Ultralytics Docker images with a Conda environment?

The Conda Dockerfile provides an isolated conda-forge environment with CUDA-enabled PyTorch. Follow the [local build instructions](#ultralytics-conda-docker-image), then run it with the GPU options shown there. The standard published GPU image and other alternatives are listed in the [Docker guide](docker-quickstart.md#installing-ultralytics-docker-images).

### How can I speed up Conda package installation in my Ultralytics environment?

Conda 23.10 and later already use the fast `libmamba` solver by default. On older Conda versions, you can enable it manually by first installing the `conda-libmamba-solver` package:

```bash
conda install conda-libmamba-solver
```

Then configure Conda to use `libmamba` as the solver:

```bash
conda config --set solver libmamba
```

This setup provides faster and more efficient package management. For more tips on optimizing your environment, read about [libmamba installation](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community).
