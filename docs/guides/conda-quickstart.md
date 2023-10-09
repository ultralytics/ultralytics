---
comments: true
description: Comprehensive guide to setting up and using Ultralytics YOLO models in a Conda environment. Learn how to install the package, manage dependencies, and get started with object detection projects.
keywords: Ultralytics, YOLO, Conda, environment setup, object detection, package installation, deep learning, machine learning, guide
---

# Conda Quickstart Guide for Ultralytics

<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/26833433/266324397-32119e21-8c86-43e5-a00e-79827d303d10.png" alt="Ultralytics Conda Package Visual">
</p>

This guide provides a comprehensive introduction to setting up a Conda environment for your Ultralytics projects. Conda is an open-source package and environment management system that offers an excellent alternative to pip for installing packages and dependencies. Its isolated environments make it particularly well-suited for data science and machine learning endeavors. For more details, visit the Ultralytics Conda package on [Anaconda](https://anaconda.org/conda-forge/ultralytics) and check out the Ultralytics feedstock repository for package updates on [GitHub](https://github.com/conda-forge/ultralytics-feedstock/).

[![Conda Recipe](https://img.shields.io/badge/recipe-ultralytics-green.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)

## What You Will Learn

- Setting up a Conda environment
- Installing Ultralytics via Conda
- Initializing Ultralytics in your environment
- Using Ultralytics Docker images with Conda

---

## Prerequisites

- You should have Anaconda or Miniconda installed on your system. If not, download and install it from [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/).

---

## Setting up a Conda Environment

First, let's create a new Conda environment. Open your terminal and run the following command:

```bash
conda create --name ultralytics-env python=3.8 -y
```

Activate the new environment:

```bash
conda activate ultralytics-env
```

---

## Installing Ultralytics

You can install the Ultralytics package from the conda-forge channel. Execute the following command:

```bash
conda install -c conda-forge ultralytics
```

### Note on CUDA Environment

If you're working in a CUDA-enabled environment, it's a good practice to install `ultralytics`, `pytorch`, and `pytorch-cuda` together to resolve any conflicts:

```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```

---

## Using Ultralytics

With Ultralytics installed, you can now start using its robust features for object detection, instance segmentation, and more. For example, to predict an image, you can run:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # initialize model
results = model('path/to/image.jpg')  # perform inference
results.show()  # display results
```

---

## Ultralytics Conda Docker Image

If you prefer using Docker, Ultralytics offers Docker images with a Conda environment included. You can pull these images from [DockerHub](https://hub.docker.com/r/ultralytics/ultralytics).

Pull the latest Ultralytics image:

```bash
# Set image name as a variable
t=ultralytics/ultralytics:latest-conda

# Pull the latest Ultralytics image from Docker Hub
sudo docker pull $t
```

Run the image:

```bash
# Run the Ultralytics image in a container with GPU support
sudo docker run -it --ipc=host --gpus all $t  # all GPUs
sudo docker run -it --ipc=host --gpus '"device=2,3"' $t  # specify GPUs
```

---

Certainly, you can include the following section in your Conda guide to inform users about speeding up installation using `libmamba`:

---

## Speeding Up Installation with Libmamba

If you're looking to [speed up the package installation](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) process in Conda, you can opt to use `libmamba`, a fast, cross-platform, and dependency-aware package manager that serves as an alternative solver to Conda's default.

### How to Enable Libmamba

To enable `libmamba` as the solver for Conda, you can perform the following steps:

1. First, install the `conda-libmamba-solver` package. This can be skipped if your Conda version is 4.11 or above, as `libmamba` is included by default.

    ```bash
    conda install conda-libmamba-solver
    ```

2. Next, configure Conda to use `libmamba` as the solver:

    ```bash
    conda config --set solver libmamba
    ```

And that's it! Your Conda installation will now use `libmamba` as the solver, which should result in a faster package installation process.

---

Congratulations! You have successfully set up a Conda environment, installed the Ultralytics package, and are now ready to explore its rich functionalities. Feel free to dive deeper into the [Ultralytics documentation](https://docs.ultralytics.com/) for more advanced tutorials and examples.
