---
comments: true
description: Learn to set up a Conda environment for Ultralytics projects. Follow our comprehensive guide for easy installation and initialization.
keywords: Ultralytics, Conda, setup, installation, environment, guide, machine learning, data science
---

# Conda Quickstart Guide for Ultralytics

<p align="center">
  <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-conda-package-visual.avif" alt="Ultralytics Conda Package Visual">
</p>

This guide provides a comprehensive introduction to setting up a Conda environment for your Ultralytics projects. Conda is an open-source package and environment management system that offers an excellent alternative to pip for installing packages and dependencies. Its isolated environments make it particularly well-suited for data science and [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) endeavors. For more details, visit the Ultralytics Conda package on [Anaconda](https://anaconda.org/conda-forge/ultralytics) and check out the Ultralytics feedstock repository for package updates on [GitHub](https://github.com/conda-forge/ultralytics-feedstock/).

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ultralytics?logo=condaforge)](https://anaconda.org/conda-forge/ultralytics)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)
[![Conda Recipe](https://img.shields.io/badge/recipe-ultralytics-green.svg)](https://anaconda.org/conda-forge/ultralytics)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)

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
conda create --name ultralytics-env python=3.11 -y
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

With Ultralytics installed, you can now start using its robust features for [object detection](https://www.ultralytics.com/glossary/object-detection), [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation), and more. For example, to predict an image, you can run:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # initialize model
results = model("path/to/image.jpg")  # perform inference
results[0].show()  # display results for the first image
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

Congratulations! You have successfully set up a Conda environment, installed the Ultralytics package, and are now ready to explore its rich functionalities. Feel free to dive deeper into the [Ultralytics documentation](../index.md) for more advanced tutorials and examples.

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

Yes, you can enhance performance by utilizing a CUDA-enabled environment. Ensure that you install `ultralytics`, `pytorch`, and `pytorch-cuda` together to avoid conflicts:

```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```

This setup enables GPU acceleration, crucial for intensive tasks like [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) model training and inference. For more information, visit the [Ultralytics installation guide](../quickstart.md).

### What are the benefits of using Ultralytics Docker images with a Conda environment?

Using Ultralytics Docker images ensures a consistent and reproducible environment, eliminating "it works on my machine" issues. These images include a pre-configured Conda environment, simplifying the setup process. You can pull and run the latest Ultralytics Docker image with the following commands:

```bash
sudo docker pull ultralytics/ultralytics:latest-conda
sudo docker run -it --ipc=host --gpus all ultralytics/ultralytics:latest-conda
```

This approach is ideal for deploying applications in production or running complex workflows without manual configuration. Learn more about [Ultralytics Conda Docker Image](../quickstart.md).

### How can I speed up Conda package installation in my Ultralytics environment?

You can speed up the package installation process by using `libmamba`, a fast dependency solver for Conda. First, install the `conda-libmamba-solver` package:

```bash
conda install conda-libmamba-solver
```

Then configure Conda to use `libmamba` as the solver:

```bash
conda config --set solver libmamba
```

This setup provides faster and more efficient package management. For more tips on optimizing your environment, read about [libmamba installation](../quickstart.md).
