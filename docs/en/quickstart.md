---
comments: true
description: Install Ultralytics YOLO with pip, conda, Docker, or from source, then run your first prediction with a pretrained YOLO26 model from the CLI or Python.
keywords: Install Ultralytics, Ultralytics, YOLO26, YOLO11, pip install ultralytics, conda, Docker, quickstart, yolo predict, object detection
---

# Install Ultralytics

Install the `ultralytics` package with pip, conda, or Docker, or from source, then run your first prediction with a pretrained [YOLO26](models/yolo26.md) model. Every method installs all required dependencies listed in [pyproject.toml](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml).

!!! example "Install"

    === "Pip (recommended)"

        Install or update the [ultralytics](https://pypi.org/project/ultralytics/) package in a Python>=3.8 environment with PyTorch>=1.8:

        ```bash
        pip install -U ultralytics
        ```

        For the latest development version, install directly from the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics):

        ```bash
        pip install git+https://github.com/ultralytics/ultralytics.git@main
        ```

    === "Conda"

        Install from [conda-forge](https://anaconda.org/conda-forge/ultralytics):

        ```bash
        conda install -c conda-forge ultralytics
        ```

        In a CUDA environment, install `ultralytics` together with the `pytorch-gpu` metapackage in the same command so conda resolves a CUDA-enabled PyTorch build. PyTorch no longer publishes new releases to the `pytorch` conda channel, so install everything from conda-forge:

        ```bash
        conda install -c conda-forge ultralytics pytorch-gpu
        ```

        See the [Conda Quickstart Guide](guides/conda-quickstart.md) for environment setup, the libmamba solver, and the Conda Docker image.

    === "Docker"

        Run the package in an isolated container with the official images on [Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics):

        ```bash
        # Pull the latest ultralytics image and run it with GPU support
        sudo docker pull ultralytics/ultralytics:latest
        sudo docker run -it --ipc=host --device nvidia.com/gpu=all ultralytics/ultralytics:latest
        ```

        The default AMD64 GPU image uses PyTorch 2.14 and CUDA 13.2. On Linux, CDI device requests require Docker >= 28.2.0 and NVIDIA Container Toolkit >= 1.18; the legacy `--gpus all` flag can lose GPU access after host daemon reloads, so use `--device` instead. See the [Docker Quickstart Guide](guides/docker-quickstart.md) for the [full image table](guides/docker-quickstart.md#installing-ultralytics-docker-images) (CPU, export, ARM64, and JetPack variants), host driver requirements, and volume mounts.

    === "Git clone"

        Clone the repository and install it in editable mode (`-e`) to run the latest source or develop locally:

        ```bash
        git clone https://github.com/ultralytics/ultralytics
        cd ultralytics
        pip install -e .
        ```

        To work from a fork or pin a custom branch in another project, see [Development Installation](help/contributing.md#development-installation).

    === "Headless"

        On servers without a display (cloud VMs, containers, CI pipelines), install the variant that depends on `opencv-python-headless` to avoid `libGL` errors:

        ```bash
        pip install ultralytics-opencv-headless
        ```

        It provides the same functionality and API as `ultralytics`, minus OpenCV's GUI components.

!!! tip

    [PyTorch](https://www.ultralytics.com/glossary/pytorch) requirements vary by operating system and CUDA version. To use a specific build, install PyTorch first by following the [PyTorch installation instructions](https://pytorch.org/get-started/locally/), then install `ultralytics`.

## Use Ultralytics with CLI

Run your first prediction from the terminal with the `yolo` command:

```bash
yolo predict model=yolo26n.pt
```

The pretrained `yolo26n.pt` weights download automatically, the model runs on two bundled sample images, and the command prints where it saved the annotated results, `runs/detect/predict` on a first run. Point `source` at your own image, video, directory, URL, or stream, or at a webcam with `source=0`:

```bash
yolo predict model=yolo26n.pt source=0 show=True
```

`predict` is the **mode**, what to do with the model: `train`, `val`, `predict`, `export`, `track`, or `benchmark`. The **task** (`detect`, `segment`, `semantic`, `depth`, `classify`, `pose`, or `obb`) is read from the model file, so the `yolo TASK MODE ARGS` syntax usually needs only the mode plus `arg=value` pairs such as `imgsz=640`. See the [CLI Guide](usage/cli.md) for every mode and the [Configuration](usage/cfg.md) page for all arguments.

## Use Ultralytics with Python

The same prediction in Python:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # load a pretrained YOLO26n detection model
results = model("https://ultralytics.com/images/bus.jpg", save=True)  # predict and save the annotated image
```

`results` is a list of [Results](modes/predict.md#working-with-results) objects, one per image, carrying boxes, masks, keypoints, or class probabilities depending on the task. The [Python Guide](usage/python.md) covers training, validation, export, and tracking with the same `YOLO` class.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/_a7cVL9hqnk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics YOLO Quick Start Guide
</p>

## Ultralytics Settings

Persistent settings such as the datasets, weights, and runs directories, the [Ultralytics Platform](https://platform.ultralytics.com) API key, and experiment-logger toggles live in a JSON file managed with `yolo settings`. See the [Settings](usage/settings.md) page to view, change, or reset them.

## What's Next

Browse the [modes](modes/index.md) YOLO runs in and the [YOLO26](models/yolo26.md) model sizes, then [train on your own data](modes/train.md) after formatting it with the [Datasets guide](datasets/index.md).

## FAQ

### How do I install Ultralytics using pip?

Install Ultralytics with pip using:

```bash
pip install -U ultralytics
```

This installs the latest stable release of the `ultralytics` package from [PyPI](https://pypi.org/project/ultralytics/). To install the development version directly from GitHub:

```bash
pip install git+https://github.com/ultralytics/ultralytics.git
```

Ensure the Git command-line tool is installed on your system.

### Can I install Ultralytics YOLO using conda?

Yes, install Ultralytics YOLO using conda with:

```bash
conda install -c conda-forge ultralytics
```

This method is a great alternative to pip, ensuring compatibility with other packages. For CUDA environments, install `ultralytics` together with the `pytorch-gpu` metapackage so conda selects a CUDA-enabled PyTorch build:

```bash
conda install -c conda-forge ultralytics pytorch-gpu
```

For more instructions, see the [Conda quickstart guide](guides/conda-quickstart.md).

### What are the advantages of using Docker to run Ultralytics YOLO?

Docker provides an isolated, consistent environment for Ultralytics YOLO, ensuring smooth performance across systems and avoiding local installation complexities. Official Docker images are available on [Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics), with variants for GPU, CPU, ARM64, [NVIDIA Jetson](guides/nvidia-jetson.md), and Conda. To pull and run the latest image:

```bash
# Pull the latest ultralytics image from Docker Hub
sudo docker pull ultralytics/ultralytics:latest

# Run the ultralytics image in a container with GPU support
sudo docker run -it --ipc=host --device nvidia.com/gpu=all ultralytics/ultralytics:latest
```

On Linux, CDI device requests require Docker >= 28.2.0 and NVIDIA Container Toolkit >= 1.18. For detailed Docker instructions, see the [Docker quickstart guide](guides/docker-quickstart.md).

### How do I clone the Ultralytics repository for development?

Clone the Ultralytics repository and set up a development environment with:

```bash
# Clone the ultralytics repository
git clone https://github.com/ultralytics/ultralytics

# Navigate to the cloned directory
cd ultralytics

# Install the package in editable mode for development
pip install -e .
```

This allows contributions to the project or experimentation with the latest source code. For forks and pinning a custom branch, see [Development Installation](help/contributing.md#development-installation).

### Why should I use Ultralytics YOLO CLI?

The Ultralytics YOLO CLI simplifies running object detection tasks without Python code, enabling single-line commands for training, validation, and prediction directly from your terminal. The basic syntax is:

```bash
yolo TASK MODE ARGS
```

For example, to train a detection model:

```bash
yolo train data=coco8.yaml model=yolo26n.pt epochs=10 lr0=0.01
```

Explore more commands and usage examples in the full [CLI Guide](usage/cli.md).
