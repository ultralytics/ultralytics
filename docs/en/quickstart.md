---
comments: true
description: Install Ultralytics YOLO and run your first prediction in under a minute. Covers pip, conda, and Docker installs, then the CLI and Python API.
keywords: Ultralytics, YOLO26, YOLO11, quickstart, install Ultralytics, pip, conda, Docker, yolo predict, machine learning, object detection
---

# Quickstart

Install Ultralytics and run your first prediction in under a minute — no dataset or training required.

## Install

Install the `ultralytics` package with [pip](https://pypi.org/project/ultralytics/):

```bash
pip install -U ultralytics
```

[PyTorch](https://www.ultralytics.com/glossary/pytorch) requirements vary by operating system and CUDA version, so install PyTorch first if you need a specific build — see the [PyTorch install instructions](https://pytorch.org/get-started/locally/).

??? example "Other installation methods: conda, Docker, git, headless servers"

    **Conda**

    Conda is an alternative to pip. For details, see [Anaconda](https://anaconda.org/conda-forge/ultralytics).

    ```bash
    conda install -c conda-forge ultralytics
    ```

    If you are installing in a CUDA environment, install `ultralytics`, `pytorch`, and `pytorch-cuda` together so conda can resolve conflicts:

    ```bash
    conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=12.1 ultralytics
    ```

    See the [Conda Quickstart Guide](guides/conda-quickstart.md) for Conda Docker images and further details.

    **Docker**

    Pull and run an official [Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics) image — GPU, CPU-only, ARM64, [Jetson](guides/nvidia-jetson.md), and Conda variants are all available:

    ```bash
    # Pull the latest ultralytics image from Docker Hub
    sudo docker pull ultralytics/ultralytics:latest

    # Run the image in a container with GPU support
    sudo docker run -it --ipc=host --device nvidia.com/gpu=all ultralytics/ultralytics:latest
    ```

    CDI device requests require Docker >= 28.2.0 and NVIDIA Container Toolkit >= 1.18. On older hosts, use the legacy `--runtime=nvidia --gpus all` flags instead. See the [Docker Quickstart Guide](guides/docker-quickstart.md) for volume mounts and all six image variants.

    **From source (development)**

    ```bash
    git clone https://github.com/ultralytics/ultralytics
    cd ultralytics
    pip install -e .
    ```

    For persistent custom modifications or contributing back, see [Development Installation](help/contributing.md#-development-installation).

    **Headless servers**

    For environments without a display (cloud VMs, CI/CD pipelines), install the `opencv-python-headless`-based variant to avoid `libGL` errors:

    ```bash
    pip install ultralytics-opencv-headless
    ```

    Both packages provide the same functionality and API.

## Run Your First Prediction

Run inference from the terminal with the `yolo` command:

```bash
yolo predict model=yolo26n.pt
```

`predict` here is the **mode** — what to do with the model. Modes are `train`, `val`, `predict`, `export`, `track`, and `benchmark`. YOLO infers the **task** (`detect`, `segment`, `semantic`, `classify`, `pose`, `obb`) from the model file itself, so you rarely need to set it explicitly.

This downloads the pretrained `yolo26n.pt` checkpoint automatically, runs it on the bundled example images, and saves the annotated results to `runs/detect/predict/`.

The same prediction in Python:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # load a pretrained detection model
results = model("https://ultralytics.com/images/bus.jpg")  # run inference
```

## Run Your First Training

Train the same model on the bundled [COCO8](datasets/detect/coco8.md) dataset for 3 epochs:

```bash
yolo train model=yolo26n.pt data=coco8.yaml epochs=3
```

Swap `coco8.yaml` for your own dataset's YAML to train on your data — see [Train mode](modes/train.md) and [Datasets](datasets/index.md) for how to format and point to it.

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

## Next Steps

Check out [YOLO26](models/yolo26.md) for benchmarks and every model variant, or browse [Tasks](tasks/index.md) to find the one that matches your problem — detection, segmentation, pose, OBB, or classification. To train on your own data, format it with the [Datasets guide](datasets/index.md) and run [Train mode](modes/train.md). For deeper API coverage beyond this page, see the [Python Guide](usage/python.md) and [CLI Guide](usage/cli.md), or skip straight to a ready-made pipeline with [Solutions](solutions/index.md).

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

This method is a great alternative to pip, ensuring compatibility with other packages. For CUDA environments, install `ultralytics`, `pytorch`, and `pytorch-cuda` together to resolve conflicts:

```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=12.1 ultralytics
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

For detailed Docker instructions, see the [Docker quickstart guide](guides/docker-quickstart.md).

### Why should I use the Ultralytics YOLO CLI?

The Ultralytics YOLO CLI runs detection, training, validation, and export tasks with a single terminal command — no Python code required. The basic syntax is `yolo TASK MODE ARGS`, for example:

```bash
yolo train data=coco8.yaml model=yolo26n.pt epochs=10 lr0=0.01
```

Explore the full grammar and every mode's arguments in the [CLI Guide](usage/cli.md).
