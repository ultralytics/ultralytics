---
comments: true
description: Install and use YOLOv8 via CLI or Python. Run single-line commands or integrate with Python projects for object detection, segmentation, and classification.
keywords: YOLOv8, object detection, segmentation, classification, pip, git, CLI, Python
---

## Install Ultralytics

Ultralytics provides various installation methods including pip, conda, and Docker. Install YOLOv8 via the `ultralytics` pip package for the latest stable release or by cloning the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics) for the most up-to-date version. Docker can be used to execute the package in an isolated container, avoiding local installation.

!!! example "Install"

    === "Pip install (recommended)"
        Install the `ultralytics` package using pip, or update an existing installation by running `pip install -U ultralytics`. Visit the Python Package Index (PyPI) for more details on the `ultralytics` package: [https://pypi.org/project/ultralytics/](https://pypi.org/project/ultralytics/).

        [![PyPI version](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

        ```bash
        # Install the ultralytics package using pip
        pip install ultralytics
        ```
    
    === "Conda install"
        Conda is an alternative package manager to pip which may also be used for installation. Visit Anaconda for more details at [https://anaconda.org/conda-forge/ultralytics](https://anaconda.org/conda-forge/ultralytics). Ultralytics feedstock repository for updating the conda package is at [https://github.com/conda-forge/ultralytics-feedstock/](https://github.com/conda-forge/ultralytics-feedstock/).


        [![Conda Recipe](https://img.shields.io/badge/recipe-ultralytics-green.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)

        ```bash
        # Install the ultralytics package using conda
        conda install ultralytics
        ```
    
    === "Git clone"
        Clone the `ultralytics` repository if you are interested in contributing to the development or wish to experiment with the latest source code. After cloning, navigate into the directory and install the package in editable mode `-e` using pip.
        ```bash
        # Clone the ultralytics repository
        git clone https://github.com/ultralytics/ultralytics
        
        # Navigate to the cloned directory
        cd ultralytics
        
        # Install the package in editable mode for development
        pip install -e .
        ```

    === "Docker"
        Utilize Docker to execute the `ultralytics` package in an isolated container. By employing the official `ultralytics` image from [Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics), you can avoid local installation. Below are the commands to get the latest image and execute it:

        <a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Docker Pulls"></a>
    
        ```bash
        # Set image name as a variable
        t=ultralytics/ultralytics:latest
        
        # Pull the latest ultralytics image from Docker Hub
        sudo docker pull $t
        
        # Run the ultralytics image in a container with GPU support
        sudo docker run -it --ipc=host --gpus all $t
        ```
    
        The above command initializes a Docker container with the latest `ultralytics` image. The `-it` flag assigns a pseudo-TTY and maintains stdin open, enabling you to interact with the container. The `--ipc=host` flag sets the IPC (Inter-Process Communication) namespace to the host, which is essential for sharing memory between processes. The `--gpus all` flag enables access to all available GPUs inside the container, which is crucial for tasks that require GPU computation.
    
        Note: To work with files on your local machine within the container, use Docker volumes for mounting a local directory into the container:
    
        ```bash
        # Mount local directory to a directory inside the container
        sudo docker run -it --ipc=host --gpus all -v /path/on/host:/path/in/container $t
        ```
    
        Alter `/path/on/host` with the directory path on your local machine, and `/path/in/container` with the desired path inside the Docker container for accessibility.

See the `ultralytics` [requirements.txt](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) file for a list of dependencies. Note that all examples above install all required dependencies.

!!! tip "Tip"

    PyTorch requirements vary by operating system and CUDA requirements, so it's recommended to install PyTorch first following instructions at [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally).

    <a href="https://pytorch.org/get-started/locally/">
        <img width="800" alt="PyTorch Installation Instructions" src="https://user-images.githubusercontent.com/26833433/228650108-ab0ec98a-b328-4f40-a40d-95355e8a84e3.png">
    </a>

## Use Ultralytics with CLI

The Ultralytics command line interface (CLI) allows for simple single-line commands without the need for a Python environment.
CLI requires no customization or Python code. You can simply run all tasks from the terminal with the `yolo` command. Check out the [CLI Guide](usage/cli.md) to learn more about using YOLOv8 from the command line.

!!! example

    === "Syntax"

        Ultralytics `yolo` commands use the following syntax:
        ```bash
        yolo TASK MODE ARGS

        Where   TASK (optional) is one of [detect, segment, classify]
                MODE (required) is one of [train, val, predict, export, track]
                ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
        ```
        See all ARGS in the full [Configuration Guide](usage/cfg.md) or with `yolo cfg`

    === "Train"

        Train a detection model for 10 epochs with an initial learning_rate of 0.01
        ```bash
        yolo train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01
        ```

    === "Predict"

        Predict a YouTube video using a pretrained segmentation model at image size 320:
        ```bash
        yolo predict model=yolov8n-seg.pt source='https://youtu.be/Zgi9g1ksQHc' imgsz=320
        ```

    === "Val"

        Val a pretrained detection model at batch-size 1 and image size 640:
        ```bash
        yolo val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640
        ```

    === "Export"

        Export a YOLOv8n classification model to ONNX format at image size 224 by 128 (no TASK required)
        ```bash
        yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128
        ```

    === "Special"

        Run special commands to see version, view settings, run checks and more:
        ```bash
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        ```

!!! warning "Warning"

    Arguments must be passed as `arg=val` pairs, split by an equals `=` sign and delimited by spaces ` ` between pairs. Do not use `--` argument prefixes or commas `,` between arguments.

    - `yolo predict model=yolov8n.pt imgsz=640 conf=0.25` &nbsp; ✅
    - `yolo predict model yolov8n.pt imgsz 640 conf 0.25` &nbsp; ❌
    - `yolo predict --model yolov8n.pt --imgsz 640 --conf 0.25` &nbsp; ❌

[CLI Guide](usage/cli.md){ .md-button .md-button--primary}

## Use Ultralytics with Python

YOLOv8's Python interface allows for seamless integration into your Python projects, making it easy to load, run, and process the model's output. Designed with simplicity and ease of use in mind, the Python interface enables users to quickly implement object detection, segmentation, and classification in their projects. This makes YOLOv8's Python interface an invaluable tool for anyone looking to incorporate these functionalities into their Python projects.

For example, users can load a model, train it, evaluate its performance on a validation set, and even export it to ONNX format with just a few lines of code. Check out the [Python Guide](usage/python.md) to learn more about using YOLOv8 within your Python projects.

!!! example

    ```python
    from ultralytics import YOLO
    
    # Create a new YOLO model from scratch
    model = YOLO('yolov8n.yaml')
    
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')
    
    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data='coco128.yaml', epochs=3)
    
    # Evaluate the model's performance on the validation set
    results = model.val()
    
    # Perform object detection on an image using the model
    results = model('https://ultralytics.com/images/bus.jpg')
    
    # Export the model to ONNX format
    success = model.export(format='onnx')
    ```

[Python Guide](usage/python.md){.md-button .md-button--primary}