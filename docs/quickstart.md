---
comments: true
description: Install and use YOLOv8 via CLI or Python. Run single-line commands or integrate with Python projects for object detection, segmentation, and classification.
keywords: YOLOv8, object detection, segmentation, classification, pip, git, CLI, Python
---

## Install

Install YOLOv8 via the `ultralytics` pip package for the latest stable release or by cloning
the [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) repository for the most
up-to-date version.

!!! example "Install"

    === "pip install (recommended)"
        ```bash
        pip install ultralytics
        ```

    === "git clone (for development)"
        ```bash
        git clone https://github.com/ultralytics/ultralytics
        cd ultralytics
        pip install -e .
        ```

See the `ultralytics` [requirements.txt](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) file for a list of dependencies. Note that `pip` automatically installs all required dependencies.

!!! tip "Tip"

    PyTorch requirements vary by operating system and CUDA requirements, so it's recommended to install PyTorch first following instructions at [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally).

    <a href="https://pytorch.org/get-started/locally/">
        <img width="800" alt="PyTorch Installation Instructions" src="https://user-images.githubusercontent.com/26833433/228650108-ab0ec98a-b328-4f40-a40d-95355e8a84e3.png">
    </a>

## Use with CLI

The YOLO command line interface (CLI) allows for simple single-line commands without the need for a Python environment.
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

## Use with Python

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