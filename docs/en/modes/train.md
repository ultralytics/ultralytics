---
comments: true
description: Step-by-step guide to train YOLOv8 models with Ultralytics YOLO including examples of single-GPU and multi-GPU training
keywords: Ultralytics, YOLOv8, YOLO, object detection, train mode, custom dataset, GPU training, multi-GPU, hyperparameters, CLI examples, Python examples
---

# Model Training with Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

Training a deep learning model involves feeding it data and adjusting its parameters so that it can make accurate predictions. Train mode in Ultralytics YOLOv8 is engineered for effective and efficient training of object detection models, fully utilizing modern hardware capabilities. This guide aims to cover all the details you need to get started with training your own models using YOLOv8's robust set of features.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train a YOLOv8 model on Your Custom Dataset in Google Colab.
</p>

## Why Choose Ultralytics YOLO for Training?

Here are some compelling reasons to opt for YOLOv8's Train mode:

- **Efficiency:** Make the most out of your hardware, whether you're on a single-GPU setup or scaling across multiple GPUs.
- **Versatility:** Train on custom datasets in addition to readily available ones like COCO, VOC, and ImageNet.
- **User-Friendly:** Simple yet powerful CLI and Python interfaces for a straightforward training experience.
- **Hyperparameter Flexibility:** A broad range of customizable hyperparameters to fine-tune model performance.

### Key Features of Train Mode

The following are some notable features of YOLOv8's Train mode:

- **Automatic Dataset Download:** Standard datasets like COCO, VOC, and ImageNet are downloaded automatically on first use.
- **Multi-GPU Support:** Scale your training efforts seamlessly across multiple GPUs to expedite the process.
- **Hyperparameter Configuration:** The option to modify hyperparameters through YAML configuration files or CLI arguments.
- **Visualization and Monitoring:** Real-time tracking of training metrics and visualization of the learning process for better insights.

!!! tip "Tip"

    * YOLOv8 datasets like COCO, VOC, ImageNet and many others automatically download on first use, i.e. `yolo train data=coco.yaml`

## Usage Examples

Train YOLOv8n on the COCO128 dataset for 100 epochs at image size 640. The training device can be specified using the `device` argument. If no argument is passed GPU `device=0` will be used if available, otherwise `device=cpu` will be used. See Arguments section below for a full list of training arguments.

!!! example "Single-GPU and CPU Training Example"

    Device is determined automatically. If a GPU is available then it will be used, otherwise training will start on CPU.

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.yaml')  # build a new model from YAML
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

        # Train the model
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### Multi-GPU Training

Multi-GPU training allows for more efficient utilization of available hardware resources by distributing the training load across multiple GPUs. This feature is available through both the Python API and the command-line interface. To enable multi-GPU training, specify the GPU device IDs you wish to use.

!!! example "Multi-GPU Training Example"

    To train with 2 GPUs, CUDA devices 0 and 1 use the following commands. Expand to additional GPUs as required.

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

        # Train the model with 2 GPUs
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device=[0, 1])
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model using GPUs 0 and 1
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0,1
        ```

### Apple M1 and M2 MPS Training

With the support for Apple M1 and M2 chips integrated in the Ultralytics YOLO models, it's now possible to train your models on devices utilizing the powerful Metal Performance Shaders (MPS) framework. The MPS offers a high-performance way of executing computation and image processing tasks on Apple's custom silicon.

To enable training on Apple M1 and M2 chips, you should specify 'mps' as your device when initiating the training process. Below is an example of how you could do this in Python and via the command line:

!!! example "MPS Training Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

        # Train the model with 2 GPUs
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device='mps')
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model using GPUs 0 and 1
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=mps
        ```

While leveraging the computational power of the M1/M2 chips, this enables more efficient processing of the training tasks. For more detailed guidance and advanced configuration options, please refer to the [PyTorch MPS documentation](https://pytorch.org/docs/stable/notes/mps.html).

### Resuming Interrupted Trainings

Resuming training from a previously saved state is a crucial feature when working with deep learning models. This can come in handy in various scenarios, like when the training process has been unexpectedly interrupted, or when you wish to continue training a model with new data or for more epochs.

When training is resumed, Ultralytics YOLO loads the weights from the last saved model and also restores the optimizer state, learning rate scheduler, and the epoch number. This allows you to continue the training process seamlessly from where it was left off.

You can easily resume training in Ultralytics YOLO by setting the `resume` argument to `True` when calling the `train` method, and specifying the path to the `.pt` file containing the partially trained model weights.

Below is an example of how to resume an interrupted training using Python and via the command line:

!!! example "Resume Training Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('path/to/last.pt')  # load a partially trained model

        # Resume training
        results = model.train(resume=True)
        ```

    === "CLI"

        ```bash
        # Resume an interrupted training
        yolo train resume model=path/to/last.pt
        ```

By setting `resume=True`, the `train` function will continue training from where it left off, using the state stored in the 'path/to/last.pt' file. If the `resume` argument is omitted or set to `False`, the `train` function will start a new training session.

Remember that checkpoints are saved at the end of every epoch by default, or at fixed interval using the `save_period` argument, so you must complete at least 1 epoch to resume a training run.

## Arguments

Training settings for YOLO models refer to the various hyperparameters and configurations used to train the model on a dataset. These settings can affect the model's performance, speed, and accuracy. Some common YOLO training settings include the batch size, learning rate, momentum, and weight decay. Other factors that may affect the training process include the choice of optimizer, the choice of loss function, and the size and composition of the training dataset. It is important to carefully tune and experiment with these settings to achieve the best possible performance for a given task.

| Key               | Value    | Description                                                                                    |
|-------------------|----------|------------------------------------------------------------------------------------------------|
| `model`           | `None`   | path to model file, i.e. yolov8n.pt, yolov8n.yaml                                              |
| `data`            | `None`   | path to data file, i.e. coco128.yaml                                                           |
| `epochs`          | `100`    | number of epochs to train for                                                                  |
| `patience`        | `50`     | epochs to wait for no observable improvement for early stopping of training                    |
| `batch`           | `16`     | number of images per batch (-1 for AutoBatch)                                                  |
| `imgsz`           | `640`    | size of input images as integer                                                                |
| `save`            | `True`   | save train checkpoints and predict results                                                     |
| `save_period`     | `-1`     | Save checkpoint every x epochs (disabled if < 1)                                               |
| `cache`           | `False`  | True/ram, disk or False. Use cache for data loading                                            |
| `device`          | `None`   | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu                           |
| `workers`         | `8`      | number of worker threads for data loading (per RANK if DDP)                                    |
| `project`         | `None`   | project name                                                                                   |
| `name`            | `None`   | experiment name                                                                                |
| `exist_ok`        | `False`  | whether to overwrite existing experiment                                                       |
| `pretrained`      | `True`   | (bool or str) whether to use a pretrained model (bool) or a model to load weights from (str)   |
| `optimizer`       | `'auto'` | optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]              |
| `verbose`         | `False`  | whether to print verbose output                                                                |
| `seed`            | `0`      | random seed for reproducibility                                                                |
| `deterministic`   | `True`   | whether to enable deterministic mode                                                           |
| `single_cls`      | `False`  | train multi-class data as single-class                                                         |
| `rect`            | `False`  | rectangular training with each batch collated for minimum padding                              |
| `cos_lr`          | `False`  | use cosine learning rate scheduler                                                             |
| `close_mosaic`    | `10`     | (int) disable mosaic augmentation for final epochs (0 to disable)                              |
| `resume`          | `False`  | resume training from last checkpoint                                                           |
| `amp`             | `True`   | Automatic Mixed Precision (AMP) training, choices=[True, False]                                |
| `fraction`        | `1.0`    | dataset fraction to train on (default is 1.0, all images in train set)                         |
| `profile`         | `False`  | profile ONNX and TensorRT speeds during training for loggers                                   |
| `freeze`          | `None`   | (int or list, optional) freeze first n layers, or freeze list of layer indices during training |
| `lr0`             | `0.01`   | initial learning rate (i.e. SGD=1E-2, Adam=1E-3)                                               |
| `lrf`             | `0.01`   | final learning rate (lr0 * lrf)                                                                |
| `momentum`        | `0.937`  | SGD momentum/Adam beta1                                                                        |
| `weight_decay`    | `0.0005` | optimizer weight decay 5e-4                                                                    |
| `warmup_epochs`   | `3.0`    | warmup epochs (fractions ok)                                                                   |
| `warmup_momentum` | `0.8`    | warmup initial momentum                                                                        |
| `warmup_bias_lr`  | `0.1`    | warmup initial bias lr                                                                         |
| `box`             | `7.5`    | box loss gain                                                                                  |
| `cls`             | `0.5`    | cls loss gain (scale with pixels)                                                              |
| `dfl`             | `1.5`    | dfl loss gain                                                                                  |
| `pose`            | `12.0`   | pose loss gain (pose-only)                                                                     |
| `kobj`            | `2.0`    | keypoint obj loss gain (pose-only)                                                             |
| `label_smoothing` | `0.0`    | label smoothing (fraction)                                                                     |
| `nbs`             | `64`     | nominal batch size                                                                             |
| `overlap_mask`    | `True`   | masks should overlap during training (segment train only)                                      |
| `mask_ratio`      | `4`      | mask downsample ratio (segment train only)                                                     |
| `dropout`         | `0.0`    | use dropout regularization (classify train only)                                               |
| `val`             | `True`   | validate/test during training                                                                  |

## Logging

In training a YOLOv8 model, you might find it valuable to keep track of the model's performance over time. This is where logging comes into play. Ultralytics' YOLO provides support for three types of loggers - Comet, ClearML, and TensorBoard.

To use a logger, select it from the dropdown menu in the code snippet above and run it. The chosen logger will be installed and initialized.

### Comet

[Comet](../integrations/comet.md) is a platform that allows data scientists and developers to track, compare, explain and optimize experiments and models. It provides functionalities such as real-time metrics, code diffs, and hyperparameters tracking.

To use Comet:

!!! example ""

    === "Python"
        ```python
        # pip install comet_ml
        import comet_ml

        comet_ml.init()
        ```

Remember to sign in to your Comet account on their website and get your API key. You will need to add this to your environment variables or your script to log your experiments.

### ClearML

[ClearML](https://www.clear.ml/) is an open-source platform that automates tracking of experiments and helps with efficient sharing of resources. It is designed to help teams manage, execute, and reproduce their ML work more efficiently.

To use ClearML:

!!! example ""

    === "Python"
        ```python
        # pip install clearml
        import clearml

        clearml.browser_login()
        ```

After running this script, you will need to sign in to your ClearML account on the browser and authenticate your session.

### TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard) is a visualization toolkit for TensorFlow. It allows you to visualize your TensorFlow graph, plot quantitative metrics about the execution of your graph, and show additional data like images that pass through it.

To use TensorBoard in [Google Colab](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb):

!!! example ""

    === "CLI"
        ```bash
        load_ext tensorboard
        tensorboard --logdir ultralytics/runs  # replace with 'runs' directory
        ```

To use TensorBoard locally run the below command and view results at http://localhost:6006/.

!!! example ""

    === "CLI"
        ```bash
        tensorboard --logdir ultralytics/runs  # replace with 'runs' directory
        ```

This will load TensorBoard and direct it to the directory where your training logs are saved.

After setting up your logger, you can then proceed with your model training. All training metrics will be automatically logged in your chosen platform, and you can access these logs to monitor your model's performance over time, compare different models, and identify areas for improvement.
