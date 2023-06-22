---
comments: true
description: Discover how to integrate hyperparameter tuning with Ray Tune and Ultralytics YOLOv8. Speed up the tuning process and optimize your model's performance.
keywords: yolov8, ray tune, hyperparameter tuning, hyperparameter optimization, machine learning, computer vision, deep learning, image recognition
---

# Hyperparameter Tuning with Ray Tune and YOLOv8

Hyperparameter tuning (or hyperparameter optimization) is the process of determining the right combination of hyperparameters that maximizes model performance. It works by running multiple trials in a single training process, evaluating the performance of each trial, and selecting the best hyperparameter values based on the evaluation results.

## Ultralytics YOLOv8 and Ray Tune Integration

[Ultralytics](https://ultralytics.com) YOLOv8 integrates hyperparameter tuning with Ray Tune, allowing you to easily optimize your YOLOv8 model's hyperparameters. By using Ray Tune, you can leverage advanced search algorithms, parallelism, and early stopping to speed up the tuning process and achieve better model performance.

### Ray Tune

<div align="center">
<a href="https://docs.ray.io/en/latest/tune/index.html" target="_blank">
<img width="480" src="https://docs.ray.io/en/latest/_images/tune_overview.png"></a>
</div>

[Ray Tune](https://docs.ray.io/en/latest/tune/index.html) is a powerful and flexible hyperparameter tuning library for machine learning models. It provides an efficient way to optimize hyperparameters by supporting various search algorithms, parallelism, and early stopping strategies. Ray Tune's flexible architecture enables seamless integration with popular machine learning frameworks, including Ultralytics YOLOv8.

### Weights & Biases

YOLOv8 also supports optional integration with [Weights & Biases](https://wandb.ai/site) (wandb) for tracking the tuning progress.

## Installation

To install the required packages, run:

!!! tip "Installation"

    ```bash
    pip install -U ultralytics "ray[tune]"  # install and/or update
    pip install wandb  # optional
    ```

## Usage

!!! example "Usage"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    results = model.tune(data="coco128.yaml")
    ```

## `tune()` Method Parameters

The `tune()` method in YOLOv8 provides an easy-to-use interface for hyperparameter tuning with Ray Tune. It accepts several arguments that allow you to customize the tuning process. Below is a detailed explanation of each parameter:

| Parameter       | Type           | Description                                                                                                                                                                                                                                                                                   | Default Value |
|-----------------|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `data`          | str            | The dataset configuration file (in YAML format) to run the tuner on. This file should specify the training and validation data paths, as well as other dataset-specific settings.                                                                                                             |               |
| `space`         | dict, optional | A dictionary defining the hyperparameter search space for Ray Tune. Each key corresponds to a hyperparameter name, and the value specifies the range of values to explore during tuning. If not provided, YOLOv8 uses a default search space with various hyperparameters.                    |               |
| `grace_period`  | int, optional  | The grace period in epochs for the [ASHA scheduler]https://docs.ray.io/en/latest/tune/api/schedulers.html) in Ray Tune. The scheduler will not terminate any trial before this number of epochs, allowing the model to have some minimum training before making a decision on early stopping. | 10            |
| `gpu_per_trial` | int, optional  | The number of GPUs to allocate per trial during tuning. This helps manage GPU usage, particularly in multi-GPU environments. If not provided, the tuner will use all available GPUs.                                                                                                          | None          |
| `max_samples`   | int, optional  | The maximum number of trials to run during tuning. This parameter helps control the total number of hyperparameter combinations tested, ensuring the tuning process does not run indefinitely.                                                                                                | 10            |
| `train_args`    | dict, optional | A dictionary of additional arguments to pass to the `train()` method during tuning. These arguments can include settings like the number of training epochs, batch size, and other training-specific configurations.                                                                          | {}            |

By customizing these parameters, you can fine-tune the hyperparameter optimization process to suit your specific needs and available computational resources.

## Default Search Space Description

The following table lists the default search space parameters for hyperparameter tuning in YOLOv8 with Ray Tune. Each parameter has a specific value range defined by `tune.uniform()`.

| Parameter       | Value Range                | Description                              |
|-----------------|----------------------------|------------------------------------------|
| lr0             | `tune.uniform(1e-5, 1e-1)` | Initial learning rate                    |
| lrf             | `tune.uniform(0.01, 1.0)`  | Final learning rate factor               |
| momentum        | `tune.uniform(0.6, 0.98)`  | Momentum                                 |
| weight_decay    | `tune.uniform(0.0, 0.001)` | Weight decay                             |
| warmup_epochs   | `tune.uniform(0.0, 5.0)`   | Warmup epochs                            |
| warmup_momentum | `tune.uniform(0.0, 0.95)`  | Warmup momentum                          |
| box             | `tune.uniform(0.02, 0.2)`  | Box loss weight                          |
| cls             | `tune.uniform(0.2, 4.0)`   | Class loss weight                        |
| hsv_h           | `tune.uniform(0.0, 0.1)`   | Hue augmentation range                   |
| hsv_s           | `tune.uniform(0.0, 0.9)`   | Saturation augmentation range            |
| hsv_v           | `tune.uniform(0.0, 0.9)`   | Value (brightness) augmentation range    |
| degrees         | `tune.uniform(0.0, 45.0)`  | Rotation augmentation range (degrees)    |
| translate       | `tune.uniform(0.0, 0.9)`   | Translation augmentation range           |
| scale           | `tune.uniform(0.0, 0.9)`   | Scaling augmentation range               |
| shear           | `tune.uniform(0.0, 10.0)`  | Shear augmentation range (degrees)       |
| perspective     | `tune.uniform(0.0, 0.001)` | Perspective augmentation range           |
| flipud          | `tune.uniform(0.0, 1.0)`   | Vertical flip augmentation probability   |
| fliplr          | `tune.uniform(0.0, 1.0)`   | Horizontal flip augmentation probability |
| mosaic          | `tune.uniform(0.0, 1.0)`   | Mosaic augmentation probability          |
| mixup           | `tune.uniform(0.0, 1.0)`   | Mixup augmentation probability           |
| copy_paste      | `tune.uniform(0.0, 1.0)`   | Copy-paste augmentation probability      |

## Custom Search Space Example

In this example, we demonstrate how to use a custom search space for hyperparameter tuning with Ray Tune and YOLOv8. By providing a custom search space, you can focus the tuning process on specific hyperparameters of interest.

!!! example "Usage"

    ```python
    from ultralytics import YOLO
    from ray import tune
    
    model = YOLO("yolov8n.pt")
    result = model.tune(
        data="coco128.yaml",
        space={"lr0": tune.uniform(1e-5, 1e-1)},
        train_args={"epochs": 50}
    )
    ```

In the code snippet above, we create a YOLO model with the "yolov8n.pt" pretrained weights. Then, we call the `tune()` method, specifying the dataset configuration with "coco128.yaml". We provide a custom search space for the initial learning rate `lr0` using a dictionary with the key "lr0" and the value `tune.uniform(1e-5, 1e-1)`. Finally, we pass additional training arguments, such as the number of epochs, using the `train_args` parameter.