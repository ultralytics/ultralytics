---
comments: true
description: Learn to integrate hyperparameter tuning using Ray Tune with Ultralytics YOLOv8, and optimize your model's performance efficiently.
keywords: yolov8, ray tune, hyperparameter tuning, hyperparameter optimization, machine learning, computer vision, deep learning, image recognition
---

# Efficient Hyperparameter Tuning with Ray Tune and YOLOv8

Hyperparameter tuning is vital in achieving peak model performance by discovering the optimal set of hyperparameters. This involves running trials with different hyperparameters and evaluating each trial’s performance.

## Accelerate Tuning with Ultralytics YOLOv8 and Ray Tune

[Ultralytics YOLOv8](https://ultralytics.com) incorporates Ray Tune for hyperparameter tuning, streamlining the optimization of YOLOv8 model hyperparameters. With Ray Tune, you can utilize advanced search strategies, parallelism, and early stopping to expedite the tuning process.

### Ray Tune

![Ray Tune Overview](https://docs.ray.io/en/latest/_images/tune_overview.png)

[Ray Tune](https://docs.ray.io/en/latest/tune/index.html) is a hyperparameter tuning library designed for efficiency and flexibility. It supports various search strategies, parallelism, and early stopping strategies, and seamlessly integrates with popular machine learning frameworks, including Ultralytics YOLOv8.

### Integration with Weights & Biases

YOLOv8 also allows optional integration with [Weights & Biases](https://wandb.ai/site) for monitoring the tuning process.

## Installation

To install the required packages, run:

!!! tip "Installation"

    ```bash
    # Install and update Ultralytics and Ray Tune pacakges
    pip install -U ultralytics 'ray[tune]'

    # Optionally install W&B for logging
    pip install wandb
    ```

## Usage

!!! example "Usage"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    result_grid = model.tune(data="coco128.yaml")
    ```

## `tune()` Method Parameters

The `tune()` method in YOLOv8 provides an easy-to-use interface for hyperparameter tuning with Ray Tune. It accepts several arguments that allow you to customize the tuning process. Below is a detailed explanation of each parameter:

| Parameter       | Type           | Description                                                                                                                                                                                                                                                                                    | Default Value |
|-----------------|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `data`          | str            | The dataset configuration file (in YAML format) to run the tuner on. This file should specify the training and validation data paths, as well as other dataset-specific settings.                                                                                                              |               |
| `space`         | dict, optional | A dictionary defining the hyperparameter search space for Ray Tune. Each key corresponds to a hyperparameter name, and the value specifies the range of values to explore during tuning. If not provided, YOLOv8 uses a default search space with various hyperparameters.                     |               |
| `grace_period`  | int, optional  | The grace period in epochs for the [ASHA scheduler](https://docs.ray.io/en/latest/tune/api/schedulers.html) in Ray Tune. The scheduler will not terminate any trial before this number of epochs, allowing the model to have some minimum training before making a decision on early stopping. | 10            |
| `gpu_per_trial` | int, optional  | The number of GPUs to allocate per trial during tuning. This helps manage GPU usage, particularly in multi-GPU environments. If not provided, the tuner will use all available GPUs.                                                                                                           | None          |
| `max_samples`   | int, optional  | The maximum number of trials to run during tuning. This parameter helps control the total number of hyperparameter combinations tested, ensuring the tuning process does not run indefinitely.                                                                                                 | 10            |
| `**train_args`  | dict, optional | Additional arguments to pass to the `train()` method during tuning. These arguments can include settings like the number of training epochs, batch size, and other training-specific configurations.                                                                                           | {}            |

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

    # Define a YOLO model    
    model = YOLO("yolov8n.pt")

    # Run Ray Tune on the model
    result_grid = model.tune(data="coco128.yaml",
                             space={"lr0": tune.uniform(1e-5, 1e-1)},
                             epochs=50)
    ```

In the code snippet above, we create a YOLO model with the "yolov8n.pt" pretrained weights. Then, we call the `tune()` method, specifying the dataset configuration with "coco128.yaml". We provide a custom search space for the initial learning rate `lr0` using a dictionary with the key "lr0" and the value `tune.uniform(1e-5, 1e-1)`. Finally, we pass additional training arguments, such as the number of epochs directly to the tune method as `epochs=50`.

# Processing Ray Tune Results

After running a hyperparameter tuning experiment with Ray Tune, you might want to perform various analyses on the obtained results. This guide will take you through common workflows for processing and analyzing these results.

## Loading Tune Experiment Results from a Directory

After running the tuning experiment with `tuner.fit()`, you can load the results from a directory. This is useful, especially if you're performing the analysis after the initial training script has exited.

```python
experiment_path = f"{storage_path}/{exp_name}"
print(f"Loading results from {experiment_path}...")

restored_tuner = tune.Tuner.restore(experiment_path, trainable=train_mnist)
result_grid = restored_tuner.get_results()
```

## Basic Experiment-Level Analysis

Get an overview of how trials performed. You can quickly check if there were any errors during the trials.

```python
if result_grid.errors:
    print("One or more trials failed!")
else:
    print("No errors!")
```

## Basic Trial-Level Analysis

Access individual trial hyperparameter configurations and the last reported metrics.

```python
for i, result in enumerate(result_grid):
    print(f"Trial #{i}: Configuration: {result.config}, Last Reported Metrics: {result.metrics}")
```

## Plotting the Entire History of Reported Metrics for a Trial

You can plot the history of reported metrics for each trial to see how the metrics evolved over time.

```python
import matplotlib.pyplot as plt

for result in result_grid:
    plt.plot(result.metrics_dataframe["training_iteration"], result.metrics_dataframe["mean_accuracy"], label=f"Trial {i}")

plt.xlabel('Training Iterations')
plt.ylabel('Mean Accuracy')
plt.legend()
plt.show()
```

## Summary

In this documentation, we covered common workflows to analyze the results of experiments run with Ray Tune using Ultralytics. The key steps include loading the experiment results from a directory, performing basic experiment-level and trial-level analysis and plotting metrics.

Explore further by looking into Ray Tune’s [Analyze Results](https://docs.ray.io/en/latest/tune/examples/tune_analyze_results.html) docs page to get the most out of your hyperparameter tuning experiments.
