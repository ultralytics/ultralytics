---
comments: true
description: Optimize YOLO11 model performance with Ray Tune. Learn efficient hyperparameter tuning using advanced search strategies, parallelism, and early stopping.
keywords: YOLO11, Ray Tune, hyperparameter tuning, model optimization, machine learning, deep learning, AI, Ultralytics, Weights & Biases
---

# Efficient [Hyperparameter Tuning](https://www.ultralytics.com/glossary/hyperparameter-tuning) with Ray Tune and YOLO11

Hyperparameter tuning is vital in achieving peak model performance by discovering the optimal set of hyperparameters. This involves running trials with different hyperparameters and evaluating each trial's performance.

## Accelerate Tuning with Ultralytics YOLO11 and Ray Tune

[Ultralytics YOLO11](https://www.ultralytics.com/) incorporates Ray Tune for hyperparameter tuning, streamlining the optimization of YOLO11 model hyperparameters. With Ray Tune, you can utilize advanced search strategies, parallelism, and early stopping to expedite the tuning process.

### Ray Tune

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/ray-tune-overview.avif" alt="Ray Tune Overview">
</p>

[Ray Tune](https://docs.ray.io/en/latest/tune/index.html) is a hyperparameter tuning library designed for efficiency and flexibility. It supports various search strategies, parallelism, and early stopping strategies, and seamlessly integrates with popular [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) frameworks, including Ultralytics YOLO11.

### Integration with Weights & Biases

YOLO11 also allows optional integration with [Weights & Biases](https://wandb.ai/site) for monitoring the tuning process.

## Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install and update Ultralytics and Ray Tune packages
        pip install -U ultralytics "ray[tune]"

        # Optionally install W&B for logging
        pip install wandb
        ```

## Usage

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11n model
        model = YOLO("yolo11n.pt")

        # Start tuning hyperparameters for YOLO11n training on the COCO8 dataset
        result_grid = model.tune(data="coco8.yaml", use_ray=True)
        ```

## `tune()` Method Parameters

The `tune()` method in YOLO11 provides an easy-to-use interface for hyperparameter tuning with Ray Tune. It accepts several arguments that allow you to customize the tuning process. Below is a detailed explanation of each parameter:

| Parameter       | Type             | Description                                                                                                                                                                                                                                                                                                                                  | Default Value |
| --------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `data`          | `str`            | The dataset configuration file (in YAML format) to run the tuner on. This file should specify the training and [validation data](https://www.ultralytics.com/glossary/validation-data) paths, as well as other dataset-specific settings.                                                                                                    |               |
| `space`         | `dict, optional` | A dictionary defining the hyperparameter search space for Ray Tune. Each key corresponds to a hyperparameter name, and the value specifies the range of values to explore during tuning. If not provided, YOLO11 uses a default search space with various hyperparameters.                                                                   |               |
| `grace_period`  | `int, optional`  | The grace period in [epochs](https://www.ultralytics.com/glossary/epoch) for the [ASHA scheduler](https://docs.ray.io/en/latest/tune/api/schedulers.html) in Ray Tune. The scheduler will not terminate any trial before this number of epochs, allowing the model to have some minimum training before making a decision on early stopping. | 10            |
| `gpu_per_trial` | `int, optional`  | The number of GPUs to allocate per trial during tuning. This helps manage GPU usage, particularly in multi-GPU environments. If not provided, the tuner will use all available GPUs.                                                                                                                                                         | None          |
| `iterations`    | `int, optional`  | The maximum number of trials to run during tuning. This parameter helps control the total number of hyperparameter combinations tested, ensuring the tuning process does not run indefinitely.                                                                                                                                               | 10            |
| `**train_args`  | `dict, optional` | Additional arguments to pass to the `train()` method during tuning. These arguments can include settings like the number of training epochs, [batch size](https://www.ultralytics.com/glossary/batch-size), and other training-specific configurations.                                                                                      | {}            |

By customizing these parameters, you can fine-tune the hyperparameter optimization process to suit your specific needs and available computational resources.

## Default Search Space Description

The following table lists the default search space parameters for hyperparameter tuning in YOLO11 with Ray Tune. Each parameter has a specific value range defined by `tune.uniform()`.

| Parameter         | Value Range                | Description                                                                 |
| ----------------- | -------------------------- | --------------------------------------------------------------------------- |
| `lr0`             | `tune.uniform(1e-5, 1e-1)` | Initial [learning rate](https://www.ultralytics.com/glossary/learning-rate) |
| `lrf`             | `tune.uniform(0.01, 1.0)`  | Final learning rate factor                                                  |
| `momentum`        | `tune.uniform(0.6, 0.98)`  | Momentum                                                                    |
| `weight_decay`    | `tune.uniform(0.0, 0.001)` | Weight decay                                                                |
| `warmup_epochs`   | `tune.uniform(0.0, 5.0)`   | Warmup epochs                                                               |
| `warmup_momentum` | `tune.uniform(0.0, 0.95)`  | Warmup momentum                                                             |
| `box`             | `tune.uniform(0.02, 0.2)`  | Box loss weight                                                             |
| `cls`             | `tune.uniform(0.2, 4.0)`   | Class loss weight                                                           |
| `hsv_h`           | `tune.uniform(0.0, 0.1)`   | Hue augmentation range                                                      |
| `hsv_s`           | `tune.uniform(0.0, 0.9)`   | Saturation augmentation range                                               |
| `hsv_v`           | `tune.uniform(0.0, 0.9)`   | Value (brightness) augmentation range                                       |
| `degrees`         | `tune.uniform(0.0, 45.0)`  | Rotation augmentation range (degrees)                                       |
| `translate`       | `tune.uniform(0.0, 0.9)`   | Translation augmentation range                                              |
| `scale`           | `tune.uniform(0.0, 0.9)`   | Scaling augmentation range                                                  |
| `shear`           | `tune.uniform(0.0, 10.0)`  | Shear augmentation range (degrees)                                          |
| `perspective`     | `tune.uniform(0.0, 0.001)` | Perspective augmentation range                                              |
| `flipud`          | `tune.uniform(0.0, 1.0)`   | Vertical flip augmentation probability                                      |
| `fliplr`          | `tune.uniform(0.0, 1.0)`   | Horizontal flip augmentation probability                                    |
| `mosaic`          | `tune.uniform(0.0, 1.0)`   | Mosaic augmentation probability                                             |
| `mixup`           | `tune.uniform(0.0, 1.0)`   | Mixup augmentation probability                                              |
| `copy_paste`      | `tune.uniform(0.0, 1.0)`   | Copy-paste augmentation probability                                         |

## Custom Search Space Example

In this example, we demonstrate how to use a custom search space for hyperparameter tuning with Ray Tune and YOLO11. By providing a custom search space, you can focus the tuning process on specific hyperparameters of interest.

!!! example "Usage"

    ```python
    from ray import tune

    from ultralytics import YOLO

    # Define a YOLO model
    model = YOLO("yolo11n.pt")

    # Run Ray Tune on the model
    result_grid = model.tune(
        data="coco8.yaml",
        space={"lr0": tune.uniform(1e-5, 1e-1)},
        epochs=50,
        use_ray=True,
    )
    ```

In the code snippet above, we create a YOLO model with the "yolo11n.pt" pretrained weights. Then, we call the `tune()` method, specifying the dataset configuration with "coco8.yaml". We provide a custom search space for the initial learning rate `lr0` using a dictionary with the key "lr0" and the value `tune.uniform(1e-5, 1e-1)`. Finally, we pass additional training arguments, such as the number of epochs directly to the tune method as `epochs=50`.

## Processing Ray Tune Results

After running a hyperparameter tuning experiment with Ray Tune, you might want to perform various analyses on the obtained results. This guide will take you through common workflows for processing and analyzing these results.

### Loading Tune Experiment Results from a Directory

After running the tuning experiment with `tuner.fit()`, you can load the results from a directory. This is useful, especially if you're performing the analysis after the initial training script has exited.

```python
experiment_path = f"{storage_path}/{exp_name}"
print(f"Loading results from {experiment_path}...")

restored_tuner = tune.Tuner.restore(experiment_path, trainable=train_mnist)
result_grid = restored_tuner.get_results()
```

### Basic Experiment-Level Analysis

Get an overview of how trials performed. You can quickly check if there were any errors during the trials.

```python
if result_grid.errors:
    print("One or more trials failed!")
else:
    print("No errors!")
```

### Basic Trial-Level Analysis

Access individual trial hyperparameter configurations and the last reported metrics.

```python
for i, result in enumerate(result_grid):
    print(f"Trial #{i}: Configuration: {result.config}, Last Reported Metrics: {result.metrics}")
```

### Plotting the Entire History of Reported Metrics for a Trial

You can plot the history of reported metrics for each trial to see how the metrics evolved over time.

```python
import matplotlib.pyplot as plt

for i, result in enumerate(result_grid):
    plt.plot(
        result.metrics_dataframe["training_iteration"],
        result.metrics_dataframe["mean_accuracy"],
        label=f"Trial {i}",
    )

plt.xlabel("Training Iterations")
plt.ylabel("Mean Accuracy")
plt.legend()
plt.show()
```

## Summary

In this documentation, we covered common workflows to analyze the results of experiments run with Ray Tune using Ultralytics. The key steps include loading the experiment results from a directory, performing basic experiment-level and trial-level analysis and plotting metrics.

Explore further by looking into Ray Tune's [Analyze Results](https://docs.ray.io/en/latest/tune/examples/tune_analyze_results.html) docs page to get the most out of your hyperparameter tuning experiments.

## FAQ

### How do I tune the hyperparameters of my YOLO11 model using Ray Tune?

To tune the hyperparameters of your Ultralytics YOLO11 model using Ray Tune, follow these steps:

1. **Install the required packages:**

    ```bash
    pip install -U ultralytics "ray[tune]"
    pip install wandb  # optional for logging
    ```

2. **Load your YOLO11 model and start tuning:**

    ```python
    from ultralytics import YOLO

    # Load a YOLO11 model
    model = YOLO("yolo11n.pt")

    # Start tuning with the COCO8 dataset
    result_grid = model.tune(data="coco8.yaml", use_ray=True)
    ```

This utilizes Ray Tune's advanced search strategies and parallelism to efficiently optimize your model's hyperparameters. For more information, check out the [Ray Tune documentation](https://docs.ray.io/en/latest/tune/index.html).

### What are the default hyperparameters for YOLO11 tuning with Ray Tune?

Ultralytics YOLO11 uses the following default hyperparameters for tuning with Ray Tune:

| Parameter       | Value Range                | Description                    |
| --------------- | -------------------------- | ------------------------------ |
| `lr0`           | `tune.uniform(1e-5, 1e-1)` | Initial learning rate          |
| `lrf`           | `tune.uniform(0.01, 1.0)`  | Final learning rate factor     |
| `momentum`      | `tune.uniform(0.6, 0.98)`  | Momentum                       |
| `weight_decay`  | `tune.uniform(0.0, 0.001)` | Weight decay                   |
| `warmup_epochs` | `tune.uniform(0.0, 5.0)`   | Warmup epochs                  |
| `box`           | `tune.uniform(0.02, 0.2)`  | Box loss weight                |
| `cls`           | `tune.uniform(0.2, 4.0)`   | Class loss weight              |
| `hsv_h`         | `tune.uniform(0.0, 0.1)`   | Hue augmentation range         |
| `translate`     | `tune.uniform(0.0, 0.9)`   | Translation augmentation range |

These hyperparameters can be customized to suit your specific needs. For a complete list and more details, refer to the [Hyperparameter Tuning](../guides/hyperparameter-tuning.md) guide.

### How can I integrate Weights & Biases with my YOLO11 model tuning?

To integrate Weights & Biases (W&B) with your Ultralytics YOLO11 tuning process:

1. **Install W&B:**

    ```bash
    pip install wandb
    ```

2. **Modify your tuning script:**

    ```python
    import wandb

    from ultralytics import YOLO

    wandb.init(project="YOLO-Tuning", entity="your-entity")

    # Load YOLO model
    model = YOLO("yolo11n.pt")

    # Tune hyperparameters
    result_grid = model.tune(data="coco8.yaml", use_ray=True)
    ```

This setup will allow you to monitor the tuning process, track hyperparameter configurations, and visualize results in W&B.

### Why should I use Ray Tune for hyperparameter optimization with YOLO11?

Ray Tune offers numerous advantages for hyperparameter optimization:

- **Advanced Search Strategies:** Utilizes algorithms like Bayesian Optimization and HyperOpt for efficient parameter search.
- **Parallelism:** Supports parallel execution of multiple trials, significantly speeding up the tuning process.
- **Early Stopping:** Employs strategies like ASHA to terminate under-performing trials early, saving computational resources.

Ray Tune seamlessly integrates with Ultralytics YOLO11, providing an easy-to-use interface for tuning hyperparameters effectively. To get started, check out the [Efficient Hyperparameter Tuning with Ray Tune and YOLO11](../guides/hyperparameter-tuning.md) guide.

### How can I define a custom search space for YOLO11 hyperparameter tuning?

To define a custom search space for your YOLO11 hyperparameter tuning with Ray Tune:

```python
from ray import tune

from ultralytics import YOLO

model = YOLO("yolo11n.pt")
search_space = {"lr0": tune.uniform(1e-5, 1e-1), "momentum": tune.uniform(0.6, 0.98)}
result_grid = model.tune(data="coco8.yaml", space=search_space, use_ray=True)
```

This customizes the range of hyperparameters like initial learning rate and momentum to be explored during the tuning process. For advanced configurations, refer to the [Custom Search Space Example](#custom-search-space-example) section.
