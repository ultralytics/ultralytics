---
comments: true
description: Master hyperparameter tuning for Ultralytics YOLO to optimize model performance with our comprehensive guide. Elevate your machine learning models today!.
keywords: Ultralytics YOLO, hyperparameter tuning, machine learning, model optimization, genetic algorithms, learning rate, batch size, epochs
---

# Ultralytics YOLO [Hyperparameter Tuning](https://www.ultralytics.com/glossary/hyperparameter-tuning) Guide

## Introduction

Hyperparameter tuning is not just a one-time set-up but an iterative process aimed at optimizing the [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) model's performance metrics, such as accuracy, precision, and recall. In the context of Ultralytics YOLO, these hyperparameters could range from learning rate to architectural details, such as the number of layers or types of activation functions used.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/j0MOGKBqx7E"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Tune Hyperparameters for Better Model Performance 🚀
</p>

### What are Hyperparameters?

Hyperparameters are high-level, structural settings for the algorithm. They are set prior to the training phase and remain constant during it. Here are some commonly tuned hyperparameters in Ultralytics YOLO:

- **Learning Rate** `lr0`: Determines the step size at each iteration while moving towards a minimum in the [loss function](https://www.ultralytics.com/glossary/loss-function).
- **[Batch Size](https://www.ultralytics.com/glossary/batch-size)** `batch`: Number of images processed simultaneously in a forward pass.
- **Number of [Epochs](https://www.ultralytics.com/glossary/epoch)** `epochs`: An epoch is one complete forward and backward pass of all the training examples.
- **Architecture Specifics**: Such as channel counts, number of layers, types of activation functions, etc.

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/hyperparameter-tuning-visual.avif" alt="Hyperparameter Tuning Visual">
</p>

For a full list of augmentation hyperparameters used in YOLO11 please refer to the [configurations page](../usage/cfg.md#augmentation-settings).

### Genetic Evolution and Mutation

Ultralytics YOLO uses genetic algorithms to optimize hyperparameters. Genetic algorithms are inspired by the mechanism of natural selection and genetics.

- **Mutation**: In the context of Ultralytics YOLO, mutation helps in locally searching the hyperparameter space by applying small, random changes to existing hyperparameters, producing new candidates for evaluation.
- **Crossover**: Although crossover is a popular genetic algorithm technique, it is not currently used in Ultralytics YOLO for hyperparameter tuning. The focus is mainly on mutation for generating new hyperparameter sets.

## Preparing for Hyperparameter Tuning

Before you begin the tuning process, it's important to:

1. **Identify the Metrics**: Determine the metrics you will use to evaluate the model's performance. This could be AP50, F1-score, or others.
2. **Set the Tuning Budget**: Define how much computational resources you're willing to allocate. Hyperparameter tuning can be computationally intensive.

## Steps Involved

### Initialize Hyperparameters

Start with a reasonable set of initial hyperparameters. This could either be the default hyperparameters set by Ultralytics YOLO or something based on your domain knowledge or previous experiments.

### Mutate Hyperparameters

Use the `_mutate` method to produce a new set of hyperparameters based on the existing set.

### Train Model

Training is performed using the mutated set of hyperparameters. The training performance is then assessed.

### Evaluate Model

Use metrics like AP50, F1-score, or custom metrics to evaluate the model's performance.

### Log Results

It's crucial to log both the performance metrics and the corresponding hyperparameters for future reference.

### Repeat

The process is repeated until either the set number of iterations is reached or the performance metric is satisfactory.

## Default Search Space Description

The following table lists the default search space parameters for hyperparameter tuning in YOLO11. Each parameter has a specific value range defined by a tuple `(min, max)`.

| Parameter         | Type    | Value Range    | Description                                                                                                      |
| ----------------- | ------- | -------------- | ---------------------------------------------------------------------------------------------------------------- |
| `lr0`             | `float` | `(1e-5, 1e-1)` | Initial learning rate at the start of training. Lower values provide more stable training but slower convergence |
| `lrf`             | `float` | `(0.01, 1.0)`  | Final learning rate factor as a fraction of lr0. Controls how much the learning rate decreases during training   |
| `momentum`        | `float` | `(0.6, 0.98)`  | SGD momentum factor. Higher values help maintain consistent gradient direction and can speed up convergence      |
| `weight_decay`    | `float` | `(0.0, 0.001)` | L2 regularization factor to prevent overfitting. Larger values enforce stronger regularization                   |
| `warmup_epochs`   | `float` | `(0.0, 5.0)`   | Number of epochs for linear learning rate warmup. Helps prevent early training instability                       |
| `warmup_momentum` | `float` | `(0.0, 0.95)`  | Initial momentum during warmup phase. Gradually increases to the final momentum value                            |
| `box`             | `float` | `(0.02, 0.2)`  | Bounding box loss weight in the total loss function. Balances box regression vs classification                   |
| `cls`             | `float` | `(0.2, 4.0)`   | Classification loss weight in the total loss function. Higher values emphasize correct class prediction          |
| `hsv_h`           | `float` | `(0.0, 0.1)`   | Random hue augmentation range in HSV color space. Helps model generalize across color variations                 |
| `hsv_s`           | `float` | `(0.0, 0.9)`   | Random saturation augmentation range in HSV space. Simulates different lighting conditions                       |
| `hsv_v`           | `float` | `(0.0, 0.9)`   | Random value (brightness) augmentation range. Helps model handle different exposure levels                       |
| `degrees`         | `float` | `(0.0, 45.0)`  | Maximum rotation augmentation in degrees. Helps model become invariant to object orientation                     |
| `translate`       | `float` | `(0.0, 0.9)`   | Maximum translation augmentation as fraction of image size. Improves robustness to object position               |
| `scale`           | `float` | `(0.0, 0.9)`   | Random scaling augmentation range. Helps model detect objects at different sizes                                 |
| `shear`           | `float` | `(0.0, 10.0)`  | Maximum shear augmentation in degrees. Adds perspective-like distortions to training images                      |
| `perspective`     | `float` | `(0.0, 0.001)` | Random perspective augmentation range. Simulates different viewing angles                                        |
| `flipud`          | `float` | `(0.0, 1.0)`   | Probability of vertical image flip during training. Useful for overhead/aerial imagery                           |
| `fliplr`          | `float` | `(0.0, 1.0)`   | Probability of horizontal image flip. Helps model become invariant to object direction                           |
| `mosaic`          | `float` | `(0.0, 1.0)`   | Probability of using mosaic augmentation, which combines 4 images. Especially useful for small object detection  |
| `mixup`           | `float` | `(0.0, 1.0)`   | Probability of using mixup augmentation, which blends two images. Can improve model robustness                   |
| `copy_paste`      | `float` | `(0.0, 1.0)`   | Probability of using copy-paste augmentation. Helps improve instance segmentation performance                    |

## Custom Search Space Example

Here's how to define a search space and use the `model.tune()` method to utilize the `Tuner` class for hyperparameter tuning of YOLO11n on COCO8 for 30 epochs with an AdamW optimizer and skipping plotting, checkpointing and validation other than on final epoch for faster Tuning.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Initialize the YOLO model
        model = YOLO("yolo11n.pt")

        # Define search space
        search_space = {
            "lr0": (1e-5, 1e-1),
            "degrees": (0.0, 45.0),
        }

        # Tune hyperparameters on COCO8 for 30 epochs
        model.tune(
            data="coco8.yaml",
            epochs=30,
            iterations=300,
            optimizer="AdamW",
            space=search_space,
            plots=False,
            save=False,
            val=False,
        )
        ```

## Results

After you've successfully completed the hyperparameter tuning process, you will obtain several files and directories that encapsulate the results of the tuning. The following describes each:

### File Structure

Here's what the directory structure of the results will look like. Training directories like `train1/` contain individual tuning iterations, i.e. one model trained with one set of hyperparameters. The `tune/` directory contains tuning results from all the individual model trainings:

```plaintext
runs/
└── detect/
    ├── train1/
    ├── train2/
    ├── ...
    └── tune/
        ├── best_hyperparameters.yaml
        ├── best_fitness.png
        ├── tune_results.csv
        ├── tune_scatter_plots.png
        └── weights/
            ├── last.pt
            └── best.pt
```

### File Descriptions

#### best_hyperparameters.yaml

This YAML file contains the best-performing hyperparameters found during the tuning process. You can use this file to initialize future trainings with these optimized settings.

- **Format**: YAML
- **Usage**: Hyperparameter results
- **Example**:

    ```yaml
    # 558/900 iterations complete ✅ (45536.81s)
    # Results saved to /usr/src/ultralytics/runs/detect/tune
    # Best fitness=0.64297 observed at iteration 498
    # Best fitness metrics are {'metrics/precision(B)': 0.87247, 'metrics/recall(B)': 0.71387, 'metrics/mAP50(B)': 0.79106, 'metrics/mAP50-95(B)': 0.62651, 'val/box_loss': 2.79884, 'val/cls_loss': 2.72386, 'val/dfl_loss': 0.68503, 'fitness': 0.64297}
    # Best fitness model is /usr/src/ultralytics/runs/detect/train498
    # Best fitness hyperparameters are printed below.

    lr0: 0.00269
    lrf: 0.00288
    momentum: 0.73375
    weight_decay: 0.00015
    warmup_epochs: 1.22935
    warmup_momentum: 0.1525
    box: 18.27875
    cls: 1.32899
    dfl: 0.56016
    hsv_h: 0.01148
    hsv_s: 0.53554
    hsv_v: 0.13636
    degrees: 0.0
    translate: 0.12431
    scale: 0.07643
    shear: 0.0
    perspective: 0.0
    flipud: 0.0
    fliplr: 0.08631
    mosaic: 0.42551
    mixup: 0.0
    copy_paste: 0.0
    ```

#### best_fitness.png

This is a plot displaying fitness (typically a performance metric like AP50) against the number of iterations. It helps you visualize how well the genetic algorithm performed over time.

- **Format**: PNG
- **Usage**: Performance visualization

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/best-fitness.avif" alt="Hyperparameter Tuning Fitness vs Iteration">
</p>

#### tune_results.csv

A CSV file containing detailed results of each iteration during the tuning. Each row in the file represents one iteration, and it includes metrics like fitness score, [precision](https://www.ultralytics.com/glossary/precision), [recall](https://www.ultralytics.com/glossary/recall), as well as the hyperparameters used.

- **Format**: CSV
- **Usage**: Per-iteration results tracking.
- **Example**:
    ```csv
      fitness,lr0,lrf,momentum,weight_decay,warmup_epochs,warmup_momentum,box,cls,dfl,hsv_h,hsv_s,hsv_v,degrees,translate,scale,shear,perspective,flipud,fliplr,mosaic,mixup,copy_paste
      0.05021,0.01,0.01,0.937,0.0005,3.0,0.8,7.5,0.5,1.5,0.015,0.7,0.4,0.0,0.1,0.5,0.0,0.0,0.0,0.5,1.0,0.0,0.0
      0.07217,0.01003,0.00967,0.93897,0.00049,2.79757,0.81075,7.5,0.50746,1.44826,0.01503,0.72948,0.40658,0.0,0.0987,0.4922,0.0,0.0,0.0,0.49729,1.0,0.0,0.0
      0.06584,0.01003,0.00855,0.91009,0.00073,3.42176,0.95,8.64301,0.54594,1.72261,0.01503,0.59179,0.40658,0.0,0.0987,0.46955,0.0,0.0,0.0,0.49729,0.80187,0.0,0.0
    ```

#### tune_scatter_plots.png

This file contains scatter plots generated from `tune_results.csv`, helping you visualize relationships between different hyperparameters and performance metrics. Note that hyperparameters initialized to 0 will not be tuned, such as `degrees` and `shear` below.

- **Format**: PNG
- **Usage**: Exploratory data analysis

<p align="center">
  <img width="1000" src="https://github.com/ultralytics/docs/releases/download/0/tune-scatter-plots.avif" alt="Hyperparameter Tuning Scatter Plots">
</p>

#### weights/

This directory contains the saved [PyTorch](https://www.ultralytics.com/glossary/pytorch) models for the last and the best iterations during the hyperparameter tuning process.

- **`last.pt`**: The last.pt are the weights from the last epoch of training.
- **`best.pt`**: The best.pt weights for the iteration that achieved the best fitness score.

Using these results, you can make more informed decisions for your future model trainings and analyses. Feel free to consult these artifacts to understand how well your model performed and how you might improve it further.

## Conclusion

The hyperparameter tuning process in Ultralytics YOLO is simplified yet powerful, thanks to its genetic algorithm-based approach focused on mutation. Following the steps outlined in this guide will assist you in systematically tuning your model to achieve better performance.

### Further Reading

1. [Hyperparameter Optimization in Wikipedia](https://en.wikipedia.org/wiki/Hyperparameter_optimization)
2. [YOLOv5 Hyperparameter Evolution Guide](../yolov5/tutorials/hyperparameter_evolution.md)
3. [Efficient Hyperparameter Tuning with Ray Tune and YOLO11](../integrations/ray-tune.md)

For deeper insights, you can explore the `Tuner` class source code and accompanying documentation. Should you have any questions, feature requests, or need further assistance, feel free to reach out to us on [GitHub](https://github.com/ultralytics/ultralytics/issues/new/choose) or [Discord](https://discord.com/invite/ultralytics).

## FAQ

### How do I optimize the [learning rate](https://www.ultralytics.com/glossary/learning-rate) for Ultralytics YOLO during hyperparameter tuning?

To optimize the learning rate for Ultralytics YOLO, start by setting an initial learning rate using the `lr0` parameter. Common values range from `0.001` to `0.01`. During the hyperparameter tuning process, this value will be mutated to find the optimal setting. You can utilize the `model.tune()` method to automate this process. For example:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Initialize the YOLO model
        model = YOLO("yolo11n.pt")

        # Tune hyperparameters on COCO8 for 30 epochs
        model.tune(data="coco8.yaml", epochs=30, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)
        ```

For more details, check the [Ultralytics YOLO configuration page](../usage/cfg.md#augmentation-settings).

### What are the benefits of using genetic algorithms for hyperparameter tuning in YOLO11?

Genetic algorithms in Ultralytics YOLO11 provide a robust method for exploring the hyperparameter space, leading to highly optimized model performance. Key benefits include:

- **Efficient Search**: Genetic algorithms like mutation can quickly explore a large set of hyperparameters.
- **Avoiding Local Minima**: By introducing randomness, they help in avoiding local minima, ensuring better global optimization.
- **Performance Metrics**: They adapt based on performance metrics such as AP50 and F1-score.

To see how genetic algorithms can optimize hyperparameters, check out the [hyperparameter evolution guide](../yolov5/tutorials/hyperparameter_evolution.md).

### How long does the hyperparameter tuning process take for Ultralytics YOLO?

The time required for hyperparameter tuning with Ultralytics YOLO largely depends on several factors such as the size of the dataset, the complexity of the model architecture, the number of iterations, and the computational resources available. For instance, tuning YOLO11n on a dataset like COCO8 for 30 epochs might take several hours to days, depending on the hardware.

To effectively manage tuning time, define a clear tuning budget beforehand ([internal section link](#preparing-for-hyperparameter-tuning)). This helps in balancing resource allocation and optimization goals.

### What metrics should I use to evaluate model performance during hyperparameter tuning in YOLO?

When evaluating model performance during hyperparameter tuning in YOLO, you can use several key metrics:

- **AP50**: The average precision at IoU threshold of 0.50.
- **F1-Score**: The harmonic mean of precision and recall.
- **Precision and Recall**: Individual metrics indicating the model's [accuracy](https://www.ultralytics.com/glossary/accuracy) in identifying true positives versus false positives and false negatives.

These metrics help you understand different aspects of your model's performance. Refer to the [Ultralytics YOLO performance metrics](../guides/yolo-performance-metrics.md) guide for a comprehensive overview.

### Can I use Ultralytics HUB for hyperparameter tuning of YOLO models?

Yes, you can use Ultralytics HUB for hyperparameter tuning of YOLO models. The HUB offers a no-code platform to easily upload datasets, train models, and perform hyperparameter tuning efficiently. It provides real-time tracking and visualization of tuning progress and results.

Explore more about using Ultralytics HUB for hyperparameter tuning in the [Ultralytics HUB Cloud Training](../hub/cloud-training.md) documentation.
