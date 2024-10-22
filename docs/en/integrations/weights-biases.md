---
comments: true
description: Learn how to enhance YOLO11 experiment tracking and visualization with Weights & Biases for better model performance and management.
keywords: YOLO11, Weights & Biases, model training, experiment tracking, Ultralytics, machine learning, computer vision, model visualization
---

# Enhancing YOLO11 Experiment Tracking and Visualization with Weights & Biases

[Object detection](https://www.ultralytics.com/glossary/object-detection) models like [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) have become integral to many [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) applications. However, training, evaluating, and deploying these complex models introduce several challenges. Tracking key training metrics, comparing model variants, analyzing model behavior, and detecting issues require significant instrumentation and experiment management.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/EeDd5P4eS6A"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to use Ultralytics YOLO11 with Weights and Biases
</p>

This guide showcases Ultralytics YOLO11 integration with Weights & Biases for enhanced experiment tracking, model-checkpointing, and visualization of model performance. It also includes instructions for setting up the integration, training, fine-tuning, and visualizing results using Weights & Biases' interactive features.

## Weights & Biases

<p align="center">
  <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/wandb-demo-experiments.avif" alt="Weights & Biases Overview">
</p>

[Weights & Biases](https://wandb.ai/site) is a cutting-edge MLOps platform designed for tracking, visualizing, and managing [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) experiments. It features automatic logging of training metrics for full experiment reproducibility, an interactive UI for streamlined data analysis, and efficient model management tools for deploying across various environments.

## YOLO11 Training with Weights & Biases

You can use Weights & Biases to bring efficiency and automation to your YOLO11 training process.

## Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required packages for Ultralytics YOLO and Weights & Biases
        pip install -U ultralytics wandb
        ```

For detailed instructions and best practices related to the installation process, be sure to check our [YOLO11 Installation guide](../quickstart.md). While installing the required packages for YOLO11, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

## Configuring Weights & Biases

After installing the necessary packages, the next step is to set up your Weights & Biases environment. This includes creating a Weights & Biases account and obtaining the necessary API key for a smooth connection between your development environment and the W&B platform.

Start by initializing the Weights & Biases environment in your workspace. You can do this by running the following command and following the prompted instructions.

!!! tip "Initial SDK Setup"

    === "Python"

        ```python
        import wandb

        # Initialize your Weights & Biases environment
        wandb.login(key="<API_KEY>")
        ```

    === "CLI"

        ```bash
        # Initialize your Weights & Biases environment
        wandb login <API_KEY>
        ```

Navigate to the Weights & Biases authorization page to create and retrieve your API key. Use this key to authenticate your environment with W&B.

## Usage: Training YOLO11 with Weights & Biases

Before diving into the usage instructions for YOLO11 model training with Weights & Biases, be sure to check out the range of [YOLO11 models offered by Ultralytics](../models/index.md). This will help you choose the most appropriate model for your project requirements.

!!! example "Usage: Training YOLO11 with Weights & Biases"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO model
        model = YOLO("yolo11n.pt")

        # Train and Fine-Tune the Model
        model.train(data="coco8.yaml", epochs=5, project="ultralytics", name="yolo11n")
        ```

    === "CLI"

        ```bash
        # Train a YOLO11 model with Weights & Biases
        yolo train data=coco8.yaml epochs=5 project=ultralytics name=yolo11n
        ```

### W&B Arguments

| Argument | Default | Description                                                                                                        |
| -------- | ------- | ------------------------------------------------------------------------------------------------------------------ |
| project  | `None`  | Specifies the name of the project logged locally and in W&B. This way you can group multiple runs together.        |
| name     | `None`  | The name of the training run. This determines the name used to create subfolders and the name used for W&B logging |

!!! tip "Enable or Disable Weights & Biases"

    If you want to enable or disable Weights & Biases logging, you can use the `wandb` command. By default, Weights & Biases logging is enabled.

    === "CLI"

        ```bash
        # Enable Weights & Biases logging
        wandb enabled

        # Disable Weights & Biases logging
        wandb disabled
        ```

### Understanding the Output

Upon running the usage code snippet above, you can expect the following key outputs:

- The setup of a new run with its unique ID, indicating the start of the training process.
- A concise summary of the model's structure, including the number of layers and parameters.
- Regular updates on important metrics such as box loss, cls loss, dfl loss, [precision](https://www.ultralytics.com/glossary/precision), [recall](https://www.ultralytics.com/glossary/recall), and mAP scores during each training [epoch](https://www.ultralytics.com/glossary/epoch).
- At the end of training, detailed metrics including the model's inference speed, and overall [accuracy](https://www.ultralytics.com/glossary/accuracy) metrics are displayed.
- Links to the Weights & Biases dashboard for in-depth analysis and visualization of the training process, along with information on local log file locations.

### Viewing the Weights & Biases Dashboard

After running the usage code snippet, you can access the Weights & Biases (W&B) dashboard through the provided link in the output. This dashboard offers a comprehensive view of your model's training process with YOLO11.

## Key Features of the Weights & Biases Dashboard

- **Real-Time Metrics Tracking**: Observe metrics like loss, accuracy, and validation scores as they evolve during the training, offering immediate insights for model tuning. [See how experiments are tracked using Weights & Biases](https://imgur.com/D6NVnmN).

- **Hyperparameter Optimization**: Weights & Biases aids in fine-tuning critical parameters such as [learning rate](https://www.ultralytics.com/glossary/learning-rate), [batch size](https://www.ultralytics.com/glossary/batch-size), and more, enhancing the performance of YOLO11.

- **Comparative Analysis**: The platform allows side-by-side comparisons of different training runs, essential for assessing the impact of various model configurations.

- **Visualization of Training Progress**: Graphical representations of key metrics provide an intuitive understanding of the model's performance across epochs. [See how Weights & Biases helps you visualize validation results](https://imgur.com/a/kU5h7W4).

- **Resource Monitoring**: Keep track of CPU, GPU, and memory usage to optimize the efficiency of the training process.

- **Model Artifacts Management**: Access and share model checkpoints, facilitating easy deployment and collaboration.

- **Viewing Inference Results with Image Overlay**: Visualize the prediction results on images using interactive overlays in Weights & Biases, providing a clear and detailed view of model performance on real-world data. For more detailed information on Weights & Biases' image overlay capabilities, check out this [link](https://docs.wandb.ai/guides/track/log/media/#image-overlays). [See how Weights & Biases' image overlays helps visualize model inferences](https://imgur.com/a/UTSiufs).

By using these features, you can effectively track, analyze, and optimize your YOLO11 model's training, ensuring the best possible performance and efficiency.

## Summary

This guide helped you explore the Ultralytics YOLO integration with Weights & Biases. It illustrates the ability of this integration to efficiently track and visualize model training and prediction results.

For further details on usage, visit [Weights & Biases' official documentation](https://docs.wandb.ai/guides/integrations/ultralytics/).

Also, be sure to check out the [Ultralytics integration guide page](../integrations/index.md), to learn more about different exciting integrations.

## FAQ

### How do I integrate Weights & Biases with Ultralytics YOLO11?

To integrate Weights & Biases with Ultralytics YOLO11:

1. Install the required packages:

```bash
pip install -U ultralytics wandb
```

2. Log in to your Weights & Biases account:

```python
import wandb

wandb.login(key="<API_KEY>")
```

3. Train your YOLO11 model with W&B logging enabled:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="coco8.yaml", epochs=5, project="ultralytics", name="yolo11n")
```

This will automatically log metrics, hyperparameters, and model artifacts to your W&B project.

### What are the key features of Weights & Biases integration with YOLO11?

The key features include:

- Real-time metrics tracking during training
- Hyperparameter optimization tools
- Comparative analysis of different training runs
- Visualization of training progress through graphs
- Resource monitoring (CPU, GPU, memory usage)
- Model artifacts management and sharing
- Viewing inference results with image overlays

These features help in tracking experiments, optimizing models, and collaborating more effectively on YOLO11 projects.

### How can I view the Weights & Biases dashboard for my YOLO11 training?

After running your training script with W&B integration:

1. A link to your W&B dashboard will be provided in the console output.
2. Click on the link or go to [wandb.ai](https://wandb.ai/) and log in to your account.
3. Navigate to your project to view detailed metrics, visualizations, and model performance data.

The dashboard offers insights into your model's training process, allowing you to analyze and improve your YOLO11 models effectively.

### Can I disable Weights & Biases logging for YOLO11 training?

Yes, you can disable W&B logging using the following command:

```bash
wandb disabled
```

To re-enable logging, use:

```bash
wandb enabled
```

This allows you to control when you want to use W&B logging without modifying your training scripts.

### How does Weights & Biases help in optimizing YOLO11 models?

Weights & Biases helps optimize YOLO11 models by:

1. Providing detailed visualizations of training metrics
2. Enabling easy comparison between different model versions
3. Offering tools for [hyperparameter tuning](https://www.ultralytics.com/glossary/hyperparameter-tuning)
4. Allowing for collaborative analysis of model performance
5. Facilitating easy sharing of model artifacts and results

These features help researchers and developers iterate faster and make data-driven decisions to improve their YOLO11 models.
