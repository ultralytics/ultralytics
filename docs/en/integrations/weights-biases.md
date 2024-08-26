---
comments: true
description: Learn how to enhance YOLOv8 experiment tracking and visualization with Weights & Biases for better model performance and management.
keywords: YOLOv8, Weights & Biases, model training, experiment tracking, Ultralytics, machine learning, computer vision, model visualization
---

# Enhancing YOLOv8 Experiment Tracking and Visualization with Weights & Biases

Object detection models like [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) have become integral to many computer vision applications. However, training, evaluating, and deploying these complex models introduces several challenges. Tracking key training metrics, comparing model variants, analyzing model behavior, and detecting issues require substantial instrumentation and experiment management.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/EeDd5P4eS6A"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to use Ultralytics YOLOv8 with Weights and Biases
</p>

This guide showcases Ultralytics YOLOv8 integration with Weights & Biases' for enhanced experiment tracking, model-checkpointing, and visualization of model performance. It also includes instructions for setting up the integration, training, fine-tuning, and visualizing results using Weights & Biases' interactive features.

## Weights & Biases

<p align="center">
  <img width="800" src="https://docs.wandb.ai/assets/images/wandb_demo_experiments-4797af7fe7236d6c5c42adbdc93deb4c.gif" alt="Weights & Biases Overview">
</p>

[Weights & Biases](https://wandb.ai/site) is a cutting-edge MLOps platform designed for tracking, visualizing, and managing machine learning experiments. It features automatic logging of training metrics for full experiment reproducibility, an interactive UI for streamlined data analysis, and efficient model management tools for deploying across various environments.

## YOLOv8 Training with Weights & Biases

You can use Weights & Biases to bring efficiency and automation to your YOLOv8 training process.

## Installation

To install the required packages, run:

!!! Tip "Installation"

    === "CLI"

        ```bash
        # Install the required packages for YOLOv8 and Weights & Biases
        pip install --upgrade ultralytics==8.0.186 wandb
        ```

For detailed instructions and best practices related to the installation process, be sure to check our [YOLOv8 Installation guide](../quickstart.md). While installing the required packages for YOLOv8, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

## Configuring Weights & Biases

After installing the necessary packages, the next step is to set up your Weights & Biases environment. This includes creating a Weights & Biases account and obtaining the necessary API key for a smooth connection between your development environment and the W&B platform.

Start by initializing the Weights & Biases environment in your workspace. You can do this by running the following command and following the prompted instructions.

!!! Tip "Initial SDK Setup"

    === "CLI"

        ```bash
        # Initialize your Weights & Biases environment
        import wandb
        wandb.login()
        ```

Navigate to the Weights & Biases authorization page to create and retrieve your API key. Use this key to authenticate your environment with W&B.

## Usage: Training YOLOv8 with Weights & Biases

Before diving into the usage instructions for YOLOv8 model training with Weights & Biases, be sure to check out the range of [YOLOv8 models offered by Ultralytics](../models/index.md). This will help you choose the most appropriate model for your project requirements.

!!! Example "Usage: Training YOLOv8 with Weights & Biases"

    === "Python"

        ```python
        import wandb
        from wandb.integration.ultralytics import add_wandb_callback

        from ultralytics import YOLO

        # Initialize a Weights & Biases run
        wandb.init(project="ultralytics", job_type="training")

        # Load a YOLO model
        model = YOLO("yolov8n.pt")

        # Add W&B Callback for Ultralytics
        add_wandb_callback(model, enable_model_checkpointing=True)

        # Train and Fine-Tune the Model
        model.train(project="ultralytics", data="coco8.yaml", epochs=5, imgsz=640)

        # Validate the Model
        model.val()

        # Perform Inference and Log Results
        model(["path/to/image1", "path/to/image2"])

        # Finalize the W&B Run
        wandb.finish()
        ```

### Understanding the Code

Let's understand the steps showcased in the usage code snippet above.

- **Step 1: Initialize a Weights & Biases Run**: Start by initializing a Weights & Biases run, specifying the project name and the job type. This run will track and manage the training and validation processes of your model.

- **Step 2: Define the YOLOv8 Model and Dataset**: Specify the model variant and the dataset you wish to use. The YOLO model is then initialized with the specified model file.

- **Step 3: Add Weights & Biases Callback for Ultralytics**: This step is crucial as it enables the automatic logging of training metrics and validation results to Weights & Biases, providing a detailed view of the model's performance.

- **Step 4: Train and Fine-Tune the Model**: Begin training the model with the specified dataset, number of epochs, and image size. The training process includes logging of metrics and predictions at the end of each epoch, offering a comprehensive view of the model's learning progress.

- **Step 5: Validate the Model**: After training, the model is validated. This step is crucial for assessing the model's performance on unseen data and ensuring its generalizability.

- **Step 6: Perform Inference and Log Results**: The model performs predictions on specified images. These predictions, along with visual overlays and insights, are automatically logged in a W&B Table for interactive exploration.

- **Step 7: Finalize the W&B Run**: This step marks the end of data logging and saves the final state of your model's training and validation process in the W&B dashboard.

### Understanding the Output

Upon running the usage code snippet above, you can expect the following key outputs:

- The setup of a new run with its unique ID, indicating the start of the training process.
- A concise summary of the model's structure, including the number of layers and parameters.
- Regular updates on important metrics such as box loss, cls loss, dfl loss, precision, recall, and mAP scores during each training epoch.
- At the end of training, detailed metrics including the model's inference speed, and overall accuracy metrics are displayed.
- Links to the Weights & Biases dashboard for in-depth analysis and visualization of the training process, along with information on local log file locations.

### Viewing the Weights & Biases Dashboard

After running the usage code snippet, you can access the Weights & Biases (W&B) dashboard through the provided link in the output. This dashboard offers a comprehensive view of your model's training process with YOLOv8.

## Key Features of the Weights & Biases Dashboard

- **Real-Time Metrics Tracking**: Observe metrics like loss, accuracy, and validation scores as they evolve during the training, offering immediate insights for model tuning. [See how experiments are tracked using Weights & Biases](https://imgur.com/D6NVnmN).

- **Hyperparameter Optimization**: Weights & Biases aids in fine-tuning critical parameters such as learning rate, batch size, and more, enhancing the performance of YOLOv8.

- **Comparative Analysis**: The platform allows side-by-side comparisons of different training runs, essential for assessing the impact of various model configurations.

- **Visualization of Training Progress**: Graphical representations of key metrics provide an intuitive understanding of the model's performance across epochs. [See how Weights & Biases helps you visualize validation results](https://imgur.com/a/kU5h7W4).

- **Resource Monitoring**: Keep track of CPU, GPU, and memory usage to optimize the efficiency of the training process.

- **Model Artifacts Management**: Access and share model checkpoints, facilitating easy deployment and collaboration.

- **Viewing Inference Results with Image Overlay**: Visualize the prediction results on images using interactive overlays in Weights & Biases, providing a clear and detailed view of model performance on real-world data. For more detailed information on Weights & Biases' image overlay capabilities, check out this [link](https://docs.wandb.ai/guides/track/log/media#image-overlays). [See how Weights & Biases' image overlays helps visualize model inferences](https://imgur.com/a/UTSiufs).

By using these features, you can effectively track, analyze, and optimize your YOLOv8 model's training, ensuring the best possible performance and efficiency.

## Summary

This guide helped you explore Ultralytics' YOLOv8 integration with Weights & Biases. It illustrates the ability of this integration to efficiently track and visualize model training and prediction results.

For further details on usage, visit [Weights & Biases' official documentation](https://docs.wandb.ai/guides/integrations/ultralytics).

Also, be sure to check out the [Ultralytics integration guide page](../integrations/index.md), to learn more about different exciting integrations.

## FAQ

### How do I install the required packages for YOLOv8 and Weights & Biases?

To install the required packages for YOLOv8 and Weights & Biases, open your command line interface and run:

```bash
pip install --upgrade ultralytics==8.0.186 wandb
```

For further guidance on installation steps, refer to our [YOLOv8 Installation guide](../quickstart.md). If you encounter issues, consult the [Common Issues guide](../guides/yolo-common-issues.md) for troubleshooting tips.

### What are the benefits of integrating Ultralytics YOLOv8 with Weights & Biases?

Integrating Ultralytics YOLOv8 with Weights & Biases offers several benefits including:

- **Real-Time Metrics Tracking:** Observe metric changes during training for immediate insights.
- **Hyperparameter Optimization:** Improve model performance by fine-tuning learning rate, batch size, etc.
- **Comparative Analysis:** Side-by-side comparison of different training runs.
- **Resource Monitoring:** Keep track of CPU, GPU, and memory usage.
- **Model Artifacts Management:** Easy access and sharing of model checkpoints.

Explore these features in detail in the Weights & Biases Dashboard section above.

### How can I configure Weights & Biases for YOLOv8 training?

To configure Weights & Biases for YOLOv8 training, follow these steps:

1. Run the command to initialize Weights & Biases:
    ```bash
    import wandb
    wandb.login()
    ```
2. Retrieve your API key from the Weights & Biases website.
3. Use the API key to authenticate your development environment.

Detailed setup instructions can be found in the Configuring Weights & Biases section above.

### How do I train a YOLOv8 model using Weights & Biases?

For training a YOLOv8 model using Weights & Biases, use the following steps in a Python script:

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO

# Initialize a Weights & Biases run
wandb.init(project="ultralytics", job_type="training")

# Load a YOLO model
model = YOLO("yolov8n.pt")

# Add W&B Callback for Ultralytics
add_wandb_callback(model, enable_model_checkpointing=True)

# Train and Fine-Tune the Model
model.train(project="ultralytics", data="coco8.yaml", epochs=5, imgsz=640)

# Validate the Model
model.val()

# Perform Inference and Log Results
model(["path/to/image1", "path/to/image2"])

# Finalize the W&B Run
wandb.finish()
```

This script initializes Weights & Biases, sets up the model, trains it, and logs results. For more details, visit the Usage section above.

### Why should I use Ultralytics YOLOv8 with Weights & Biases over other platforms?

Ultralytics YOLOv8 integrated with Weights & Biases offers several unique advantages:

- **High Efficiency:** Real-time tracking of training metrics and performance optimization.
- **Scalability:** Easily manage large-scale training jobs with robust resource monitoring and utilization tools.
- **Interactivity:** A user-friendly interactive UI for data visualization and model management.
- **Community and Support:** Strong integration documentation and community support with flexible customization and enhancement options.

For comparisons with other platforms like Comet and ClearML, refer to [Ultralytics integrations](../integrations/index.md).
