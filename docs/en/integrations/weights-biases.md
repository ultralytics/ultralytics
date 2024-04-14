---
comments: true
description: Discover how to train your YOLOv8 models efficiently with Weights & Biases. This guide walks through integrating Weights & Biases with YOLOv8 to enable seamless experiment tracking, result visualization, and model explainability.
keywords: Ultralytics, YOLOv8, Object Detection, Weights & Biases, Model Training, Experiment Tracking, Visualizing Results
---

# Enhancing YOLOv8 Experiment Tracking and Visualization with Weights & Biases

Object detection models like [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) have become integral to many computer vision applications. However, training, evaluating, and deploying these complex models introduces several challenges. Tracking key training metrics, comparing model variants, analyzing model behavior, and detecting issues require substantial instrumentation and experiment management.

This guide showcases Ultralytics YOLOv8 integration with Weights & Biases’ for enhanced experiment tracking, model-checkpointing, and visualization of model performance. It also includes instructions for setting up the integration, training, fine-tuning, and visualizing results using Weights & Biases’ interactive features.

## Weights & Biases

<p align="center">
  <img width="640" src="https://docs.wandb.ai/assets/images/wandb_demo_experiments-4797af7fe7236d6c5c42adbdc93deb4c.gif" alt="Weights & Biases Overview">
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
       from ultralytics import YOLO
       from wandb.integration.ultralytics import add_wandb_callback
       import wandb

       # Step 1: Initialize a Weights & Biases run
       wandb.init(project="ultralytics", job_type="training")

       # Step 2: Define the YOLOv8 Model and Dataset
       model_name = "yolov8n"
       dataset_name = "coco128.yaml"
       model = YOLO(f"{model_name}.pt")

       # Step 3: Add W&B Callback for Ultralytics
       add_wandb_callback(model, enable_model_checkpointing=True)

       # Step 4: Train and Fine-Tune the Model
       model.train(project="ultralytics", data=dataset_name, epochs=5, imgsz=640)

       # Step 5: Validate the Model
       model.val()

       # Step 6: Perform Inference and Log Results
       model(["path/to/image1", "path/to/image2"])

       # Step 7: Finalize the W&B Run
       wandb.finish()
       ```

### Understanding the Code

Let’s understand the steps showcased in the usage code snippet above.

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
- A concise summary of the model’s structure, including the number of layers and parameters.
- Regular updates on important metrics such as box loss, cls loss, dfl loss, precision, recall, and mAP scores during each training epoch.
- At the end of training, detailed metrics including the model's inference speed, and overall accuracy metrics are displayed.
- Links to the Weights & Biases dashboard for in-depth analysis and visualization of the training process, along with information on local log file locations.

### Viewing the Weights & Biases Dashboard

After running the usage code snippet, you can access the Weights & Biases (W&B) dashboard through the provided link in the output. This dashboard offers a comprehensive view of your model's training process with YOLOv8.

## Key Features of the Weights & Biases Dashboard

- **Real-Time Metrics Tracking**: Observe metrics like loss, accuracy, and validation scores as they evolve during the training, offering immediate insights for model tuning.

    <div style="text-align:center;"><blockquote class="imgur-embed-pub" lang="en" data-id="a/TB76U9O"><a href="//imgur.com/D6NVnmN">Take a look at how the experiments are tracked using Weights & Biases.</a></blockquote></div><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

- **Hyperparameter Optimization**: Weights & Biases aids in fine-tuning critical parameters such as learning rate, batch size, and more, enhancing the performance of YOLOv8.

- **Comparative Analysis**: The platform allows side-by-side comparisons of different training runs, essential for assessing the impact of various model configurations.

- **Visualization of Training Progress**: Graphical representations of key metrics provide an intuitive understanding of the model's performance across epochs.

    <div style="text-align:center;"><blockquote class="imgur-embed-pub" lang="en" data-id="a/kU5h7W4" data-context="false" ><a href="//imgur.com/a/kU5h7W4">Take a look at how Weights & Biases helps you visualize validation results.</a></blockquote></div><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

- **Resource Monitoring**: Keep track of CPU, GPU, and memory usage to optimize the efficiency of the training process.

- **Model Artifacts Management**: Access and share model checkpoints, facilitating easy deployment and collaboration.

- **Viewing Inference Results with Image Overlay**: Visualize the prediction results on images using interactive overlays in Weights & Biases, providing a clear and detailed view of model performance on real-world data. For more detailed information on Weights & Biases’ image overlay capabilities, check out this [link](https://docs.wandb.ai/guides/track/log/media#image-overlays).

    <div style="text-align:center;"><blockquote class="imgur-embed-pub" lang="en" data-id="a/UTSiufs" data-context="false" ><a href="//imgur.com/a/UTSiufs">Take a look at how Weights & Biases’ image overlays helps visualize model inferences.</a></blockquote></div><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

By using these features, you can effectively track, analyze, and optimize your YOLOv8 model's training, ensuring the best possible performance and efficiency.

## Summary

This guide helped you explore Ultralytics’ YOLOv8 integration with Weights & Biases. It illustrates the ability of this integration to efficiently track and visualize model training and prediction results.

For further details on usage, visit [Weights & Biases' official documentation](https://docs.wandb.ai/guides/integrations/ultralytics).

Also, be sure to check out the [Ultralytics integration guide page](../integrations/index.md), to learn more about different exciting integrations.
