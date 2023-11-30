---
comments: true
description: Learn how to streamline and optimize your YOLOv8 model training with ClearML. This guide provides insights into integrating ClearML's MLOps tools for efficient model training, from initial setup to advanced experiment tracking and model management.
keywords: Ultralytics, YOLOv8, Object Detection, ClearML, Model Training, MLOps, Experiment Tracking, Workflow Optimization
---

# Training YOLOv8 with ClearML: Streamlining Your MLOps Workflow

MLOps bridges the gap between creating and deploying machine learning models in real-world settings. It focuses on efficient deployment, scalability, and ongoing management to ensure models perform well in practical applications.

[Ultralytics YOLOv8](https://ultralytics.com) effortlessly integrates with ClearML, streamlining and enhancing your object detection model's training and management. This guide will walk you through the integration process, detailing how to set up ClearML, manage experiments, automate model management, and collaborate effectively.

## ClearML

<p align="center">
  <img width="100%" src="https://clear.ml/wp-content/uploads/2023/06/DataOps@2x-1.png" alt="ClearML Overview">
</p>

[ClearML](https://clear.ml/) is an innovative open-source MLOps platform that is skillfully designed to automate, monitor, and orchestrate machine learning workflows. Its key features include automated logging of all training and inference data for full experiment reproducibility, an intuitive web UI for easy data visualization and analysis, advanced hyperparameter optimization algorithms, and robust model management for efficient deployment across various platforms.

## YOLOv8 Training with ClearML

You can bring automation and efficiency to your machine learning workflow by improving your training process by integrating YOLOv8 with ClearML.

## Installation

To install the required packages, run:

!!! Tip "Installation"

    === "CLI"

        ```bash
        # Install the required packages for YOLOv8 and ClearML
        pip install ultralytics clearml
        ```

For detailed instructions and best practices related to the installation process, be sure to check our [YOLOv8 Installation guide](../quickstart.md). While installing the required packages for YOLOv8, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

## Configuring ClearML

Once you have installed the necessary packages, the next step is to initialize and configure your ClearML SDK. This involves setting up your ClearML account and obtaining the necessary credentials for a seamless connection between your development environment and the ClearML server.

Begin by initializing the ClearML SDK in your environment. The ‘clearml-init’ command starts the setup process and prompts you for the necessary credentials.

!!! Tip "Initial SDK Setup"

    === "CLI"

        ```bash
        # Initialize your ClearML SDK setup process
        clearml-init
        ```

After executing this command, visit the [ClearML Settings page](https://app.clear.ml/settings/workspace-configuration). Navigate to the top right corner and select "Settings." Go to the "Workspace" section and click on "Create new credentials." Use the credentials provided in the "Create Credentials" pop-up to complete the setup as instructed, depending on whether you are configuring ClearML in a Jupyter Notebook or a local Python environment.

## Usage

Before diving into the usage instructions, be sure to check out the range of [YOLOv8 models offered by Ultralytics](../models/index.md). This will help you choose the most appropriate model for your project requirements.

!!! Example "Usage"

    === "Python"

        ```python
        from clearml import Task
        from ultralytics import YOLO

        # Step 1: Creating a ClearML Task
        task = Task.init(
            project_name="my_project",
            task_name="my_yolov8_task"
        )

        # Step 2: Selecting the YOLOv8 Model
        model_variant = "yolov8n"
        task.set_parameter("model_variant", model_variant)

        # Step 3: Loading the YOLOv8 Model
        model = YOLO(f'{model_variant}.pt')

        # Step 4: Setting Up Training Arguments
        args = dict(data="coco128.yaml", epochs=16)
        task.connect(args)

        # Step 5: Initiating Model Training
        results = model.train(**args)
        ```

### Understanding the Code

Let’s understand the steps showcased in the usage code snippet above.

**Step 1: Creating a ClearML Task**: A new task is initialized in ClearML, specifying your project and task names. This task will track and manage your model's training.

**Step 2: Selecting the YOLOv8 Model**: The `model_variant` variable is set to 'yolov8n', one of the YOLOv8 models. This variant is then logged in ClearML for tracking.

**Step 3: Loading the YOLOv8 Model**: The selected YOLOv8 model is loaded using Ultralytics' YOLO class, preparing it for training.

**Step 4: Setting Up Training Arguments**: Key training arguments like the dataset (`coco128.yaml`) and the number of epochs (`16`) are organized in a dictionary and connected to the ClearML task. This allows for tracking and potential modification via the ClearML UI. For a detailed understanding of the model training process and best practices, refer to our [YOLOv8 Model Training guide](../modes/train.md).

**Step 5: Initiating Model Training**: The model training is started with the specified arguments. The results of the training process are captured in the `results` variable.

### Understanding the Output

Upon running the usage code snippet above, you can expect the following output:

- A confirmation message indicating the creation of a new ClearML task, along with its unique ID.
- An informational message about the script code being stored, indicating that the code execution is being tracked by ClearML.
- A URL link to the ClearML results page where you can monitor the training progress and view detailed logs.
- Download progress for the YOLOv8 model and the specified dataset, followed by a summary of the model architecture and training configuration.
- Initialization messages for various training components like TensorBoard, Automatic Mixed Precision (AMP), and dataset preparation.
- Finally, the training process starts, with progress updates as the model trains on the specified dataset. For an in-depth understanding of the performance metrics used during training, read [our guide on performance metrics](../guides/yolo-performance-metrics.md).

### Viewing the ClearML Results Page

By clicking on the URL link to the ClearML results page in the output of the usage code snippet, you can access a comprehensive view of your model's training process.

#### Key Features of the ClearML Results Page

- **Real-Time Metrics Tracking**
    - Track critical metrics like loss, accuracy, and validation scores as they occur.
    - Provides immediate feedback for timely model performance adjustments.

- **Experiment Comparison**
    - Compare different training runs side-by-side.
    - Essential for hyperparameter tuning and identifying the most effective models.

- **Detailed Logs and Outputs**
    - Access comprehensive logs, graphical representations of metrics, and console outputs.
    - Gain a deeper understanding of model behavior and issue resolution.

- **Resource Utilization Monitoring**
    - Monitor the utilization of computational resources, including CPU, GPU, and memory.
    - Key to optimizing training efficiency and costs.

- **Model Artifacts Management**
    - View, download, and share model artifacts like trained models and checkpoints.
    - Enhances collaboration and streamlines model deployment and sharing.

For a visual walkthrough of what the ClearML Results Page looks like, watch the video below:

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/iLcC7m3bCes?si=oSEAoZbrg8inCg_2"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> YOLOv8 MLOps Integration using ClearML
</p>

### Advanced Features in ClearML

ClearML offers several advanced features to enhance your MLOps experience.

#### Remote Execution

ClearML's remote execution feature facilitates the reproduction and manipulation of experiments on different machines. It logs essential details like installed packages and uncommitted changes. When a task is enqueued, the ClearML Agent pulls it, recreates the environment, and runs the experiment, reporting back with detailed results.

Deploying a ClearML Agent is straightforward and can be done on various machines using the following command:

```bash
clearml-agent daemon --queue <queues_to_listen_to> [--docker]
```

This setup is applicable to cloud VMs, local GPUs, or laptops. ClearML Autoscalers help manage cloud workloads on platforms like AWS, GCP, and Azure, automating the deployment of agents and adjusting resources based on your resource budget.

### Cloning, Editing, and Enqueuing

ClearML's user-friendly interface allows easy cloning, editing, and enqueuing of tasks. Users can clone an existing experiment, adjust parameters or other details through the UI, and enqueue the task for execution. This streamlined process ensures that the ClearML Agent executing the task uses updated configurations, making it ideal for iterative experimentation and model fine-tuning.

<p align="center"><br>
  <img width="100%" src="https://clear.ml/docs/latest/assets/images/integrations_yolov5-2483adea91df4d41bfdf1a37d28864d4.gif" alt="Cloning, Editing, and Enqueuing with ClearML">
</p>

## Summary

This guide has led you through the process of integrating ClearML with Ultralytics' YOLOv8. Covering everything from initial setup to advanced model management, you've discovered how to leverage ClearML for efficient training, experiment tracking, and workflow optimization in your machine learning projects.

For further details on usage, visit [ClearML's official documentation](https://clear.ml/docs/latest/docs/integrations/yolov8/).

Additionally, explore more integrations and capabilities of Ultralytics by visiting the [Ultralytics integration guide page](../integrations/index.md), which is a treasure trove of resources and insights.
