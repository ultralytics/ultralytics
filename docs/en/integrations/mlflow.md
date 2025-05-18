---
comments: true
description: Learn how to set up and use MLflow logging with Ultralytics YOLO for enhanced experiment tracking, model reproducibility, and performance improvements.
keywords: MLflow, Ultralytics YOLO, machine learning, experiment tracking, metrics logging, parameter logging, artifact logging
---

# MLflow Integration for Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/mlflow-integration-ultralytics-yolo.avif" alt="MLflow ecosystem">

## Introduction

Experiment logging is a crucial aspect of [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) workflows that enables tracking of various metrics, parameters, and artifacts. It helps to enhance model reproducibility, debug issues, and improve model performance. [Ultralytics](https://www.ultralytics.com/) YOLO, known for its real-time [object detection](https://www.ultralytics.com/glossary/object-detection) capabilities, now offers integration with [MLflow](https://mlflow.org/), an open-source platform for complete machine learning lifecycle management.

This documentation page is a comprehensive guide to setting up and utilizing the MLflow logging capabilities for your Ultralytics YOLO project.

## What is MLflow?

[MLflow](https://mlflow.org/) is an open-source platform developed by [Databricks](https://www.databricks.com/) for managing the end-to-end machine learning lifecycle. It includes tools for tracking experiments, packaging code into reproducible runs, and sharing and deploying models. MLflow is designed to work with any machine learning library and programming language.

## Features

- **Metrics Logging**: Logs metrics at the end of each epoch and at the end of the training.
- **Parameter Logging**: Logs all the parameters used in the training.
- **Artifacts Logging**: Logs model artifacts, including weights and configuration files, at the end of the training.

## Setup and Prerequisites

Ensure MLflow is installed. If not, install it using pip:

```bash
pip install mlflow
```

Make sure that MLflow logging is enabled in Ultralytics settings. Usually, this is controlled by the settings `mlflow` key. See the [settings](../quickstart.md#ultralytics-settings) page for more info.

!!! example "Update Ultralytics MLflow Settings"

    === "Python"

        Within the Python environment, call the `update` method on the `settings` object to change your settings:
        ```python
        from ultralytics import settings

        # Update a setting
        settings.update({"mlflow": True})

        # Reset settings to default values
        settings.reset()
        ```

    === "CLI"

        If you prefer using the command-line interface, the following commands will allow you to modify your settings:
        ```bash
        # Update a setting
        yolo settings mlflow=True

        # Reset settings to default values
        yolo settings reset
        ```

## How to Use

### Commands

1. **Set a Project Name**: You can set the project name via an environment variable:

    ```bash
    export MLFLOW_EXPERIMENT_NAME=YOUR_EXPERIMENT_NAME
    ```

    Or use the `project=<project>` argument when training a YOLO model, i.e. `yolo train project=my_project`.

2. **Set a Run Name**: Similar to setting a project name, you can set the run name via an environment variable:

    ```bash
    export MLFLOW_RUN=YOUR_RUN_NAME
    ```

    Or use the `name=<name>` argument when training a YOLO model, i.e. `yolo train project=my_project name=my_name`.

3. **Start Local MLflow Server**: To start tracking, use:

    ```bash
    mlflow server --backend-store-uri runs/mlflow
    ```

    This will start a local server at `http://127.0.0.1:5000` by default and save all mlflow logs to the 'runs/mlflow' directory. To specify a different URI, set the `MLFLOW_TRACKING_URI` environment variable.

4. **Kill MLflow Server Instances**: To stop all running MLflow instances, run:

    ```bash
    ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9
    ```

### Logging

The logging is taken care of by the `on_pretrain_routine_end`, `on_fit_epoch_end`, and `on_train_end` [callback functions](../reference/utils/callbacks/mlflow.md). These functions are automatically called during the respective stages of the training process, and they handle the logging of parameters, metrics, and artifacts.

## Examples

1. **Logging Custom Metrics**: You can add custom metrics to be logged by modifying the `trainer.metrics` dictionary before `on_fit_epoch_end` is called.

2. **View Experiment**: To view your logs, navigate to your MLflow server (usually `http://127.0.0.1:5000`) and select your experiment and run. <img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/yolo-mlflow-experiment.avif" alt="YOLO MLflow Experiment">

3. **View Run**: Runs are individual models inside an experiment. Click on a Run and see the Run details, including uploaded artifacts and model weights. <img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/yolo-mlflow-run.avif" alt="YOLO MLflow Run">

## Disabling MLflow

To turn off MLflow logging:

```bash
yolo settings mlflow=False
```

## Conclusion

MLflow logging integration with Ultralytics YOLO offers a streamlined way to keep track of your [machine learning experiments](https://www.ultralytics.com/blog/log-ultralytics-yolo-experiments-using-mlflow-integration). It empowers you to monitor performance metrics and manage artifacts effectively, thus aiding in robust model development and deployment. For further details please visit the MLflow [official documentation](https://mlflow.org/docs/latest/index.html).

## FAQ

### How do I set up MLflow logging with Ultralytics YOLO?

To set up MLflow logging with Ultralytics YOLO, you first need to ensure MLflow is installed. You can install it using pip:

```bash
pip install mlflow
```

Next, enable MLflow logging in Ultralytics settings. This can be controlled using the `mlflow` key. For more information, see the [settings guide](../quickstart.md#ultralytics-settings).

!!! example "Update Ultralytics MLflow Settings"

    === "Python"

        ```python
        from ultralytics import settings

        # Update a setting
        settings.update({"mlflow": True})

        # Reset settings to default values
        settings.reset()
        ```

    === "CLI"

        ```bash
        # Update a setting
        yolo settings mlflow=True

        # Reset settings to default values
        yolo settings reset
        ```

Finally, start a local MLflow server for tracking:

```bash
mlflow server --backend-store-uri runs/mlflow
```

### What metrics and parameters can I log using MLflow with Ultralytics YOLO?

Ultralytics YOLO with MLflow supports logging various metrics, parameters, and artifacts throughout the training process:

- **Metrics Logging**: Tracks metrics at the end of each [epoch](https://www.ultralytics.com/glossary/epoch) and upon training completion.
- **Parameter Logging**: Logs all parameters used in the training process.
- **Artifacts Logging**: Saves model artifacts like weights and configuration files after training.

For more detailed information, visit the [Ultralytics YOLO tracking documentation](#features).

### Can I disable MLflow logging once it is enabled?

Yes, you can disable MLflow logging for Ultralytics YOLO by updating the settings. Here's how you can do it using the CLI:

```bash
yolo settings mlflow=False
```

For further customization and resetting settings, refer to the [settings guide](../quickstart.md#ultralytics-settings).

### How can I start and stop an MLflow server for Ultralytics YOLO tracking?

To start an MLflow server for tracking your experiments in Ultralytics YOLO, use the following command:

```bash
mlflow server --backend-store-uri runs/mlflow
```

This command starts a local server at `http://127.0.0.1:5000` by default. If you need to stop running MLflow server instances, use the following bash command:

```bash
ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9
```

Refer to the [commands section](#commands) for more command options.

### What are the benefits of integrating MLflow with Ultralytics YOLO for experiment tracking?

Integrating MLflow with Ultralytics YOLO offers several benefits for managing your machine learning experiments:

- **Enhanced Experiment Tracking**: Easily track and compare different runs and their outcomes.
- **Improved Model Reproducibility**: Ensure that your experiments are reproducible by logging all parameters and artifacts.
- **Performance Monitoring**: Visualize performance metrics over time to make data-driven decisions for model improvements.
- **Streamlined Workflow**: Automate the logging process to focus more on model development rather than manual tracking.
- **Collaborative Development**: Share experiment results with team members for better collaboration and knowledge sharing.

For an in-depth look at setting up and leveraging MLflow with Ultralytics YOLO, explore the [MLflow Integration for Ultralytics YOLO](#introduction) documentation.
