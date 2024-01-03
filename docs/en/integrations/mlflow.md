---
comments: true
description: Uncover the utility of MLflow for effective experiment logging in your Ultralytics YOLO projects.
keywords: ultralytics docs, YOLO, MLflow, experiment logging, metrics tracking, parameter logging, artifact logging
---

# MLflow Integration for Ultralytics YOLO

<img width="1024" src="https://user-images.githubusercontent.com/26833433/274929143-05e37e72-c355-44be-a842-b358592340b7.png" alt="MLflow ecosystem">

## Introduction

Experiment logging is a crucial aspect of machine learning workflows that enables tracking of various metrics, parameters, and artifacts. It helps to enhance model reproducibility, debug issues, and improve model performance. [Ultralytics](https://ultralytics.com) YOLO, known for its real-time object detection capabilities, now offers integration with [MLflow](https://mlflow.org/), an open-source platform for complete machine learning lifecycle management.

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

Make sure that MLflow logging is enabled in Ultralytics settings. Usually, this is controlled by the settings `mflow` key. See the [settings](https://docs.ultralytics.com/quickstart/#ultralytics-settings) page for more info.

!!! Example "Update Ultralytics MLflow Settings"

    === "Python"
        Within the Python environment, call the `update` method on the `settings` object to change your settings:
        ```python
        from ultralytics import settings

        # Update a setting
        settings.update({'mlflow': True})

        # Reset settings to default values
        settings.reset()
        ```

    === "CLI"
        If you prefer using the command-line interface, the following commands will allow you to modify your settings:
        ```bash
        # Update a setting
        yolo settings runs_dir='/path/to/runs'

        # Reset settings to default values
        yolo settings reset
        ```

## How to Use

### Commands

1. **Set a Project Name**: You can set the project name via an environment variable:
    ```bash
    export MLFLOW_EXPERIMENT_NAME=<your_experiment_name>
    ```
   Or use the `project=<project>` argument when training a YOLO model, i.e. `yolo train project=my_project`.

2. **Set a Run Name**: Similar to setting a project name, you can set the run name via an environment variable:
    ```bash
    export MLFLOW_RUN=<your_run_name>
    ```
   Or use the `name=<name>` argument when training a YOLO model, i.e. `yolo train project=my_project name=my_name`.

3. **Start Local MLflow Server**: To start tracking, use:
    ```bash
    mlflow server --backend-store-uri runs/mlflow'
    ```
   This will start a local server at http://127.0.0.1:5000 by default and save all mlflow logs to the 'runs/mlflow' directory. To specify a different URI, set the `MLFLOW_TRACKING_URI` environment variable.

4. **Kill MLflow Server Instances**: To stop all running MLflow instances, run:
    ```bash
    ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9
    ```

### Logging

The logging is taken care of by the `on_pretrain_routine_end`, `on_fit_epoch_end`, and `on_train_end` callback functions. These functions are automatically called during the respective stages of the training process, and they handle the logging of parameters, metrics, and artifacts.

## Examples

1. **Logging Custom Metrics**: You can add custom metrics to be logged by modifying the `trainer.metrics` dictionary before `on_fit_epoch_end` is called.

2. **View Experiment**: To view your logs, navigate to your MLflow server (usually http://127.0.0.1:5000) and select your experiment and run.
   <img width="1024" src="https://user-images.githubusercontent.com/26833433/274933329-3127aa8c-4491-48ea-81df-ed09a5837f2a.png" alt="YOLO MLflow Experiment">

3. **View Run**: Runs are individual models inside an experiment. Click on a Run and see the Run details, including uploaded artifacts and model weights.
   <img width="1024" src="https://user-images.githubusercontent.com/26833433/274933337-ac61371c-2867-4099-a733-147a2583b3de.png" alt="YOLO MLflow Run">

## Disabling MLflow

To turn off MLflow logging:

```bash
yolo settings mlflow=False
```

## Conclusion

MLflow logging integration with Ultralytics YOLO offers a streamlined way to keep track of your machine learning experiments. It empowers you to monitor performance metrics and manage artifacts effectively, thus aiding in robust model development and deployment. For further details please visit the MLflow [official documentation](https://mlflow.org/docs/latest/index.html).
