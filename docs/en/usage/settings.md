---
comments: true
description: Learn how to view, modify, and reset Ultralytics' persistent settings, including dataset, weights, and run directories, plus integration toggles.
keywords: Ultralytics settings, SettingsManager, YOLO settings, runs_dir, datasets_dir, weights_dir, integration toggles, YOLO configuration
---

# Ultralytics Settings

The Ultralytics library includes a `SettingsManager` for fine-grained control over experiments, allowing users to access and modify settings easily. Stored in a JSON file within the environment's user configuration directory, these settings can be viewed or modified in the Python environment or via the Command-Line Interface (CLI).

## Inspecting Settings

To view the current configuration of your settings:

!!! example "View settings"

    === "Python"

        Use Python to view your settings by importing the `settings` object from the `ultralytics` module. Print and return settings with these commands:
        ```python
        from ultralytics import settings

        # View all settings
        print(settings)

        # Return a specific setting
        value = settings["runs_dir"]
        ```

    === "CLI"

        The command-line interface allows you to check your settings with:
        ```bash
        yolo settings
        ```

## Modifying Settings

Ultralytics makes it easy to modify settings in the following ways:

!!! example "Update settings"

    === "Python"

        In Python, use the `update` method on the `settings` object:
        ```python
        from ultralytics import settings

        # Update a setting
        settings.update({"runs_dir": "/path/to/runs"})

        # Update multiple settings
        settings.update({"runs_dir": "/path/to/runs", "tensorboard": False})

        # Reset settings to default values
        settings.reset()
        ```

    === "CLI"

        To modify settings using the command-line interface:
        ```bash
        # Update a setting
        yolo settings runs_dir='/path/to/runs'

        # Update multiple settings
        yolo settings runs_dir='/path/to/runs' tensorboard=False

        # Reset settings to default values
        yolo settings reset
        ```

## Understanding Settings

The table below overviews the adjustable settings within Ultralytics, including example values, data types, and descriptions.

| Name               | Example Value         | Data Type | Description                                                                                                      |
| ------------------ | --------------------- | --------- | ---------------------------------------------------------------------------------------------------------------- |
| `settings_version` | `'0.0.6'`             | `str`     | Ultralytics _settings_ version (distinct from the Ultralytics [pip] version)                                     |
| `datasets_dir`     | `'/path/to/datasets'` | `str`     | Directory where datasets are stored                                                                              |
| `weights_dir`      | `'/path/to/weights'`  | `str`     | Directory where model weights are stored                                                                         |
| `runs_dir`         | `'/path/to/runs'`     | `str`     | Directory where experiment runs are stored                                                                       |
| `uuid`             | `'a1b2c3d4'`          | `str`     | Unique identifier for the current settings                                                                       |
| `sync`             | `True`                | `bool`    | Option to sync analytics and crashes to [Ultralytics Platform]                                                   |
| `api_key`          | `''`                  | `str`     | [Ultralytics Platform] API Key                                                                                   |
| `openai_api_key`   | `''`                  | `str`     | OpenAI API Key for the [Explorer](../datasets/explorer/index.md) dashboard's Ask AI feature                      |
| `clearml`          | `True`                | `bool`    | Option to use [ClearML] logging                                                                                  |
| `comet`            | `True`                | `bool`    | Option to use [Comet ML] for experiment tracking and visualization                                               |
| `dvc`              | `True`                | `bool`    | Option to use [DVC for experiment tracking] and version control                                                  |
| `hub`              | `True`                | `bool`    | Option to use [Ultralytics Platform] integration                                                                 |
| `mlflow`           | `True`                | `bool`    | Option to use [MLFlow] for experiment tracking                                                                   |
| `neptune`          | `True`                | `bool`    | Option to use [Neptune] for experiment tracking                                                                  |
| `raytune`          | `True`                | `bool`    | Option to use [Ray Tune] for [hyperparameter tuning](https://www.ultralytics.com/glossary/hyperparameter-tuning) |
| `tensorboard`      | `False`               | `bool`    | Option to use [TensorBoard] for visualization                                                                    |
| `wandb`            | `False`               | `bool`    | Option to use [Weights & Biases] logging                                                                         |
| `vscode_msg`       | `True`                | `bool`    | When a VS Code terminal is detected, enables a prompt to download the [Ultralytics-Snippets] extension.          |
| `openvino_msg`     | `True`                | `bool`    | Shows a one-time tip to export to OpenVINO when exporting on Intel CPU hardware                                  |

Revisit these settings as you progress through projects or experiments to ensure optimal configuration.

<!-- Article Links -->

[Ultralytics Platform]: https://platform.ultralytics.com
[pip]: https://pypi.org/project/ultralytics/
[DVC for experiment tracking]: https://dvc.org/doc/dvclive/ml-frameworks/yolo
[Comet ML]: https://bit.ly/yolov8-readme-comet
[ClearML]: ../integrations/clearml.md
[MLFlow]: ../integrations/mlflow.md
[Neptune]: https://neptune.ai/
[Tensorboard]: ../integrations/tensorboard.md
[Ray Tune]: ../integrations/ray-tune.md
[Weights & Biases]: ../integrations/weights-biases.md
[Ultralytics-Snippets]: ../integrations/vscode.md
