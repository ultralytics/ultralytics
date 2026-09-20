---
comments: true
description: View, change, and reset persistent Ultralytics settings such as the datasets, weights, and runs directories, the Platform API key, and experiment-logger toggles.
keywords: Ultralytics settings, SettingsManager, yolo settings, runs_dir, datasets_dir, weights_dir, experiment tracking, YOLO configuration
---

# Ultralytics Settings

Ultralytics stores persistent, per-machine settings in a JSON file in the user configuration directory and exposes them through the `settings` object in Python and the `yolo settings` command. They control where datasets, weights, and runs are stored, the [Ultralytics Platform](https://platform.ultralytics.com) API key, and which experiment trackers are enabled. Per-run arguments such as `imgsz` and `epochs` are documented on the [Configuration](cfg.md) page instead.

!!! example "View, update, and reset settings"

    === "Python"

        ```python
        from ultralytics import settings

        print(settings)  # view all settings
        value = settings["runs_dir"]  # read one setting
        settings.update({"runs_dir": "/path/to/runs", "tensorboard": False})  # update one or more settings
        settings.reset()  # restore defaults
        ```

    === "CLI"

        ```bash
        yolo settings                                            # view all settings
        yolo settings runs_dir='/path/to/runs' tensorboard=False # update one or more settings
        yolo settings reset                                      # restore defaults
        ```

## Settings Reference

| Name               | Example Value         | Data Type | Description                                                                                                                                  |
| ------------------ | --------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `settings_version` | `'0.0.8'`             | `str`     | Settings schema version, distinct from the `ultralytics` package version                                                                     |
| `datasets_dir`     | `'/path/to/datasets'` | `str`     | Directory where datasets are stored                                                                                                          |
| `weights_dir`      | `'/path/to/weights'`  | `str`     | Directory where model weights are stored                                                                                                     |
| `runs_dir`         | `'/path/to/runs'`     | `str`     | Directory where experiment runs are stored                                                                                                   |
| `uuid`             | `'a1b2c3d4'`          | `str`     | Anonymized machine identifier (SHA-256 hash) used for analytics                                                                              |
| `sync`             | `True`                | `bool`    | Send anonymized analytics and crash reports to Ultralytics, see [Privacy](../help/privacy.md)                                                |
| `api_key`          | `''`                  | `str`     | [Ultralytics Platform](https://platform.ultralytics.com) API key                                                                             |
| `openai_api_key`   | `''`                  | `str`     | OpenAI API key for the [Explorer](../datasets/explorer/index.md) Ask AI feature, available up to `ultralytics==8.3.11`                       |
| `clearml`          | `True`                | `bool`    | Enable [ClearML](../integrations/clearml.md) logging                                                                                         |
| `comet`            | `True`                | `bool`    | Enable [Comet ML](../integrations/comet.md) experiment tracking                                                                              |
| `dvc`              | `True`                | `bool`    | Enable [DVC](../integrations/dvc.md) experiment tracking                                                                                     |
| `mlflow`           | `True`                | `bool`    | Enable [MLflow](../integrations/mlflow.md) experiment tracking                                                                               |
| `raytune`          | `True`                | `bool`    | Enable [Ray Tune](../integrations/ray-tune.md) [hyperparameter tuning](https://www.ultralytics.com/glossary/hyperparameter-tuning) reporting |
| `tensorboard`      | `False`               | `bool`    | Enable [TensorBoard](../integrations/tensorboard.md) logging                                                                                 |
| `wandb`            | `False`               | `bool`    | Enable [Weights & Biases](../integrations/weights-biases.md) logging                                                                         |
| `vscode_msg`       | `True`                | `bool`    | Show a prompt to install the [Ultralytics-Snippets](../integrations/vscode.md) extension when a VS Code terminal is detected                 |
| `openvino_msg`     | `True`                | `bool`    | On Intel CPUs, show a tip suggesting [OpenVINO](../integrations/openvino.md) export for faster inference                                     |
