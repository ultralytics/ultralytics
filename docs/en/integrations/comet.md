---
title: YOLO26 Training Logging with Comet ML
comments: true
description: Log Ultralytics YOLO26 training to Comet, covering installation, API key setup, automatic metrics and checkpoints, custom logging variables, and offline mode.
keywords: YOLO26, Comet, Comet ML, logging, machine learning, training, model checkpoints, metrics, installation, configuration, real-time insights, custom logging
---

# Elevating YOLO26 Training: Simplify Your Logging Process with Comet

Logging key training details such as parameters, metrics, image predictions, and model checkpoints is essential in [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml)—it keeps your project transparent, your progress measurable, and your results repeatable.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/LPodYpvKkvI"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Use Comet for Ultralytics YOLO Model Training Logs and Metrics 🚀
</p>

[Ultralytics YOLO26](../models/yolo26.md) integrates with Comet (formerly Comet ML) to capture the training process of your YOLO26 [object detection](https://www.ultralytics.com/glossary/object-detection) models. This guide covers installation, Comet setup, real-time insights, custom logging, and offline usage, so that your YOLO26 training runs are documented and reproducible.

## Comet

<p align="center">
  <img width="640" src="https://www.comet.com/docs/v2/img/landing/home-hero.svg" alt="Comet ML experiment tracking dashboard">
</p>

[Comet](https://www.comet.com/site/) is a platform for tracking, comparing, explaining, and optimizing machine learning models and experiments. It allows you to log metrics, parameters, media, and more during your model training and monitor your experiments through a web interface. Comet helps data scientists iterate more rapidly, enhances transparency and reproducibility, and aids in the development of production models.

## Harnessing the Power of YOLO26 and Comet

Combining Ultralytics YOLO26 with Comet offers several benefits, including simplified [experiment tracking](https://www.ultralytics.com/glossary/experiment-tracking), real-time insights for quick adjustments, flexible and tailored logging options, and the ability to log experiments offline when internet access is limited. This integration helps you make data-driven decisions and analyze performance metrics.

## Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install YOLO26 and Comet
        pip install ultralytics comet_ml
        ```

## Configuring Comet

After installing the required packages, you'll need to sign up, get a [Comet API Key](https://www.comet.com/signup), and configure it.

!!! tip "Configuring Comet"

    === "CLI"

        ```bash
        # Set your Comet API Key
        export COMET_API_KEY=YOUR_API_KEY
        ```

Then, authenticate with Comet. The Ultralytics callback creates the experiment and uses the YOLO `project` training argument as the Comet project name.

!!! example "Initialize Comet project"

    === "Python"

        ```python
        import comet_ml

        comet_ml.login()
        ```

If you are using a Google Colab notebook, the code above will prompt you to enter your API key for initialization.

## Usage

Before training, review the available [YOLO26 models](https://platform.ultralytics.com/ultralytics/yolo26) to choose the size and task that fit your project.

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")

        # Train the model
        results = model.train(
            data="coco8.yaml",
            project="comet-example-yolo26-coco128",
            batch=32,
            save_period=1,
            save_json=True,
            epochs=3,
        )
        ```

After running the training code, Comet will create an experiment in your Comet workspace to track the run automatically. You will then be provided with a link to view the detailed logging of your [YOLO26 model's training](../modes/train.md) process.

Once Comet is installed and configured, its callback is enabled by default and logs metrics such as mAP and loss, hyperparameters, model checkpoints, and an interactive confusion matrix. Image [bounding box](https://www.ultralytics.com/glossary/bounding-box) predictions from the validation set are logged for detect and segment training with `save_json=True`. Run `yolo settings comet=False` to disable the integration.

## Understanding Your Model's Performance with Comet Visualizations

Once your YOLO26 model begins training, the Comet dashboard presents automatically logged information through visuals and statistics:

**Experiment Panels**

The experiment panels section of the Comet dashboard organizes and presents the different runs and their metrics, such as segment mask loss, class loss, precision, and [mean average precision](https://www.ultralytics.com/glossary/mean-average-precision-map).

<p align="center">
  <img width="640" src="https://cdn.ul.run/i/1c5db1d4fb62e6c797aa07edb63d5bd1.avif" alt="Comet experiment panels comparing YOLO26 training runs and metrics">
</p>

**Metrics**

In the metrics section, you have the option to examine the metrics in a tabular format as well, which is displayed in a dedicated pane as illustrated here.

<p align="center">
  <img width="640" src="https://cdn.ul.run/i/bafc0048079ac6eb4f75810873cebad7.avif" alt="Comet metrics pane listing YOLO26 training metrics in a table">
</p>

**Interactive [Confusion Matrix](https://www.ultralytics.com/glossary/confusion-matrix)**

The confusion matrix, found in the Confusion Matrix tab, provides an interactive way to assess the model's classification [accuracy](https://www.ultralytics.com/glossary/accuracy). It details the correct and incorrect predictions, allowing you to understand the model's strengths and weaknesses.

<p align="center">
  <img width="640" src="https://cdn.ul.run/i/3b562147e59c9e850c163417886a8ea8.avif" alt="Comet interactive confusion matrix for YOLO26 validation results">
</p>

**System Metrics**

Comet logs system metrics to help identify any bottlenecks in the training process. It includes metrics such as GPU utilization, GPU memory usage, CPU utilization, and RAM usage. These are essential for monitoring the efficiency of resource usage during model training.

<p align="center">
  <img width="640" src="https://cdn.ul.run/i/7182de7672f7dd9f5bb9a52fdd19c265.avif" alt="Comet system metrics showing GPU and CPU utilization during training">
</p>

## Customizing Comet Logging

Comet offers the flexibility to customize its logging behavior by setting environment variables. These configurations allow you to tailor Comet to your specific needs and preferences. The Ultralytics callback reads the following environment variables (set them before training starts):

| Environment Variable                | Default        | Description                                                                                               |
| ----------------------------------- | -------------- | --------------------------------------------------------------------------------------------------------- |
| `COMET_START_ONLINE`                | `1`            | Run the experiment in online (`1`) or offline (`0`) mode.                                                 |
| `COMET_PROJECT_NAME`                | `args.project` | Comet workspace project. Falls back to the YOLO `project` training argument when unset.                   |
| `COMET_MODEL_NAME`                  | `Ultralytics`  | Name registered for the logged model artifact.                                                            |
| `COMET_MAX_IMAGE_PREDICTIONS`       | `100`          | Total number of validation image predictions to log per run.                                              |
| `COMET_EVAL_BATCH_LOGGING_INTERVAL` | `1`            | Log image predictions every Nth validation batch.                                                         |
| `COMET_EVAL_LOG_IMAGE_PREDICTIONS`  | `true`         | Toggle image prediction logging on (`true`) or off (`false`).                                             |
| `COMET_EVAL_LOG_CONFUSION_MATRIX`   | `false`        | Log a confusion matrix at each `save_period` checkpoint. A final matrix is always logged after training.  |
| `COMET_MAX_CONFIDENCE_SCORE`        | `100.0`        | Multiplier applied to detection confidence scores before logging (Comet's UI expects a percentage scale). |
| `COMET_MODE` _(deprecated)_         | `online`       | Legacy alias of `COMET_START_ONLINE` (`"online"` ↔ `1`, `"offline"` ↔ `0`). Emits a deprecation warning.  |

For the boolean toggles (`COMET_EVAL_LOG_IMAGE_PREDICTIONS` and `COMET_EVAL_LOG_CONFUSION_MATRIX`), the values `1`, `true`, `yes`, `on`, `y`, and `t` are parsed as `True` (case-insensitive). Any other value is `False`.

### Logging Image Predictions

You can control the number of image predictions that Comet logs during your experiments. With `save_json=True`, Comet logs up to 100 image predictions from the validation set. However, you can change this number to better suit your requirements. For example, to log 200 image predictions, use the following code:

```python
import os

os.environ["COMET_MAX_IMAGE_PREDICTIONS"] = "200"
```

To disable annotated image prediction logging (for example, to reduce upload volume on slow connections), set `COMET_EVAL_LOG_IMAGE_PREDICTIONS` to `"false"`:

```python
import os

os.environ["COMET_EVAL_LOG_IMAGE_PREDICTIONS"] = "false"
```

The `train_batch*.jpg` and `val_batch*.jpg` plot images are still logged at the end of training.

### Batch Logging Interval

Comet allows you to specify how often batches of image predictions are logged. The `COMET_EVAL_BATCH_LOGGING_INTERVAL` environment variable controls this frequency. The default setting is 1, which logs predictions from every validation batch. You can adjust this value to log predictions at a different interval. For instance, setting it to 4 will log predictions from every fourth batch.

```python
import os

os.environ["COMET_EVAL_BATCH_LOGGING_INTERVAL"] = "4"
```

### Per-Epoch Confusion Matrix Logging

By default, the confusion matrix from your validation set is logged once, after the training is completed. To also log it at every checkpoint saved by `save_period` (`save_period=1` saves every [epoch](https://www.ultralytics.com/glossary/epoch)), set the `COMET_EVAL_LOG_CONFUSION_MATRIX` environment variable to "true."

```python
import os

os.environ["COMET_EVAL_LOG_CONFUSION_MATRIX"] = "true"
```

### Online and Offline Mode

By default, Comet runs in online mode and streams experiment data to the Comet servers. If you need to train without internet access, set `COMET_START_ONLINE=0` before training starts. Experiment data is saved locally and can be uploaded later with the [`comet upload`](https://www.comet.com/docs/v2/api-and-sdk/command-line/reference/) CLI.

```python
import os

os.environ["COMET_START_ONLINE"] = "0"  # 1 (default) = online, 0 = offline
```

!!! note "`COMET_MODE` is deprecated"

    Earlier versions used `COMET_MODE="offline"` for this purpose. The variable is still honored for backward compatibility but emits a deprecation warning. Use `COMET_START_ONLINE` going forward.

### Project Name

By default, the Comet callback passes the YOLO `project` training argument to Comet (or `None` when the argument is unset, in which case Comet uses your workspace default). Override this with `COMET_PROJECT_NAME` to send all experiments to a specific Comet workspace project regardless of the YOLO training argument:

```python
import os

os.environ["COMET_PROJECT_NAME"] = "my-yolo26-experiments"
```

### Model Artifact Name

`COMET_MODEL_NAME` sets the name Comet registers for the logged model artifact (defaults to `Ultralytics`). Use it to differentiate model variants in a shared workspace:

```python
import os

os.environ["COMET_MODEL_NAME"] = "yolo26n-coco128"
```

### Confidence Score Scaling

Detection confidence scores are emitted in the `[0, 1]` range, but the Comet UI displays them on a percentage scale by default. The callback multiplies each score by `COMET_MAX_CONFIDENCE_SCORE` (default `100.0`) before logging. Adjust this if you prefer raw probabilities or a different scale:

```python
import os

os.environ["COMET_MAX_CONFIDENCE_SCORE"] = "1.0"  # log raw [0, 1] scores
```

## Summary

This guide has walked you through integrating Comet with Ultralytics YOLO26. From installation to customization, you've learned to streamline experiment management, gain real-time insights, and adapt logging to your project's needs.

Explore [Comet's official YOLOv8 integration documentation](https://www.comet.com/docs/v2/integrations/third-party-tools/yolov8/), which also applies to YOLO26 projects, and the Ultralytics blog post [YOLOv5 + Comet: Track & Optimize Models](https://www.ultralytics.com/blog/how-to-use-yolov5-with-comet) for an earlier walkthrough of the same workflow.

For a practical [image segmentation](https://www.ultralytics.com/glossary/image-segmentation) example, the Comet blog post [Fine-Tuning YOLOv8 for Image Segmentation with Comet](https://www.comet.com/site/blog/fine-tuning-yolov8-for-image-segmentation-with-comet/) walks through a segmentation fine-tuning project step by step; the same workflow applies to YOLO26.

To explore other Ultralytics integrations, visit the [Ultralytics integrations guide](../integrations/index.md).

## FAQ

### How do I integrate Comet with Ultralytics YOLO26 for training?

To integrate Comet with Ultralytics YOLO26, follow these steps:

1. **Install the required packages**:

    ```bash
    pip install ultralytics comet_ml
    ```

2. **Set up your Comet API Key**:

    ```bash
    export COMET_API_KEY=YOUR_API_KEY
    ```

3. **Authenticate with Comet in your Python code**:

    ```python
    import comet_ml

    comet_ml.login()
    ```

4. **Train your YOLO26 model and log metrics**:

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    results = model.train(
        data="coco8.yaml",
        project="comet-example-yolo26-coco128",
        batch=32,
        save_period=1,
        save_json=True,
        epochs=3,
    )
    ```

For more detailed instructions, refer to the [Comet configuration section](#configuring-comet).

### What are the benefits of using Comet with YOLO26?

By integrating Ultralytics YOLO26 with Comet, you can:

- **Monitor real-time insights**: Get instant feedback on your training results, allowing for quick adjustments.
- **Log extensive metrics**: Automatically capture essential metrics such as mAP, loss, hyperparameters, and model checkpoints.
- **Track experiments offline**: Log your training runs locally when internet access is unavailable.
- **Compare different training runs**: Use the interactive Comet dashboard to analyze and compare multiple experiments.

With these features, you can optimize your machine learning workflows for better performance and reproducibility. For more integrations, visit the [Ultralytics integrations guide](../integrations/index.md).

### How do I customize the logging behavior of Comet during YOLO26 training?

Comet allows for extensive customization of its logging behavior using environment variables:

- **Change the number of image predictions logged**:

    ```python
    import os

    os.environ["COMET_MAX_IMAGE_PREDICTIONS"] = "200"
    ```

- **Adjust batch logging interval**:

    ```python
    import os

    os.environ["COMET_EVAL_BATCH_LOGGING_INTERVAL"] = "4"
    ```

- **Log the confusion matrix at every `save_period` checkpoint**:

    ```python
    import os

    os.environ["COMET_EVAL_LOG_CONFUSION_MATRIX"] = "true"
    ```

- **Set the Comet project name**:

    ```python
    import os

    os.environ["COMET_PROJECT_NAME"] = "my-yolo26-experiments"
    ```

- **Set the logged model artifact name**:

    ```python
    import os

    os.environ["COMET_MODEL_NAME"] = "yolo26n-coco128"
    ```

See the [Customizing Comet Logging](#customizing-comet-logging) section for the full list, including image prediction toggles, confidence score scaling, and online/offline mode.

### How do I view detailed metrics and visualizations of my YOLO26 training on Comet?

Once your YOLO26 model starts training, you can access a wide range of metrics and visualizations on the Comet dashboard. Key features include:

- **Experiment Panels**: View different runs and their metrics, including segment mask loss, class loss, and mean average precision.
- **Metrics**: Examine metrics in tabular format for detailed analysis.
- **Interactive Confusion Matrix**: Assess classification accuracy with an interactive confusion matrix.
- **System Metrics**: Monitor GPU and CPU utilization, memory usage, and other system metrics.

For a detailed overview of these features, visit the [Understanding Your Model's Performance with Comet Visualizations](#understanding-your-models-performance-with-comet-visualizations) section.

### Can I use Comet for offline logging when training YOLO26 models?

Yes. Set `COMET_START_ONLINE=0` before training starts to log locally:

```python
import os

os.environ["COMET_START_ONLINE"] = "0"
```

Experiment data is saved on disk and can be uploaded to Comet later with the [`comet upload`](https://www.comet.com/docs/v2/api-and-sdk/command-line/reference/) CLI when connectivity is available. The earlier `COMET_MODE="offline"` variable still works but emits a deprecation warning. For more details, see the [Online and Offline Mode](#online-and-offline-mode) section.
