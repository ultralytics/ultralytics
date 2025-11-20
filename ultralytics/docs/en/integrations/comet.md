---
comments: true
description: Learn to simplify the logging of YOLO11 training with Comet ML. This guide covers installation, setup, real-time insights, and custom logging.
keywords: YOLO11, Comet ML, logging, machine learning, training, model checkpoints, metrics, installation, configuration, real-time insights, custom logging
---

# Elevating YOLO11 Training: Simplify Your Logging Process with Comet ML

Logging key training details such as parameters, metrics, image predictions, and model checkpoints is essential in [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml)—it keeps your project transparent, your progress measurable, and your results repeatable.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/LPodYpvKkvI"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Use Comet ML for Ultralytics YOLO Model Training Logs and Metrics 🚀
</p>

[Ultralytics YOLO11](https://www.ultralytics.com/) seamlessly integrates with Comet ML, efficiently capturing and optimizing every aspect of your YOLO11 [object detection](https://www.ultralytics.com/glossary/object-detection) model's training process. In this guide, we'll cover the installation process, Comet ML setup, real-time insights, custom logging, and offline usage, ensuring that your YOLO11 training is thoroughly documented and fine-tuned for outstanding results.

## Comet ML

<p align="center">
  <img width="640" src="https://www.comet.com/docs/v2/img/landing/home-hero.svg" alt="Comet ML Overview">
</p>

[Comet ML](https://www.comet.com/site/) is a platform for tracking, comparing, explaining, and optimizing machine learning models and experiments. It allows you to log metrics, parameters, media, and more during your model training and monitor your experiments through an aesthetically pleasing web interface. Comet ML helps data scientists iterate more rapidly, enhances transparency and reproducibility, and aids in the development of production models.

## Harnessing the Power of YOLO11 and Comet ML

By combining Ultralytics YOLO11 with Comet ML, you unlock a range of benefits. These include simplified experiment management, real-time insights for quick adjustments, flexible and tailored logging options, and the ability to log experiments offline when internet access is limited. This integration empowers you to make data-driven decisions, analyze performance metrics, and achieve exceptional results.

## Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required packages for YOLO11 and Comet ML
        pip install ultralytics comet_ml torch torchvision
        ```

## Configuring Comet ML

After installing the required packages, you'll need to sign up, get a [Comet API Key](https://www.comet.com/signup), and configure it.

!!! tip "Configuring Comet ML"

    === "CLI"

        ```bash
        # Set your Comet API Key
        export COMET_API_KEY=YOUR_API_KEY
        ```

Then, you can initialize your Comet project. Comet will automatically detect the API key and proceed with the setup.

!!! example "Initialize Comet project"

    === "Python"

        ```python
        import comet_ml

        comet_ml.login(project_name="comet-example-yolo11-coco128")
        ```

If you are using a Google Colab notebook, the code above will prompt you to enter your API key for initialization.

## Usage

Before diving into the usage instructions, be sure to check out the range of [YOLO11 models offered by Ultralytics](../models/yolo11.md). This will help you choose the most appropriate model for your project requirements.

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")

        # Train the model
        results = model.train(
            data="coco8.yaml",
            project="comet-example-yolo11-coco128",
            batch=32,
            save_period=1,
            save_json=True,
            epochs=3,
        )
        ```

After running the training code, Comet ML will create an experiment in your Comet workspace to track the run automatically. You will then be provided with a link to view the detailed logging of your [YOLO11 model's training](../modes/train.md) process.

Comet automatically logs the following data with no additional configuration: metrics such as mAP and loss, hyperparameters, model checkpoints, interactive confusion matrix, and image [bounding box](https://www.ultralytics.com/glossary/bounding-box) predictions.

## Understanding Your Model's Performance with Comet ML Visualizations

Let's dive into what you'll see on the Comet ML dashboard once your YOLO11 model begins training. The dashboard is where all the action happens, presenting a range of automatically logged information through visuals and statistics. Here's a quick tour:

**Experiment Panels**

The experiment panels section of the Comet ML dashboard organizes and presents the different runs and their metrics, such as segment mask loss, class loss, precision, and [mean average precision](https://www.ultralytics.com/glossary/mean-average-precision-map).

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/comet-ml-dashboard-overview.avif" alt="Comet ML Overview">
</p>

**Metrics**

In the metrics section, you have the option to examine the metrics in a tabular format as well, which is displayed in a dedicated pane as illustrated here.

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/comet-ml-metrics-tabular.avif" alt="Comet ML Overview">
</p>

**Interactive [Confusion Matrix](https://www.ultralytics.com/glossary/confusion-matrix)**

The confusion matrix, found in the Confusion Matrix tab, provides an interactive way to assess the model's classification [accuracy](https://www.ultralytics.com/glossary/accuracy). It details the correct and incorrect predictions, allowing you to understand the model's strengths and weaknesses.

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/comet-ml-interactive-confusion-matrix.avif" alt="Comet ML Overview">
</p>

**System Metrics**

Comet ML logs system metrics to help identify any bottlenecks in the training process. It includes metrics such as GPU utilization, GPU memory usage, CPU utilization, and RAM usage. These are essential for monitoring the efficiency of resource usage during model training.

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/comet-ml-system-metrics.avif" alt="Comet ML Overview">
</p>

## Customizing Comet ML Logging

Comet ML offers the flexibility to customize its logging behavior by setting environment variables. These configurations allow you to tailor Comet ML to your specific needs and preferences. Here are some helpful customization options:

### Logging Image Predictions

You can control the number of image predictions that Comet ML logs during your experiments. By default, Comet ML logs 100 image predictions from the validation set. However, you can change this number to better suit your requirements. For example, to log 200 image predictions, use the following code:

```python
import os

os.environ["COMET_MAX_IMAGE_PREDICTIONS"] = "200"
```

### Batch Logging Interval

Comet ML allows you to specify how often batches of image predictions are logged. The `COMET_EVAL_BATCH_LOGGING_INTERVAL` environment variable controls this frequency. The default setting is 1, which logs predictions from every validation batch. You can adjust this value to log predictions at a different interval. For instance, setting it to 4 will log predictions from every fourth batch.

```python
import os

os.environ["COMET_EVAL_BATCH_LOGGING_INTERVAL"] = "4"
```

### Disabling Confusion Matrix Logging

In some cases, you may not want to log the confusion matrix from your validation set after every [epoch](https://www.ultralytics.com/glossary/epoch). You can disable this feature by setting the `COMET_EVAL_LOG_CONFUSION_MATRIX` environment variable to "false." The confusion matrix will only be logged once, after the training is completed.

```python
import os

os.environ["COMET_EVAL_LOG_CONFUSION_MATRIX"] = "false"
```

### Offline Logging

If you find yourself in a situation where internet access is limited, Comet ML provides an offline logging option. You can set the `COMET_MODE` environment variable to "offline" to enable this feature. Your experiment data will be saved locally in a directory that you can later upload to Comet ML when internet connectivity is available.

```python
import os

os.environ["COMET_MODE"] = "offline"
```

## Summary

This guide has walked you through integrating Comet ML with Ultralytics' YOLO11. From installation to customization, you've learned to streamline experiment management, gain real-time insights, and adapt logging to your project's needs.

Explore [Comet ML's official YOLOv8 integration documentation](https://www.comet.com/docs/v2/integrations/third-party-tools/yolov8/), which also applies to YOLO11 projects.

Furthermore, if you're looking to dive deeper into the practical applications of YOLO11, specifically for [image segmentation](https://www.ultralytics.com/glossary/image-segmentation) tasks, this detailed guide on [fine-tuning YOLO11 with Comet ML](https://www.comet.com/site/blog/fine-tuning-yolov8-for-image-segmentation-with-comet/) offers valuable insights and step-by-step instructions to enhance your model's performance.

Additionally, to explore other exciting integrations with Ultralytics, check out the [integration guide page](../integrations/index.md), which offers a wealth of resources and information.

## FAQ

### How do I integrate Comet ML with Ultralytics YOLO11 for training?

To integrate Comet ML with Ultralytics YOLO11, follow these steps:

1. **Install the required packages**:

    ```bash
    pip install ultralytics comet_ml torch torchvision
    ```

2. **Set up your Comet API Key**:

    ```bash
    export COMET_API_KEY=YOUR_API_KEY
    ```

3. **Initialize your Comet project in your Python code**:

    ```python
    import comet_ml

    comet_ml.login(project_name="comet-example-yolo11-coco128")
    ```

4. **Train your YOLO11 model and log metrics**:

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    results = model.train(
        data="coco8.yaml",
        project="comet-example-yolo11-coco128",
        batch=32,
        save_period=1,
        save_json=True,
        epochs=3,
    )
    ```

For more detailed instructions, refer to the [Comet ML configuration section](#configuring-comet-ml).

### What are the benefits of using Comet ML with YOLO11?

By integrating Ultralytics YOLO11 with Comet ML, you can:

- **Monitor real-time insights**: Get instant feedback on your training results, allowing for quick adjustments.
- **Log extensive metrics**: Automatically capture essential metrics such as mAP, loss, hyperparameters, and model checkpoints.
- **Track experiments offline**: Log your training runs locally when internet access is unavailable.
- **Compare different training runs**: Use the interactive Comet ML dashboard to analyze and compare multiple experiments.

By leveraging these features, you can optimize your machine learning workflows for better performance and reproducibility. For more information, visit the [Comet ML integration guide](../integrations/index.md).

### How do I customize the logging behavior of Comet ML during YOLO11 training?

Comet ML allows for extensive customization of its logging behavior using environment variables:

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

- **Disable confusion matrix logging**:

    ```python
    import os

    os.environ["COMET_EVAL_LOG_CONFUSION_MATRIX"] = "false"
    ```

Refer to the [Customizing Comet ML Logging](#customizing-comet-ml-logging) section for more customization options.

### How do I view detailed metrics and visualizations of my YOLO11 training on Comet ML?

Once your YOLO11 model starts training, you can access a wide range of metrics and visualizations on the Comet ML dashboard. Key features include:

- **Experiment Panels**: View different runs and their metrics, including segment mask loss, class loss, and mean average [precision](https://www.ultralytics.com/glossary/precision).
- **Metrics**: Examine metrics in tabular format for detailed analysis.
- **Interactive Confusion Matrix**: Assess classification accuracy with an interactive confusion matrix.
- **System Metrics**: Monitor GPU and CPU utilization, memory usage, and other system metrics.

For a detailed overview of these features, visit the [Understanding Your Model's Performance with Comet ML Visualizations](#understanding-your-models-performance-with-comet-ml-visualizations) section.

### Can I use Comet ML for offline logging when training YOLO11 models?

Yes, you can enable offline logging in Comet ML by setting the `COMET_MODE` environment variable to "offline":

```python
import os

os.environ["COMET_MODE"] = "offline"
```

This feature allows you to log your experiment data locally, which can later be uploaded to Comet ML when internet connectivity is available. This is particularly useful when working in environments with limited internet access. For more details, refer to the [Offline Logging](#offline-logging) section.
