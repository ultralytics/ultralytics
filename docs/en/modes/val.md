---
comments: true
description: Guide for Validating YOLOv8 Models. Learn how to evaluate the performance of your YOLO models using validation settings and metrics with Python and CLI examples.
keywords: Ultralytics, YOLO Docs, YOLOv8, validation, model evaluation, hyperparameters, accuracy, metrics, Python, CLI
---

# Model Validation with Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

Validation is a critical step in the machine learning pipeline, allowing you to assess the quality of your trained models. Val mode in Ultralytics YOLOv8 provides a robust suite of tools and metrics for evaluating the performance of your object detection models. This guide serves as a complete resource for understanding how to effectively use the Val mode to ensure that your models are both accurate and reliable.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?start=47"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics Modes Tutorial: Validation
</p>

## Why Validate with Ultralytics YOLO?

Here's why using YOLOv8's Val mode is advantageous:

- **Precision:** Get accurate metrics like mAP50, mAP75, and mAP50-95 to comprehensively evaluate your model.
- **Convenience:** Utilize built-in features that remember training settings, simplifying the validation process.
- **Flexibility:** Validate your model with the same or different datasets and image sizes.
- **Hyperparameter Tuning:** Use validation metrics to fine-tune your model for better performance.

### Key Features of Val Mode

These are the notable functionalities offered by YOLOv8's Val mode:

- **Automated Settings:** Models remember their training configurations for straightforward validation.
- **Multi-Metric Support:** Evaluate your model based on a range of accuracy metrics.
- **CLI and Python API:** Choose from command-line interface or Python API based on your preference for validation.
- **Data Compatibility:** Works seamlessly with datasets used during the training phase as well as custom datasets.

!!! Tip "Tip"

    * YOLOv8 models automatically remember their training settings, so you can validate a model at the same image size and on the original dataset easily with just `yolo val model=yolov8n.pt` or `model('yolov8n.pt').val()`

## Usage Examples

Validate trained YOLOv8n model accuracy on the COCO128 dataset. No argument need to passed as the `model` retains it's training `data` and arguments as model attributes. See Arguments section below for a full list of export arguments.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.pt')  # load an official model
        model = YOLO('path/to/best.pt')  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # a list contains map50-95 of each category
        ```

    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt  # val official model
        yolo detect val model=path/to/best.pt  # val custom model
        ```

## Arguments

The table below outlines the various settings and hyperparameters you can adjust to evaluate the performance of YOLO models. These settings influence aspects such as model accuracy, detection speed, and resource allocation during the validation phase. Understanding and fine-tuning these parameters is crucial for optimizing model performance on your specific dataset.

| Key           | Default Value | Description                                                                                                                                                                                                                                   |
|---------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `data`        | `None`        | Specifies the path to the data configuration file (e.g., `coco128.yaml`). This file contains information about the dataset, including paths to validation images and class labels.                                                            |
| `imgsz`       | `640`         | Sets the input image size in pixels (`height`=`width`) for validation. Resizing images to a standard size can impact both the speed and accuracy of the model.                                                                                |
| `batch`       | `16`          | Determines the number of images processed in a single batch. A larger batch size can speed up the validation process but requires more memory. The special value `-1` triggers AutoBatch, optimizing batch size based on available resources. |
| `save_json`   | `False`       | If set to `True`, saves the validation results in a JSON file for further analysis or external use. Useful for benchmarking or sharing results.                                                                                               |
| `save_hybrid` | `False`       | When set to `True`, saves a hybrid format of labels combining original dataset labels with additional predictions made by the model. This can be helpful for analyzing model performance in detail.                                           |
| `conf`        | `0.001`       | Sets the minimum confidence threshold for detections to be considered valid. Detections with confidence scores below this threshold are discarded. Adjusting this value can affect the precision and recall of the model.                     |
| `iou`         | `0.6`         | Specifies the Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). NMS is a technique to eliminate redundant detections, keeping only the most confident ones.                                                          |
| `max_det`     | `300`         | Limits the maximum number of detections allowed per image. This constraint can prevent the model from generating too many predictions on dense scenes.                                                                                        |
| `half`        | `True`        | Enables half-precision (FP16) computation, which can significantly speed up validation on compatible hardware (e.g., recent NVIDIA GPUs) with minimal impact on accuracy.                                                                     |
| `device`      | `None`        | Selects the computational device for validation (`cpu` or `cuda` along with device number like `0`, `1`, etc.). If not specified, automatically uses the best available device.                                                               |
| `dnn`         | `False`       | When enabled, uses OpenCV's Deep Neural Network (DNN) module for inference with ONNX models. This can be a fallback option for environments where CUDA is not available.                                                                      |
| `plots`       | `False`       | If set to `True`, generates and saves plots and images during validation. This visual feedback can be invaluable for understanding model performance and issues.                                                                              |
| `rect`        | `False`       | Enables rectangular validation, which adjusts input images to a rectangular shape that fits the original aspect ratio with minimal padding. This can improve speed and accuracy by reducing unnecessary computations.                         |
| `split`       | `val`         | Defines the dataset split to use for validation (`val`, `test`, or `train`). This allows flexibility in choosing different parts of the dataset to evaluate model performance.                                                                |

Each of these settings plays a vital role in the validation process, allowing for a customizable and efficient evaluation of YOLO models. Adjusting these parameters according to your specific needs and resources can help achieve the best balance between accuracy and performance.
