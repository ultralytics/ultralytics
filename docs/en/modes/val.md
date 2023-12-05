---
comments: true
description: Guide for Validating YOLOv8 Models. Learn how to evaluate the performance of your YOLO models using validation settings and metrics with Python and CLI examples.
keywords: Ultralytics, YOLO Docs, YOLOv8, validation, model evaluation, hyperparameters, accuracy, metrics, Python, CLI
---

# Model Validation with Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

Validation is a critical step in the machine learning pipeline, allowing you to assess the quality of your trained models. Val mode in Ultralytics YOLOv8 provides a robust suite of tools and metrics for evaluating the performance of your object detection models. This guide serves as a complete resource for understanding how to effectively use the Val mode to ensure that your models are both accurate and reliable.

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

Validation settings for YOLO models refer to the various hyperparameters and configurations used to evaluate the model's performance on a validation dataset. These settings can affect the model's performance, speed, and accuracy. Some common YOLO validation settings include the batch size, the frequency with which validation is performed during training, and the metrics used to evaluate the model's performance. Other factors that may affect the validation process include the size and composition of the validation dataset and the specific task the model is being used for. It is important to carefully tune and experiment with these settings to ensure that the model is performing well on the validation dataset and to detect and prevent overfitting.

| Key           | Value   | Description                                                        |
|---------------|---------|--------------------------------------------------------------------|
| `data`        | `None`  | path to data file, i.e. coco128.yaml                               |
| `imgsz`       | `640`   | size of input images as integer                                    |
| `batch`       | `16`    | number of images per batch (-1 for AutoBatch)                      |
| `save_json`   | `False` | save results to JSON file                                          |
| `save_hybrid` | `False` | save hybrid version of labels (labels + additional predictions)    |
| `conf`        | `0.001` | object confidence threshold for detection                          |
| `iou`         | `0.6`   | intersection over union (IoU) threshold for NMS                    |
| `max_det`     | `300`   | maximum number of detections per image                             |
| `half`        | `True`  | use half precision (FP16)                                          |
| `device`      | `None`  | device to run on, i.e. cuda device=0/1/2/3 or device=cpu           |
| `dnn`         | `False` | use OpenCV DNN for ONNX inference                                  |
| `plots`       | `False` | save plots and images during train/val                             |
| `rect`        | `False` | rectangular val with each batch collated for minimum padding       |
| `split`       | `val`   | dataset split to use for validation, i.e. 'val', 'test' or 'train' |
