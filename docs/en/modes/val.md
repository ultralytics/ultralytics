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

## Arguments for YOLO Model Validation

When validating YOLO models, several arguments can be fine-tuned to optimize the evaluation process. These arguments control aspects such as input image size, batch processing, and performance thresholds. Below is a detailed breakdown of each argument to help you customize your validation settings effectively.

| Key           | Default Value | Description                                                                                                                                                                                   |
|---------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `data`        | `None`        | The path to the dataset configuration file (e.g., `coco128.yaml`). This file specifies the dataset's structure, including the classes, train, and validation set paths.                       |
| `imgsz`       | `640`         | The input image size as an integer. This size is used to resize images during validation, impacting detection accuracy and inference speed.                                                   |
| `batch`       | `16`          | The number of images processed in each batch. A larger batch size can speed up validation but requires more memory. Use `-1` for AutoBatch to automatically adjust based on available memory. |
| `save_json`   | `False`       | If set to `True`, validation results are saved in a JSON format, useful for further analysis or submission to evaluation servers.                                                             |
| `save_hybrid` | `False`       | When `True`, saves a hybrid version of labels combining ground truth with model predictions. This can be useful for visualizing model performance or training enhancements.                   |
| `conf`        | `0.001`       | The minimum confidence threshold for considering detections. Increasing this value may reduce false positives but could also miss less confident detections.                                  |
| `iou`         | `0.6`         | The Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Higher values result in fewer detections by eliminating more overlapping boxes.                                |
| `max_det`     | `300`         | The maximum number of detections allowed per image. Useful for limiting outputs in images with many objects.                                                                                  |
| `half`        | `True`        | Enables half precision (FP16) to speed up validation on compatible hardware without significantly affecting accuracy.                                                                         |
| `device`      | `None`        | Specifies the computation device, such as a specific GPU (`cuda:0`) or CPU (`cpu`). This setting allows for model validation on different hardware configurations.                            |
| `dnn`         | `False`       | If `True`, uses OpenCV's DNN module for ONNX model inference. This option can be beneficial for environments where CUDA is unavailable.                                                       |
| `plots`       | `False`       | Enables the generation of plots and saved images during validation, providing visual insights into model performance.                                                                         |
| `rect`        | `False`       | Applies rectangular inference, minimizing padding by processing images in their original aspect ratio. This can improve accuracy and speed but may require more memory.                       |
| `split`       | `val`         | Defines the dataset split to use for validation (e.g., 'val', 'test', 'train'). This allows for flexible validation across different parts of the dataset.                                    |

Each of these settings plays a vital role in the validation process, allowing for a customizable and efficient evaluation of YOLO models. Adjusting these parameters according to your specific needs and resources can help achieve the best balance between accuracy and performance.

### Example Validation with Arguments

The below examples showcase YOLO model validation with custom arguments in Python and CLI.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO('yolov8n.pt')
        
        # Customize validation settings
        validation_results = model.val(data='coco8.yaml',
                                       imgsz=640,
                                       batch=16,
                                       conf=0.25,
                                       iou=0.6,
                                       device='0')
        ```

    === "CLI"

        ```bash
        yolo val model=yolov8n.pt data=coco8.yaml imgsz=640 batch=16 conf=0.25 iou=0.6 device=0
        ```
