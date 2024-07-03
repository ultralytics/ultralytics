---
comments: true
description: Learn how to validate your YOLOv8 model with precise metrics, easy-to-use tools, and custom settings for optimal performance.
keywords: Ultralytics, YOLOv8, model validation, machine learning, object detection, mAP metrics, Python API, CLI
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

Validate trained YOLOv8n model accuracy on the COCO8 dataset. No argument need to passed as the `model` retains its training `data` and arguments as model attributes. See Arguments section below for a full list of export arguments.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # a list contains map50-95 of each category
        ```

    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt  # val official model
        yolo detect val model=path/to/best.pt  # val custom model
        ```

## Arguments for YOLO Model Validation

When validating YOLO models, several arguments can be fine-tuned to optimize the evaluation process. These arguments control aspects such as input image size, batch processing, and performance thresholds. Below is a detailed breakdown of each argument to help you customize your validation settings effectively.

| Argument      | Type    | Default | Description                                                                                                                                                 |
| ------------- | ------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data`        | `str`   | `None`  | Specifies the path to the dataset configuration file (e.g., `coco8.yaml`). This file includes paths to validation data, class names, and number of classes. |
| `imgsz`       | `int`   | `640`   | Defines the size of input images. All images are resized to this dimension before processing.                                                               |
| `batch`       | `int`   | `16`    | Sets the number of images per batch. Use `-1` for AutoBatch, which automatically adjusts based on GPU memory availability.                                  |
| `save_json`   | `bool`  | `False` | If `True`, saves the results to a JSON file for further analysis or integration with other tools.                                                           |
| `save_hybrid` | `bool`  | `False` | If `True`, saves a hybrid version of labels that combines original annotations with additional model predictions.                                           |
| `conf`        | `float` | `0.001` | Sets the minimum confidence threshold for detections. Detections with confidence below this threshold are discarded.                                        |
| `iou`         | `float` | `0.6`   | Sets the Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Helps in reducing duplicate detections.                                 |
| `max_det`     | `int`   | `300`   | Limits the maximum number of detections per image. Useful in dense scenes to prevent excessive detections.                                                  |
| `half`        | `bool`  | `True`  | Enables half-precision (FP16) computation, reducing memory usage and potentially increasing speed with minimal impact on accuracy.                          |
| `device`      | `str`   | `None`  | Specifies the device for validation (`cpu`, `cuda:0`, etc.). Allows flexibility in utilizing CPU or GPU resources.                                          |
| `dnn`         | `bool`  | `False` | If `True`, uses the OpenCV DNN module for ONNX model inference, offering an alternative to PyTorch inference methods.                                       |
| `plots`       | `bool`  | `False` | When set to `True`, generates and saves plots of predictions versus ground truth for visual evaluation of the model's performance.                          |
| `rect`        | `bool`  | `False` | If `True`, uses rectangular inference for batching, reducing padding and potentially increasing speed and efficiency.                                       |
| `split`       | `str`   | `val`   | Determines the dataset split to use for validation (`val`, `test`, or `train`). Allows flexibility in choosing the data segment for performance evaluation. |

Each of these settings plays a vital role in the validation process, allowing for a customizable and efficient evaluation of YOLO models. Adjusting these parameters according to your specific needs and resources can help achieve the best balance between accuracy and performance.

### Example Validation with Arguments

The below examples showcase YOLO model validation with custom arguments in Python and CLI.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")

        # Customize validation settings
        validation_results = model.val(data="coco8.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")
        ```

    === "CLI"

        ```bash
        yolo val model=yolov8n.pt data=coco8.yaml imgsz=640 batch=16 conf=0.25 iou=0.6 device=0
        ```

## FAQ

### How do I validate my YOLOv8 model using the CLI?

To validate your YOLOv8 model using the command-line interface (CLI), you can use the `yolo val` command. This tool simplifies the validation process by remembering the training settings. Here's a basic example:

```bash
yolo val model=yolov8n.pt
```

This command validates a pre-trained "yolov8n.pt" model. If you have a custom model, replace "yolov8n.pt" with the path to your model. For more details, refer to the [ClI usage documentation](../usage/cli.md).

### What are the key features of the Val mode in YOLOv8?

YOLOv8's Val mode offers numerous features to ensure accurate and comprehensive model evaluation:

- **Automated Settings:** Models remember their training configurations.
- **Multi-Metric Support:** Evaluate using metrics like mAP50, mAP75, and mAP50-95.
- **CLI and Python API:** Choose between command-line or Python-based validation.
- **Data Compatibility:** Works with both original and custom datasets.

Explore the comprehensive features in the [Val Mode guide](../modes/val.md).

### Why should I validate my YOLOv8 model?

Model validation is critical to assess the quality and reliability of your YOLOv8 models. Validation provides:

- **Precision:** Get metrics such as mAP50-95 for a detailed performance evaluation.
- **Convenience:** Use built-in features that recall training settings.
- **Flexibility:** Validate across multiple datasets and image sizes.
- **Hyperparameter Tuning:** Utilize metrics to fine-tune your model for better accuracy.

Learn more about the importance of validation in our [Model Validation guide](../modes/val.md).

### How can I customize the validation arguments for YOLOv8?

YOLOv8 allows customization of the validation process through various arguments. Here’s an example in Python:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

# Customize validation settings
validation_results = model.val(data="coco8.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")
```

Similarly, for CLI:

```bash
yolo val model=yolov8n.pt data=coco8.yaml imgsz=640 batch=16 conf=0.25 iou=0.6 device=0
```

Explore the full list of validation arguments and their descriptions in the [Validation Arguments section](#arguments-for-yolo-model-validation).

### Can I use my custom dataset for model validation in YOLOv8?

Yes, YOLOv8 supports validation using custom datasets. You can specify the path to your dataset configuration file with the `data` argument. Here's an example:

```python
validation_results = model.val(data="path/to/your_custom_dataset.yaml")
```

Ensure your dataset configuration file includes paths to validation data, class names, and the number of classes. For further details, see the [datasets section](../datasets/index.md).
