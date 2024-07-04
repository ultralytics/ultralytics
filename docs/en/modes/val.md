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

### How do I validate a model using Ultralytics YOLOv8?

Validating a model in Ultralytics YOLOv8 is straightforward. You can use either the Python API or the Command-Line Interface (CLI). For the Python API:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # or "path/to/best.pt" for custom models

# Validate the model
metrics = model.val()  # Dataset and settings are remembered from training
print(metrics.box.map)  # mAP50-95
```

For the CLI:

```
yolo val model=yolov8n.pt  # or model=path/to/best.pt for custom models
```

### What metrics are available for model validation in YOLOv8?

Ultralytics YOLOv8 provides several metrics for model validation to ensure comprehensive performance evaluation:

- **mAP50**: Mean Average Precision at IoU threshold 0.50.
- **mAP75**: Mean Average Precision at IoU threshold 0.75.
- **mAP50-95**: Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95 in increments of 0.05.
- **BoxMAP**: List of mAP scores for each category.

These metrics are accessed under the `metrics.box` attribute in both CLI and Python API. For example, `metrics.box.map50` gives the mAP at IoU 0.50.

### Why should I use the Val mode in Ultralytics YOLOv8 for model validation?

Val mode in Ultralytics YOLOv8 is specifically designed for robust and efficient model validation:

- **Precision**: Access detailed metrics including mAP50, mAP75, and mAP50-95 for thorough validation.
- **Convenience**: Built-in features automatically remember training configurations, simplifying subsequent validations.
- **Flexibility**: Validate against the same or different datasets and image sizes.
- **Hyperparameter Tuning**: Use validation metrics to fine-tune your model for improved performance.

Learn more in our [detailed guide](https://docs.ultralytics.com/modes/#val).

### Can I customize the validation settings for Ultralytics YOLOv8?

Yes, you can customize the validation settings by specifying various arguments. Here are some key arguments:

- `data`: Path to the dataset configuration file.
- `imgsz`: Size of the input images.
- `batch`: Number of images per batch.
- `save_json`: Save results to a JSON file for further analysis.
- `conf`: Minimum confidence threshold for detections.
- `iou`: Intersection Over Union threshold for Non-Maximum Suppression.

Example in Python:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
validation_results = model.val(data="coco8.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")
```

More details on each argument can be found in the [Arguments for YOLO Model Validation](../modes/export.md) section.

### What datasets are compatible with Ultralytics YOLOv8 for validation?

Ultralytics YOLOv8 is compatible with a range of datasets, including COCO, VOC, and custom datasets defined in YAML format. The dataset configuration file typically includes paths to validation data, class names, and the number of classes.

For instance, you can use the COCO dataset configuration by setting the `data` argument to `coco.yaml`.

```bash
yolo val model=yolov8n.pt data=coco.yaml
```

Refer to our [datasets documentation](../datasets/index.md) for a comprehensive list and detailed usage instructions.
