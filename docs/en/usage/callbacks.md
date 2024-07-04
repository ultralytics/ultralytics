---
comments: true
description: Explore Ultralytics callbacks for training, validation, exporting, and prediction. Learn how to use and customize them for your ML models.
keywords: Ultralytics, callbacks, training, validation, export, prediction, ML models, YOLOv8, Python, machine learning
---

## Callbacks

Ultralytics framework supports callbacks as entry points in strategic stages of train, val, export, and predict modes. Each callback accepts a `Trainer`, `Validator`, or `Predictor` object depending on the operation type. All properties of these objects can be found in Reference section of the docs.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=67"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Mastering Ultralytics YOLOv8: Callbacks
</p>

## Examples

### Returning additional information with Prediction

In this example, we want to return the original frame with each result object. Here's how we can do that

```python
from ultralytics import YOLO


def on_predict_batch_end(predictor):
    """Handle prediction batch end by combining results with corresponding frames; modifies predictor results."""
    _, image, _, _ = predictor.batch

    # Ensure that image is a list
    image = image if isinstance(image, list) else [image]

    # Combine the prediction results with the corresponding frames
    predictor.results = zip(predictor.results, image)


# Create a YOLO model instance
model = YOLO("yolov8n.pt")

# Add the custom callback to the model
model.add_callback("on_predict_batch_end", on_predict_batch_end)

# Iterate through the results and frames
for result, frame in model.predict():  # or model.track()
    pass
```

## All callbacks

Here are all supported callbacks. See callbacks [source code](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py) for additional details.

### Trainer Callbacks

| Callback                    | Description                                             |
| --------------------------- | ------------------------------------------------------- |
| `on_pretrain_routine_start` | Triggered at the beginning of pre-training routine      |
| `on_pretrain_routine_end`   | Triggered at the end of pre-training routine            |
| `on_train_start`            | Triggered when the training starts                      |
| `on_train_epoch_start`      | Triggered at the start of each training epoch           |
| `on_train_batch_start`      | Triggered at the start of each training batch           |
| `optimizer_step`            | Triggered during the optimizer step                     |
| `on_before_zero_grad`       | Triggered before gradients are zeroed                   |
| `on_train_batch_end`        | Triggered at the end of each training batch             |
| `on_train_epoch_end`        | Triggered at the end of each training epoch             |
| `on_fit_epoch_end`          | Triggered at the end of each fit epoch                  |
| `on_model_save`             | Triggered when the model is saved                       |
| `on_train_end`              | Triggered when the training process ends                |
| `on_params_update`          | Triggered when model parameters are updated             |
| `teardown`                  | Triggered when the training process is being cleaned up |

### Validator Callbacks

| Callback             | Description                                     |
| -------------------- | ----------------------------------------------- |
| `on_val_start`       | Triggered when the validation starts            |
| `on_val_batch_start` | Triggered at the start of each validation batch |
| `on_val_batch_end`   | Triggered at the end of each validation batch   |
| `on_val_end`         | Triggered when the validation ends              |

### Predictor Callbacks

| Callback                     | Description                                       |
| ---------------------------- | ------------------------------------------------- |
| `on_predict_start`           | Triggered when the prediction process starts      |
| `on_predict_batch_start`     | Triggered at the start of each prediction batch   |
| `on_predict_postprocess_end` | Triggered at the end of prediction postprocessing |
| `on_predict_batch_end`       | Triggered at the end of each prediction batch     |
| `on_predict_end`             | Triggered when the prediction process ends        |

### Exporter Callbacks

| Callback          | Description                              |
| ----------------- | ---------------------------------------- |
| `on_export_start` | Triggered when the export process starts |
| `on_export_end`   | Triggered when the export process ends   |

## FAQ

### How do Ultralytics callbacks enhance model training and validation?

Ultralytics callbacks provide strategic entry points at different stages of the training, validation, export, and prediction processes. They work by accepting `Trainer`, `Validator`, or `Predictor` objects to perform specific tasks or modify results. This customization allows users to optimize and extend the functionality of their machine learning models, as demonstrated in the [prediction batch example](#returning-additional-information-with-prediction). For a detailed list of the supported callbacks, refer to the [source code](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py).

### What are practical examples of using Ultralytics callbacks?

Callbacks can be used to add custom functionality during different phases of the machine learning pipeline. A practical example is updating prediction results with additional information. By defining a custom callback, you can zip prediction results with images, as shown in the [Returning additional information with Prediction](#returning-additional-information-with-prediction) example. This flexibility helps you tailor the output to your specific application needs.

```python
from ultralytics import YOLO


def on_predict_batch_end(predictor):
    """Handle prediction batch end by combining results with corresponding frames; modifies predictor results."""
    _, image, _, _ = predictor.batch
    image = image if isinstance(image, list) else [image]
    predictor.results = zip(predictor.results, image)


model = YOLO("yolov8n.pt")
model.add_callback("on_predict_batch_end", on_predict_batch_end)

# Iterate through the results and frames
for result, frame in model.predict():
    pass
```

### Why should I use Ultralytics YOLOv8 for my object detection projects?

Ultralytics YOLOv8 offers state-of-the-art object detection, segmentation, and classification capabilities. It is designed for real-time performance and high accuracy, which makes it suitable for various applications such as security systems, inventory management, and wildlife monitoring. Additionally, the extensive documentation and support for features like callbacks, as detailed in the [training](../modes/train.md) and [predict](../modes/predict.md) modes, make it easy to customize and optimize your models.

### How can I export a YOLOv8 model using Ultralytics?

Exporting a YOLOv8 model can be done using the export mode. Ultralytics supports exporting to various formats including ONNX, TensorRT, and CoreML, among others. This feature ensures compatibility and performance optimization across different platforms. Detailed instructions on how to export your model can be found in the [Export](../modes/export.md) section.

### What unique features do Ultralytics' Validator callbacks offer?

Validator callbacks in Ultralytics are designed to trigger actions at different stages of the validation process, helping to streamline and enhance model evaluation. These callbacks can start at the beginning of validation (`on_val_start`), during each validation batch (`on_val_batch_start`, `on_val_batch_end`), and at the end (`on_val_end`). By utilizing these callbacks, you can monitor performance or capture specific metrics, thereby facilitating a more effective model validation process. For a complete list of validator callbacks, check the [Validator Callbacks](#validator-callbacks) section.
