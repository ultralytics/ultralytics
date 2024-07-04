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

### What are Ultralytics callbacks and how do they enhance the model training process?

Ultralytics callbacks are functions that are invoked at specific stages during the model training, validation, prediction, and exporting processes. They enhance the model training experience by allowing users to execute custom code at various points in the workflow. For example, the `on_train_start` callback is triggered when training starts, providing an opportunity to initialize resources or log the start of training. For a complete list of all supported callbacks, refer to the [callbacks source code](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py).

### How can I customize prediction results using Ultralytics YOLOv8?

You can customize prediction results in Ultralytics YOLOv8 by defining and adding custom callbacks. For instance, to return the original frame with each result object during prediction, you can use the `on_predict_batch_end` callback. Here’s an example:

```python
from ultralytics import YOLO


def on_predict_batch_end(predictor):
    _, image, _, _ = predictor.batch
    image = image if isinstance(image, list) else [image]
    predictor.results = zip(predictor.results, image)


model = YOLO("yolov8n.pt")
model.add_callback("on_predict_batch_end", on_predict_batch_end)

for result, frame in model.predict():
    pass
```

This code snippet shows how you can modify the predictor results to include both the prediction outputs and the original input frames.

### Why should I use Ultralytics YOLO for my machine learning projects?

Ultralytics YOLO (You Only Look Once) is a state-of-the-art real-time object detection and image segmentation model that offers high speed and accuracy. It is highly versatile and supports numerous tasks including detection, segmentation, pose estimation, and more. Additionally, Ultralytics provides extensive documentation, user-friendly APIs, and comprehensive support for deploying models across various platforms. Whether you are a beginner or an advanced user, Ultralytics YOLO enhances your machine learning projects with its robust features and easy integration.

### What are the key differences between Trainer, Validator, and Predictor objects in Ultralytics?

In Ultralytics, Trainer, Validator, and Predictor objects are tailored for different stages of the machine learning workflow.

- **Trainer**: Used during the training phase, it handles the loading of datasets, model training, and logging of metrics.
- **Validator**: Utilized during the validation phase, it evaluates the model’s performance using a validation dataset and calculates metrics such as accuracy and loss.
- **Predictor**: Employed in the prediction phase, it takes new input data, runs the model inference, and processes the prediction results.

Each of these objects supports distinct callbacks, making it easier to customize and control the behavior at each stage of the ML pipeline. For more details, you can refer to the [Reference section](../reference).

### How do I incorporate Ultralytics HUB into my project?

Ultralytics HUB provides a seamless, no-code environment for generating, training, and deploying AI models, including YOLOv8. By using Ultralytics HUB, you can quickly set up your models, manage datasets, and monitor training progress in a user-friendly interface. To get started, visit the [Ultralytics HUB](https://www.ultralytics.com/hub) and follow the quickstart guide to upload datasets, train YOLO models, and manage projects easily. With its intuitive platform, Ultralytics HUB allows users of all skill levels to leverage advanced AI technology without extensive coding knowledge.
