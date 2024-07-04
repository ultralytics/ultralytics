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

### What are callbacks in the Ultralytics framework, and how can they be used?

**Callbacks** in the Ultralytics framework are entry points at strategic stages of various operations like training, validation, exporting, and prediction. These callbacks accept a `Trainer`, `Validator`, or `Predictor` object based on the mode of operation. For instance, a callback can be used to log information at the end of each training batch or modify prediction results after each prediction batch.

For a detailed guide on how to add custom callbacks, check out the [callbacks documentation](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py).

### How do I return additional information with prediction callbacks in Ultralytics YOLOv8?

To return additional information with prediction results in Ultralytics YOLOv8, you can utilize the `on_predict_batch_end` callback. For example, to return the original frame with each result object:

```python
from ultralytics import YOLO

def on_predict_batch_end(predictor):
    """Handle prediction batch end by combining results with corresponding frames; modifies predictor results."""
    _, image, _, _ = predictor.batch
    image = image if isinstance(image, list) else [image]
    predictor.results = zip(predictor.results, image)

model = YOLO("yolov8n.pt")
model.add_callback("on_predict_batch_end", on_predict_batch_end)

for result, frame in model.predict():  # or model.track()
    pass
```

This approach ensures that the original images are paired with their prediction results, making post-process analysis more straightforward.

### Why should I use Ultralytics callbacks for my ML models?

Using Ultralytics **callbacks** offers numerous advantages:

1. **Customization:** Tailor different stages like training, validation, prediction, and export to fit your specific requirements.
2. **Monitoring:** Track and log training metrics, validation accuracy, and other performance indicators seamlessly.
3. **Automation:** Automate repetitive tasks such as saving models or adjusting hyperparameters dynamically.

Explore more about how to integrate and use callbacks in the [Callbacks Usage Guide](https://docs.ultralytics.com/usage/callbacks/).

### Can Ultralytics callbacks be used for custom training workflows?

Yes, **Ultralytics callbacks** can be seamlessly integrated into custom training workflows. By adding custom callbacks at various stages (e.g., start and end of training epochs, before and after each batch), you can manipulate the training process to suit specific needs, such as dynamic learning rate adjustments or custom logging.

For more details, refer to the complete list of [Trainer Callbacks](#trainer-callbacks) and how they can be utilized.

### What types of operations support callbacks in the Ultralytics framework?

The Ultralytics framework supports callbacks for the following operations:
- **Training:** e.g., `on_train_start`, `on_train_epoch_end`
- **Validation:** e.g., `on_val_start`, `on_val_end`
- **Prediction:** e.g., `on_predict_start`, `on_predict_batch_end`
- **Export:** e.g., `on_export_start`, `on_export_end`

Each operation type accepts specific objects (`Trainer`, `Validator`, `Predictor`, `Exporter`). For a full list, refer to the [All Callbacks](#all-callbacks) section in the documentation.