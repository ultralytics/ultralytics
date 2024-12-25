---
comments: true
description: Explore Ultralytics callbacks for training, validation, exporting, and prediction. Learn how to use and customize them for your ML models.
keywords: Ultralytics, callbacks, training, validation, export, prediction, ML models, YOLO11, Python, machine learning
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
  <strong>Watch:</strong> Mastering Ultralytics YOLO: Callbacks
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
model = YOLO("yolo11n.pt")

# Add the custom callback to the model
model.add_callback("on_predict_batch_end", on_predict_batch_end)

# Iterate through the results and frames
for result, frame in model.predict():  # or model.track()
    pass
```

## All callbacks

Here are all supported callbacks. See callbacks [source code](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py) for additional details.

### Trainer Callbacks

| Callback                    | Description                                                                                 |
| --------------------------- | ------------------------------------------------------------------------------------------- |
| `on_pretrain_routine_start` | Triggered at the beginning of pre-training routine                                          |
| `on_pretrain_routine_end`   | Triggered at the end of pre-training routine                                                |
| `on_train_start`            | Triggered when the training starts                                                          |
| `on_train_epoch_start`      | Triggered at the start of each training [epoch](https://www.ultralytics.com/glossary/epoch) |
| `on_train_batch_start`      | Triggered at the start of each training batch                                               |
| `optimizer_step`            | Triggered during the optimizer step                                                         |
| `on_before_zero_grad`       | Triggered before gradients are zeroed                                                       |
| `on_train_batch_end`        | Triggered at the end of each training batch                                                 |
| `on_train_epoch_end`        | Triggered at the end of each training epoch                                                 |
| `on_fit_epoch_end`          | Triggered at the end of each fit epoch                                                      |
| `on_model_save`             | Triggered when the model is saved                                                           |
| `on_train_end`              | Triggered when the training process ends                                                    |
| `on_params_update`          | Triggered when model parameters are updated                                                 |
| `teardown`                  | Triggered when the training process is being cleaned up                                     |

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

### What are Ultralytics callbacks and how can I use them?

**Ultralytics callbacks** are specialized entry points triggered during key stages of model operations like training, validation, exporting, and prediction. These callbacks allow for custom functionality at specific points in the process, enabling enhancements and modifications to the workflow. Each callback accepts a `Trainer`, `Validator`, or `Predictor` object, depending on the operation type. For detailed properties of these objects, refer to the [Reference section](../reference/cfg/__init__.md).

To use a callback, you can define a function and then add it to the model with the `add_callback` method. Here's an example of how to return additional information during prediction:

```python
from ultralytics import YOLO


def on_predict_batch_end(predictor):
    """Handle prediction batch end by combining results with corresponding frames; modifies predictor results."""
    _, image, _, _ = predictor.batch
    image = image if isinstance(image, list) else [image]
    predictor.results = zip(predictor.results, image)


model = YOLO("yolo11n.pt")
model.add_callback("on_predict_batch_end", on_predict_batch_end)
for result, frame in model.predict():
    pass
```

### How can I customize Ultralytics training routine using callbacks?

To customize your Ultralytics training routine using callbacks, you can inject your logic at specific stages of the training process. Ultralytics YOLO provides a variety of training callbacks such as `on_train_start`, `on_train_end`, and `on_train_batch_end`. These allow you to add custom metrics, processing, or logging.

Here's an example of how to freeze BatchNorm statistics when freezing layers with callbacks:

```python
from ultralytics import YOLO


# Add a callback to put the frozen layers in eval mode to prevent BN values from changing
def put_in_eval_mode(trainer):
    n_layers = trainer.args.freeze
    if not isinstance(n_layers, int):
        return

    for i, (name, module) in enumerate(trainer.model.named_modules()):
        if name.endswith("bn") and int(name.split(".")[1]) < n_layers:
            module.eval()
            module.track_running_stats = False


model = YOLO("yolo11n.pt")
model.add_callback("on_train_epoch_start", put_in_eval_mode)
model.train(data="coco.yaml", epochs=10)
```

Refer to the [Training Guide](../modes/train.md) for more details on how to effectively use training callbacks.

### Why should I use callbacks during validation in Ultralytics YOLO?

Using **callbacks during validation** in Ultralytics YOLO can enhance model evaluation by allowing custom processing, logging, or metrics calculation. Callbacks such as `on_val_start`, `on_val_batch_end`, and `on_val_end` provide entry points to inject custom logic, ensuring detailed and comprehensive validation processes.

For instance, you might want to plot all the validation batches, instead of just the first 3. Here's how you can do that:

```python
import inspect

from ultralytics import YOLO


def plot_samples(validator):
    frame = inspect.currentframe().f_back.f_back
    v = frame.f_locals
    validator.plot_val_samples(v["batch"], v["batch_i"])
    validator.plot_predictions(v["batch"], v["preds"], v["batch_i"])


model = YOLO("yolo11n.pt")
model.add_callback("on_val_batch_end", plot_samples)
model.val(data="coco.yaml")
```

Check out the [Validation Guide](../modes/val.md) for further insights on incorporating callbacks into your validation process.

### How do I attach a custom callback for the prediction mode in Ultralytics YOLO?

To attach a custom callback for the **prediction mode** in Ultralytics YOLO, you define a callback function and register it with the prediction process. Common prediction callbacks include `on_predict_start`, `on_predict_batch_end`, and `on_predict_end`. These allow for modification of prediction outputs and integration of additional functionalities like data logging or result transformation.

Here is an example where a custom callback is used to save predictions based on whether an object of a particular class is present:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

class_id = 2


def save_on_object(predictor):
    r = predictor.results[0]
    if class_id in r.boxes.cls:
        predictor.args.save = True
    else:
        predictor.args.save = False


model.add_callback("on_predict_postprocess_end", save_on_object)
results = model("pedestrians.mp4", stream=True, save=True)

for results in results:
    pass
```

For more comprehensive usage, refer to the [Prediction Guide](../modes/predict.md) which includes detailed instructions and additional customization options.

### What are some practical examples of using callbacks in Ultralytics YOLO?

Ultralytics YOLO supports various practical implementations of callbacks to enhance and customize different phases like training, validation, and prediction. Some practical examples include:

1. **Logging Custom Metrics**: Log additional metrics at different stages, such as the end of training or validation epochs.
2. **[Data Augmentation](https://www.ultralytics.com/glossary/data-augmentation)**: Implement custom data transformations or augmentations during prediction or training batches.
3. **Intermediate Results**: Save intermediate results such as predictions or frames for further analysis or visualization.

Example: Combining frames with prediction results during prediction using `on_predict_batch_end`:

```python
from ultralytics import YOLO


def on_predict_batch_end(predictor):
    """Combine prediction results with frames."""
    _, image, _, _ = predictor.batch
    image = image if isinstance(image, list) else [image]
    predictor.results = zip(predictor.results, image)


model = YOLO("yolo11n.pt")
model.add_callback("on_predict_batch_end", on_predict_batch_end)
for result, frame in model.predict():
    pass
```

Explore the [Complete Callback Reference](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py) to find more options and examples.
