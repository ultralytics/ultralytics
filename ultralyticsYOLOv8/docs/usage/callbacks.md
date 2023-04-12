## Callbacks

Ultralytics framework supports callbacks as entry points in strategic stages of train, val, export, and predict modes.
Each callback accepts a `Trainer`, `Validator`, or `Predictor` object depending on the operation type. All properties of
these objects can be found in Reference section of the docs.

## Examples

### Returning additional information with Prediction

In this example, we want to return the original frame with each result object. Here's how we can do that

```python
def on_predict_batch_end(predictor):
    # Retrieve the batch data
    _, _, im0s, _, _ = predictor.batch
    
    # Ensure that im0s is a list
    im0s = im0s if isinstance(im0s, list) else [im0s]
    
    # Combine the prediction results with the corresponding frames
    predictor.results = zip(predictor.results, im0s)

# Create a YOLO model instance
model = YOLO(f'yolov8n.pt')

# Add the custom callback to the model
model.add_callback("on_predict_batch_end", on_predict_batch_end)

# Iterate through the results and frames
for (result, frame) in model.track/predict():
    pass
```

## All callbacks

Here are all supported callbacks. See callbacks [source code](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/utils/callbacks/base.py) for additional details.


### Trainer Callbacks

| Callback                    | Description                                             |
|-----------------------------|---------------------------------------------------------|
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
|----------------------|-------------------------------------------------|
| `on_val_start`       | Triggered when the validation starts            |
| `on_val_batch_start` | Triggered at the start of each validation batch |
| `on_val_batch_end`   | Triggered at the end of each validation batch   |
| `on_val_end`         | Triggered when the validation ends              |


### Predictor Callbacks

| Callback                     | Description                                       |
|------------------------------|---------------------------------------------------|
| `on_predict_start`           | Triggered when the prediction process starts      |
| `on_predict_batch_start`     | Triggered at the start of each prediction batch   |
| `on_predict_postprocess_end` | Triggered at the end of prediction postprocessing |
| `on_predict_batch_end`       | Triggered at the end of each prediction batch     |
| `on_predict_end`             | Triggered when the prediction process ends        |

### Exporter Callbacks

| Callback          | Description                              |
|-------------------|------------------------------------------|
| `on_export_start` | Triggered when the export process starts |
| `on_export_end`   | Triggered when the export process ends   |
