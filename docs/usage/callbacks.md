## Callbacks

Ultralytics framework supports callbacks as entry points in strategic stages of train, val, export, and predict modes.
Each callback accepts a `Trainer`, `Validator`, or `Predictor` object depending on the operation type. All properties of
these objects can be found in Reference section of the docs.

## Examples

### Returning additional information with Prediction

In this example, we want to return the original frame with each result object. Here's how we can do that

```python
def on_predict_batch_end(predictor):
    # results -> List[batch_size]
    _, _, im0s, _, _ = predictor.batch
    im0s = im0s if isinstance(im0s, list) else [im0s]
    predictor.results = zip(predictor.results, im0s)

model = YOLO(f'yolov8n.pt')
model.add_callback("on_predict_batch_end", on_predict_batch_end)
for (result, frame) in model.track/predict():
    pass
```

## All callbacks

Here are all supported callbacks.

### Trainer

`on_pretrain_routine_start`

`on_pretrain_routine_end`

`on_train_start`

`on_train_epoch_start`

`on_train_batch_start`

`optimizer_step`

`on_before_zero_grad`

`on_train_batch_end`

`on_train_epoch_end`

`on_fit_epoch_end`

`on_model_save`

`on_train_end`

`on_params_update`

`teardown`

### Validator

`on_val_start`

`on_val_batch_start`

`on_val_batch_end`

`on_val_end`

### Predictor

`on_predict_start`

`on_predict_batch_start`

`on_predict_postprocess_end`

`on_predict_batch_end`

`on_predict_end`

### Exporter

`on_export_start`

`on_export_end`
