---
comments: true
description: Validate and improve YOLOv8n model accuracy on COCO128 and other datasets using hyperparameter & configuration tuning, in Test mode.
---

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png">

**Test mode** is used for validating a YOLOv8 model after it has been trained using test data. In this mode, the model is evaluated on a
validation set to measure its accuracy and generalization performance. This mode can be used to tune the hyperparameters
of the model to improve its performance. Test mode actually have the same effect with Val mode with `split=test`.


!!! tip "Tip"

    * YOLOv8 models automatically remember their training settings, so you can test a model at the same image size and on the original dataset easily with just `yolo test model=yolov8n.pt` or `model('yolov8n.pt').test()`

## Usage Examples

Test trained YOLOv8n model accuracy on the COCO128 dataset. No argument need to passed as the `model` retains it's
training `data` and arguments as model attributes. See Arguments section below for a full list of export arguments.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO('yolov8n.pt')  # load an official model
        model = YOLO('path/to/best.pt')  # load a custom model
        
        # Validate the model
        metrics = model.test()  # no arguments needed, dataset and settings remembered
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # a list contains map50-95 of each category
        ```
    === "CLI"
    
        ```bash
        yolo detect test model=yolov8n.pt  # val official model
        yolo detect test model=path/to/best.pt  # val custom model
        ```

## Arguments
See full argument details in the [Val](https://docs.ultralytics.com/modes/val/) page.

