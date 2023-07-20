---
comments: true
description: 'Guide for Validating YOLOv8 Models: Learn how to evaluate the performance of your YOLO models using validation settings and metrics with Python and CLI examples.'
keywords: Ultralytics, YOLO Docs, YOLOv8, validation, model evaluation, hyperparameters, accuracy, metrics, Python, CLI
---

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png">

**Val mode** is used for validating a YOLOv8 model after it has been trained. In this mode, the model is evaluated on a validation set to measure its accuracy and generalization performance. This mode can be used to tune the hyperparameters of the model to improve its performance.

!!! tip "Tip"

    * YOLOv8 models automatically remember their training settings, so you can validate a model at the same image size and on the original dataset easily with just `yolo val model=yolov8n.pt` or `model('yolov8n.pt').val()`

## Usage Examples

Validate trained YOLOv8n model accuracy on the COCO128 dataset. No argument need to passed as the `model` retains it's training `data` and arguments as model attributes. See Arguments section below for a full list of export arguments.

!!! example ""

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

## Arguments

Validation settings for YOLO models refer to the various hyperparameters and configurations used to evaluate the model's performance on a validation dataset. These settings can affect the model's performance, speed, and accuracy. Some common YOLO validation settings include the batch size, the frequency with which validation is performed during training, and the metrics used to evaluate the model's performance. Other factors that may affect the validation process include the size and composition of the validation dataset and the specific task the model is being used for. It is important to carefully tune and experiment with these settings to ensure that the model is performing well on the validation dataset and to detect and prevent overfitting.

| Key           | Value   | Description                                                        |
|---------------|---------|--------------------------------------------------------------------|
| `data`        | `None`  | path to data file, i.e. coco128.yaml                               |
| `imgsz`       | `640`   | image size as scalar or (h, w) list, i.e. (640, 480)               |
| `batch`       | `16`    | number of images per batch (-1 for AutoBatch)                      |
| `save_json`   | `False` | save results to JSON file                                          |
| `save_hybrid` | `False` | save hybrid version of labels (labels + additional predictions)    |
| `conf`        | `0.001` | object confidence threshold for detection                          |
| `iou`         | `0.6`   | intersection over union (IoU) threshold for NMS                    |
| `max_det`     | `300`   | maximum number of detections per image                             |
| `half`        | `True`  | use half precision (FP16)                                          |
| `device`      | `None`  | device to run on, i.e. cuda device=0/1/2/3 or device=cpu           |
| `dnn`         | `False` | use OpenCV DNN for ONNX inference                                  |
| `plots`       | `False` | show plots during training                                         |
| `rect`        | `False` | rectangular val with each batch collated for minimum padding       |
| `split`       | `val`   | dataset split to use for validation, i.e. 'val', 'test' or 'train' |
|