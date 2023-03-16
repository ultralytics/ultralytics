<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png">

**Train mode** is used for training a YOLOv8 model on a custom dataset. In this mode, the model is trained using the
specified dataset and hyperparameters. The training process involves optimizing the model's parameters so that it can
accurately predict the classes and locations of objects in an image.

!!! tip "Tip"

    * YOLOv8 datasets like COCO, VOC, ImageNet and many others automatically download on first use, i.e. `yolo train data=coco.yaml`

## Usage Examples

Train YOLOv8n on the COCO128 dataset for 100 epochs at image size 640. See Arguments section below for a full list of
training arguments.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO('yolov8n.yaml')  # build a new model from YAML
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
        
        # Train the model
        model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"
    
        ```bash
        # Build a new model from YAML and start training from scratch
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

## Arguments

Training settings for YOLO models refer to the various hyperparameters and configurations used to train the model on a
dataset. These settings can affect the model's performance, speed, and accuracy. Some common YOLO training settings
include the batch size, learning rate, momentum, and weight decay. Other factors that may affect the training process
include the choice of optimizer, the choice of loss function, and the size and composition of the training dataset. It
is important to carefully tune and experiment with these settings to achieve the best possible performance for a given
task.

| Key               | Value    | Description                                                                 |
|-------------------|----------|-----------------------------------------------------------------------------|
| `model`           | `None`   | path to model file, i.e. yolov8n.pt, yolov8n.yaml                           |
| `data`            | `None`   | path to data file, i.e. coco128.yaml                                        |
| `epochs`          | `100`    | number of epochs to train for                                               |
| `patience`        | `50`     | epochs to wait for no observable improvement for early stopping of training |
| `batch`           | `16`     | number of images per batch (-1 for AutoBatch)                               |
| `imgsz`           | `640`    | size of input images as integer or w,h                                      |
| `save`            | `True`   | save train checkpoints and predict results                                  |
| `save_period`     | `-1`     | Save checkpoint every x epochs (disabled if < 1)                            |
| `cache`           | `False`  | True/ram, disk or False. Use cache for data loading                         |
| `device`          | `None`   | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu        |
| `workers`         | `8`      | number of worker threads for data loading (per RANK if DDP)                 |
| `project`         | `None`   | project name                                                                |
| `name`            | `None`   | experiment name                                                             |
| `exist_ok`        | `False`  | whether to overwrite existing experiment                                    |
| `pretrained`      | `False`  | whether to use a pretrained model                                           |
| `optimizer`       | `'SGD'`  | optimizer to use, choices=['SGD', 'Adam', 'AdamW', 'RMSProp']               |
| `verbose`         | `False`  | whether to print verbose output                                             |
| `seed`            | `0`      | random seed for reproducibility                                             |
| `deterministic`   | `True`   | whether to enable deterministic mode                                        |
| `single_cls`      | `False`  | train multi-class data as single-class                                      |
| `image_weights`   | `False`  | use weighted image selection for training                                   |
| `rect`            | `False`  | support rectangular training                                                |
| `cos_lr`          | `False`  | use cosine learning rate scheduler                                          |
| `close_mosaic`    | `10`     | disable mosaic augmentation for final 10 epochs                             |
| `resume`          | `False`  | resume training from last checkpoint                                        |
| `lr0`             | `0.01`   | initial learning rate (i.e. SGD=1E-2, Adam=1E-3)                            |
| `lrf`             | `0.01`   | final learning rate (lr0 * lrf)                                             |
| `momentum`        | `0.937`  | SGD momentum/Adam beta1                                                     |
| `weight_decay`    | `0.0005` | optimizer weight decay 5e-4                                                 |
| `warmup_epochs`   | `3.0`    | warmup epochs (fractions ok)                                                |
| `warmup_momentum` | `0.8`    | warmup initial momentum                                                     |
| `warmup_bias_lr`  | `0.1`    | warmup initial bias lr                                                      |
| `box`             | `7.5`    | box loss gain                                                               |
| `cls`             | `0.5`    | cls loss gain (scale with pixels)                                           |
| `dfl`             | `1.5`    | dfl loss gain                                                               |
| `fl_gamma`        | `0.0`    | focal loss gamma (efficientDet default gamma=1.5)                           |
| `label_smoothing` | `0.0`    | label smoothing (fraction)                                                  |
| `nbs`             | `64`     | nominal batch size                                                          |
| `overlap_mask`    | `True`   | masks should overlap during training (segment train only)                   |
| `mask_ratio`      | `4`      | mask downsample ratio (segment train only)                                  |
| `dropout`         | `0.0`    | use dropout regularization (classify train only)                            |
| `val`             | `True`   | validate/test during training                                               |
