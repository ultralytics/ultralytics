YOLO settings and hyperparameters play a critical role in the model's performance, speed, and accuracy. These settings
and hyperparameters can affect the model's behavior at various stages of the model development process, including
training, validation, and prediction.

YOLOv8 'yolo' CLI commands use the following syntax:

!!! example ""

    === "CLI"
    
        ```bash
        yolo TASK MODE ARGS
        ```

Where:

- `TASK` (optional) is one of `[detect, segment, classify]`. If it is not passed explicitly YOLOv8 will try to guess
  the `TASK` from the model type.
- `MODE` (required) is one of `[train, val, predict, export]`
- `ARGS` (optional) are any number of custom `arg=value` pairs like `imgsz=320` that override defaults.
  For a full list of available `ARGS` see the [Configuration](cfg.md) page and `defaults.yaml`
  GitHub [source](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/cfg/default.yaml).

#### Tasks

YOLO models can be used for a variety of tasks, including detection, segmentation, and classification. These tasks
differ in the type of output they produce and the specific problem they are designed to solve.

- **Detect**: Detection tasks involve identifying and localizing objects or regions of interest in an image or video.
  YOLO models can be used for object detection tasks by predicting the bounding boxes and class labels of objects in an
  image.
- **Segment**: Segmentation tasks involve dividing an image or video into regions or pixels that correspond to
  different objects or classes. YOLO models can be used for image segmentation tasks by predicting a mask or label for
  each pixel in an image.
- **Classify**: Classification tasks involve assigning a class label to an input, such as an image or text. YOLO
  models can be used for image classification tasks by predicting the class label of an input image.

#### Modes

YOLO models can be used in different modes depending on the specific problem you are trying to solve. These modes
include train, val, and predict.

- **Train**: The train mode is used to train the model on a dataset. This mode is typically used during the development
  and
  testing phase of a model.
- **Val**: The val mode is used to evaluate the model's performance on a validation dataset. This mode is typically used
  to
  tune the model's hyperparameters and detect overfitting.
- **Predict**: The predict mode is used to make predictions with the model on new data. This mode is typically used in
  production or when deploying the model to users.

| Key    | Value    | Description                                                                                   |
|--------|----------|-----------------------------------------------------------------------------------------------|
| task   | 'detect' | inference task, i.e. detect, segment, or classify                                             |
| mode   | 'train'  | YOLO mode, i.e. train, val, predict, or export                                                |
| resume | False    | resume training from last checkpoint or custom checkpoint if passed as resume=path/to/best.pt |
| model  | null     | path to model file, i.e. yolov8n.pt, yolov8n.yaml                                             |
| data   | null     | path to data file, i.e. i.e. coco128.yaml                                                     |

### Training

Training settings for YOLO models refer to the various hyperparameters and configurations used to train the model on a
dataset. These settings can affect the model's performance, speed, and accuracy. Some common YOLO training settings
include the batch size, learning rate, momentum, and weight decay. Other factors that may affect the training process
include the choice of optimizer, the choice of loss function, and the size and composition of the training dataset. It
is important to carefully tune and experiment with these settings to achieve the best possible performance for a given
task.

| Key             | Value  | Description                                                                 |
|-----------------|--------|-----------------------------------------------------------------------------|
| model           | null   | path to model file, i.e. yolov8n.pt, yolov8n.yaml                           |
| data            | null   | path to data file, i.e. i.e. coco128.yaml                                   |
| epochs          | 100    | number of epochs to train for                                               |
| patience        | 50     | epochs to wait for no observable improvement for early stopping of training |
| batch           | 16     | number of images per batch (-1 for AutoBatch)                               |
| imgsz           | 640    | size of input images as integer or w,h                                      |
| save            | True   | save train checkpoints and predict results                                  |
| cache           | False  | True/ram, disk or False. Use cache for data loading                         |
| device          | null   | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu        |
| workers         | 8      | number of worker threads for data loading (per RANK if DDP)                 |
| project         | null   | project name                                                                |
| name            | null   | experiment name                                                             |
| exist_ok        | False  | whether to overwrite existing experiment                                    |
| pretrained      | False  | whether to use a pretrained model                                           |
| optimizer       | 'SGD'  | optimizer to use, choices=['SGD', 'Adam', 'AdamW', 'RMSProp']               |
| verbose         | False  | whether to print verbose output                                             |
| seed            | 0      | random seed for reproducibility                                             |
| deterministic   | True   | whether to enable deterministic mode                                        |
| single_cls      | False  | train multi-class data as single-class                                      |
| image_weights   | False  | use weighted image selection for training                                   |
| rect            | False  | support rectangular training                                                |
| cos_lr          | False  | use cosine learning rate scheduler                                          |
| close_mosaic    | 10     | disable mosaic augmentation for final 10 epochs                             |
| resume          | False  | resume training from last checkpoint                                        |
| lr0             | 0.01   | initial learning rate (i.e. SGD=1E-2, Adam=1E-3)                            |
| lrf             | 0.01   | final learning rate (lr0 * lrf)                                             |
| momentum        | 0.937  | SGD momentum/Adam beta1                                                     |
| weight_decay    | 0.0005 | optimizer weight decay 5e-4                                                 |
| warmup_epochs   | 3.0    | warmup epochs (fractions ok)                                                |
| warmup_momentum | 0.8    | warmup initial momentum                                                     |
| warmup_bias_lr  | 0.1    | warmup initial bias lr                                                      |
| box             | 7.5    | box loss gain                                                               |
| cls             | 0.5    | cls loss gain (scale with pixels)                                           |
| dfl             | 1.5    | dfl loss gain                                                               |
| fl_gamma        | 0.0    | focal loss gamma (efficientDet default gamma=1.5)                           |
| label_smoothing | 0.0    | label smoothing (fraction)                                                  |
| nbs             | 64     | nominal batch size                                                          |
| overlap_mask    | True   | masks should overlap during training (segment train only)                   |
| mask_ratio      | 4      | mask downsample ratio (segment train only)                                  |
| dropout         | 0.0    | use dropout regularization (classify train only)                            |

### Prediction

Prediction settings for YOLO models refer to the various hyperparameters and configurations used to make predictions
with the model on new data. These settings can affect the model's performance, speed, and accuracy. Some common YOLO
prediction settings include the confidence threshold, non-maximum suppression (NMS) threshold, and the number of classes
to consider. Other factors that may affect the prediction process include the size and format of the input data, the
presence of additional features such as masks or multiple labels per box, and the specific task the model is being used
for. It is important to carefully tune and experiment with these settings to achieve the best possible performance for a
given task.

| Key            | Value                | Description                                             |
|----------------|----------------------|---------------------------------------------------------|
| source         | 'ultralytics/assets' | source directory for images or videos                   |
| show           | False                | show results if possible                                |
| save_txt       | False                | save results as .txt file                               |
| save_conf      | False                | save results with confidence scores                     |
| save_crop      | Fasle                | save cropped images with results                        |
| hide_labels    | False                | hide labels                                             |
| hide_conf      | False                | hide confidence scores                                  |
| vid_stride     | False                | video frame-rate stride                                 |
| line_thickness | 3                    | bounding box thickness (pixels)                         |
| visualize      | False                | visualize model features                                |
| augment        | False                | apply image augmentation to prediction sources          |
| agnostic_nms   | False                | class-agnostic NMS                                      |
| retina_masks   | False                | use high-resolution segmentation masks                  |
| classes        | null                 | filter results by class, i.e. class=0, or class=[0,2,3] |

### Validation

Validation settings for YOLO models refer to the various hyperparameters and configurations used to
evaluate the model's performance on a validation dataset. These settings can affect the model's performance, speed, and
accuracy. Some common YOLO validation settings include the batch size, the frequency with which validation is performed
during training, and the metrics used to evaluate the model's performance. Other factors that may affect the validation
process include the size and composition of the validation dataset and the specific task the model is being used for. It
is important to carefully tune and experiment with these settings to ensure that the model is performing well on the
validation dataset and to detect and prevent overfitting.

| Key         | Value | Description                                                                 |
|-------------|-------|-----------------------------------------------------------------------------|
| val         | True  | validate/test during training                                               |
| save_json   | False | save results to JSON file                                                   |
| save_hybrid | False | save hybrid version of labels (labels + additional predictions)             |
| conf        | 0.001 | object confidence threshold for detection (default 0.25 predict, 0.001 val) |
| iou         | 0.6   | intersection over union (IoU) threshold for NMS                             |
| max_det     | 300   | maximum number of detections per image                                      |
| half        | True  | use half precision (FP16)                                                   |
| dnn         | False | use OpenCV DNN for ONNX inference                                           |
| plots       | False | show plots during training                                                  |

### Export

Export settings for YOLO models refer to the various configurations and options used to save or
export the model for use in other environments or platforms. These settings can affect the model's performance, size,
and compatibility with different systems. Some common YOLO export settings include the format of the exported model
file (e.g. ONNX, TensorFlow SavedModel), the device on which the model will be run (e.g. CPU, GPU), and the presence of
additional features such as masks or multiple labels per box. Other factors that may affect the export process include
the specific task the model is being used for and the requirements or constraints of the target environment or platform.
It is important to carefully consider and configure these settings to ensure that the exported model is optimized for
the intended use case and can be used effectively in the target environment.

### Augmentation

Augmentation settings for YOLO models refer to the various transformations and modifications
applied to the training data to increase the diversity and size of the dataset. These settings can affect the model's
performance, speed, and accuracy. Some common YOLO augmentation settings include the type and intensity of the
transformations applied (e.g. random flips, rotations, cropping, color changes), the probability with which each
transformation is applied, and the presence of additional features such as masks or multiple labels per box. Other
factors that may affect the augmentation process include the size and composition of the original dataset and the
specific task the model is being used for. It is important to carefully tune and experiment with these settings to
ensure that the augmented dataset is diverse and representative enough to train a high-performing model.

| Key         | Value | Description                                     |
|-------------|-------|-------------------------------------------------|
| hsv_h       | 0.015 | image HSV-Hue augmentation (fraction)           |
| hsv_s       | 0.7   | image HSV-Saturation augmentation (fraction)    |
| hsv_v       | 0.4   | image HSV-Value augmentation (fraction)         |
| degrees     | 0.0   | image rotation (+/- deg)                        |
| translate   | 0.1   | image translation (+/- fraction)                |
| scale       | 0.5   | image scale (+/- gain)                          |
| shear       | 0.0   | image shear (+/- deg)                           |
| perspective | 0.0   | image perspective (+/- fraction), range 0-0.001 |
| flipud      | 0.0   | image flip up-down (probability)                |
| fliplr      | 0.5   | image flip left-right (probability)             |
| mosaic      | 1.0   | image mosaic (probability)                      |
| mixup       | 0.0   | image mixup (probability)                       |
| copy_paste  | 0.0   | segment copy-paste (probability)                |

### Logging, checkpoints, plotting and file management

Logging, checkpoints, plotting, and file management are important considerations when training a YOLO model.

- Logging: It is often helpful to log various metrics and statistics during training to track the model's progress and
  diagnose any issues that may arise. This can be done using a logging library such as TensorBoard or by writing log
  messages to a file.
- Checkpoints: It is a good practice to save checkpoints of the model at regular intervals during training. This allows
  you to resume training from a previous point if the training process is interrupted or if you want to experiment with
  different training configurations.
- Plotting: Visualizing the model's performance and training progress can be helpful for understanding how the model is
  behaving and identifying potential issues. This can be done using a plotting library such as matplotlib or by
  generating plots using a logging library such as TensorBoard.
- File management: Managing the various files generated during the training process, such as model checkpoints, log
  files, and plots, can be challenging. It is important to have a clear and organized file structure to keep track of
  these files and make it easy to access and analyze them as needed.

Effective logging, checkpointing, plotting, and file management can help you keep track of the model's progress and make
it easier to debug and optimize the training process.

| Key      | Value  | Description                                                                                    |
|----------|--------|------------------------------------------------------------------------------------------------|
| project  | 'runs' | project name                                                                                   |
| name     | 'exp'  | experiment name. `exp` gets automatically incremented if not specified, i.e, `exp`, `exp2` ... |
| exist_ok | False  | whether to overwrite existing experiment                                                       |
| plots    | False  | save plots during train/val                                                                    |
| save     | False  | save train checkpoints and predict results                                                     |