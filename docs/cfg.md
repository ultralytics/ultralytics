YOLO settings and hyperparameters play a critical role in the model's performance, speed, and accuracy. These settings
and hyperparameters can affect the model's behavior at various stages of the model development process, including
training, validation, and prediction.

Properly setting and tuning these parameters can have a significant impact on the model's ability to learn effectively
from the training data and generalize to new data. For example, choosing an appropriate learning rate, batch size, and
optimization algorithm can greatly affect the model's convergence speed and accuracy. Similarly, setting the correct
confidence threshold and non-maximum suppression (NMS) threshold can affect the model's performance on detection tasks.

It is important to carefully consider and experiment with these settings and hyperparameters to achieve the best
possible performance for a given task. This can involve trial and error, as well as using techniques such as
hyperparameter optimization to search for the optimal set of parameters.

In summary, YOLO settings and hyperparameters are a key factor in the success of a YOLO model, and it is important to
pay careful attention to them to achieve the desired results.

### Setting the operation type

YOLO models can be used for a variety of tasks, including detection, segmentation, and classification. These tasks
differ in the type of output they produce and the specific problem they are designed to solve.

- Detection: Detection tasks involve identifying and localizing objects or regions of interest in an image or video.
  YOLO models can be used for object detection tasks by predicting the bounding boxes and class labels of objects in an
  image.
- Segmentation: Segmentation tasks involve dividing an image or video into regions or pixels that correspond to
  different objects or classes. YOLO models can be used for image segmentation tasks by predicting a mask or label for
  each pixel in an image.
- Classification: Classification tasks involve assigning a class label to an input, such as an image or text. YOLO
  models can be used for image classification tasks by predicting the class label of an input image.

YOLO models can be used in different modes depending on the specific problem you are trying to solve. These modes
include train, val, and predict.

- Train: The train mode is used to train the model on a dataset. This mode is typically used during the development and
  testing phase of a model.
- Val: The val mode is used to evaluate the model's performance on a validation dataset. This mode is typically used to
  tune the model's hyperparameters and detect overfitting.
- Predict: The predict mode is used to make predictions with the model on new data. This mode is typically used in
  production or when deploying the model to users.

| Key    | Value    | Description                                                                                            |
|--------|----------|--------------------------------------------------------------------------------------------------------|
| task   | `detect` | Set the task via CLI. See Tasks for all supported tasks like - `detect`, `segment`, `classify`         |
| mode   | `train`  | Set the mode via CLI. It can be `train`, `val`, `predict`, `export`                                    |
| resume | `False`  | Resume last given task when set to `True`. <br> Resume from a given checkpoint is `model.pt` is passed |
| model  | null     | Set the model. Format can differ for task type. Supports `model_name`, `model.yaml` & `model.pt`       |
| data   | null     | Set the data. Format can differ for task type. Supports `data.yaml`, `data_folder`, `dataset_name`     |

### Training

Training settings for YOLO models refer to the various hyperparameters and configurations used to train the model on a
dataset. These settings can affect the model's performance, speed, and accuracy. Some common YOLO training settings
include the batch size, learning rate, momentum, and weight decay. Other factors that may affect the training process
include the choice of optimizer, the choice of loss function, and the size and composition of the training dataset. It
is important to carefully tune and experiment with these settings to achieve the best possible performance for a given
task.

| Key             | Value   | Description                                                                 |
|-----------------|---------|-----------------------------------------------------------------------------|
| device          | ''      | cuda device, i.e. 0 or 0,1,2,3 or cpu. `''` selects available cuda 0 device |
| epochs          | 100     | Number of epochs to train                                                   |
| workers         | 8       | Number of cpu workers used per process. Scales automatically with DDP       |
| batch           | 16      | Batch size of the dataloader                                                |
| imgsz           | 640     | Image size of data in dataloader                                            |
| optimizer       | SGD     | Optimizer used. Supported optimizer are: `Adam`, `SGD`, `RMSProp`           |
| single_cls      | False   | Train on multi-class data as single-class                                   |
| image_weights   | False   | Use weighted image selection for training                                   |
| rect            | False   | Enable rectangular training                                                 |
| cos_lr          | False   | Use cosine LR scheduler                                                     |
| lr0             | 0.01    | Initial learning rate                                                       |
| lrf             | 0.01    | Final OneCycleLR learning rate                                              |
| momentum        | 0.937   | Use as `momentum` for SGD and `beta1` for Adam                              |
| weight_decay    | 0.0005  | Optimizer weight decay                                                      |
| warmup_epochs   | 3.0     | Warmup epochs. Fractions are ok.                                            |
| warmup_momentum | 0.8     | Warmup initial momentum                                                     |
| warmup_bias_lr  | 0.1     | Warmup initial bias lr                                                      |
| box             | 0.05    | Box loss gain                                                               |
| cls             | 0.5     | cls loss gain                                                               |
| cls_pw          | 1.0     | cls BCELoss positive_weight                                                 |
| obj             | 1.0     | bj loss gain (scale with pixels)                                            |
| obj_pw          | 1.0     | obj BCELoss positive_weight                                                 |
| iou_t           | 0.20    | IOU training threshold                                                      |
| anchor_t        | 4.0     | anchor-multiple threshold                                                   |
| fl_gamma        | 0.0     | focal loss gamma                                                            |
| label_smoothing | 0.0     |                                                                             |
| nbs             | 64      | nominal batch size                                                          |
| overlap_mask    | `True`  | **Segmentation**: Use mask overlapping during training                      |
| mask_ratio      | 4       | **Segmentation**: Set mask downsampling                                     |
| dropout         | `False` | **Classification**: Use dropout while training                              |

### Prediction

Prediction settings for YOLO models refer to the various hyperparameters and configurations used to make predictions
with the model on new data. These settings can affect the model's performance, speed, and accuracy. Some common YOLO
prediction settings include the confidence threshold, non-maximum suppression (NMS) threshold, and the number of classes
to consider. Other factors that may affect the prediction process include the size and format of the input data, the
presence of additional features such as masks or multiple labels per box, and the specific task the model is being used
for. It is important to carefully tune and experiment with these settings to achieve the best possible performance for a
given task.

| Key            | Value                | Description                                     |
|----------------|----------------------|-------------------------------------------------|
| source         | `ultralytics/assets` | Input source. Accepts image, folder, video, url |
| show           | `False`              | View the prediction images                      |
| save_txt       | `False`              | Save the results in a txt file                  |
| save_conf      | `False`              | Save the condidence scores                      |
| save_crop      | `Fasle`              |                                                 |
| hide_labels    | `False`              | Hide the labels                                 |
| hide_conf      | `False`              | Hide the confidence scores                      |
| vid_stride     | `False`              | Input video frame-rate stride                   |
| line_thickness | `3`                  | Bounding-box thickness (pixels)                 |
| visualize      | `False`              | Visualize model features                        |
| augment        | `False`              | Augmented inference                             |
| agnostic_nms   | `False`              | Class-agnostic NMS                              |
| retina_masks   | `False`              | **Segmentation:** High resolution masks         |

### Validation

Validation settings for YOLO models refer to the various hyperparameters and configurations used to
evaluate the model's performance on a validation dataset. These settings can affect the model's performance, speed, and
accuracy. Some common YOLO validation settings include the batch size, the frequency with which validation is performed
during training, and the metrics used to evaluate the model's performance. Other factors that may affect the validation
process include the size and composition of the validation dataset and the specific task the model is being used for. It
is important to carefully tune and experiment with these settings to ensure that the model is performing well on the
validation dataset and to detect and prevent overfitting.

| Key         | Value   | Description                       |
|-------------|---------|-----------------------------------|
| noval       | `False` | ???                               |
| save_json   | `False` |                                   |
| save_hybrid | `False` |                                   |
| conf        | `0.001` | Confidence threshold              |
| iou         | `0.6`   | IoU threshold                     |
| max_det     | `300`   | Maximum number of detections      |
| half        | `True`  | Use .half() mode.                 |
| dnn         | `False` | Use OpenCV DNN for ONNX inference |
| plots       | `False` |                                   |

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

| hsv_h       | 0.015 | Image HSV-Hue augmentation (fraction)           |
|-------------|-------|-------------------------------------------------|
| hsv_s       | 0.7   | Image HSV-Saturation augmentation (fraction)    |
| hsv_v       | 0.4   | Image HSV-Value augmentation (fraction)         |
| degrees     | 0.0   | Image rotation (+/- deg)                        |
| translate   | 0.1   | Image translation (+/- fraction)                |
| scale       | 0.5   | Image scale (+/- gain)                          |
| shear       | 0.0   | Image shear (+/- deg)                           |
| perspective | 0.0   | Image perspective (+/- fraction), range 0-0.001 |
| flipud      | 0.0   | Image flip up-down (probability)                |
| fliplr      | 0.5   | Image flip left-right (probability)             |
| mosaic      | 1.0   | Image mosaic (probability)                      |
| mixup       | 0.0   | Image mixup (probability)                       |
| copy_paste  | 0.0   | Segment copy-paste (probability)                |

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

| Key       | Value   | Description                                                                                 |
|-----------|---------|---------------------------------------------------------------------------------------------|
| project:  | 'runs'  | The project name                                                                            |
| name:     | 'exp'   | The run name. `exp` gets automatically incremented if not specified, i.e, `exp`, `exp2` ... |
| exist_ok: | `False` | Will replace current directory contents if set to True and output directory exists.         |
| plots     | `False` | **Validation**: Save plots while validation                                                 |
| save      | `False` | Save any plots, models or files                                                             |