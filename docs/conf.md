## Ultralytics YOLO

Default training settings and hyperparameters for medium-augmentation COCO training

### Setting the operation type
???+ note "Operation"

    | Key    | Value    | Description                                                                                                                                                                                 |
    |--------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    | task  | `detect` | Set the task via CLI. See Tasks for all supported tasks like - `detect`, `segment`, `classify`.<br> - `init` is a special case that creates a copy of default.yaml configs to the current working dir |
    | mode  | `train`  | Set the mode via CLI. It can be `train`, `val`, `predict`   |
    | resume  | `False`  | Resume last given task when set to `True`. <br> Resume from a given checkpoint is `model.pt` is passed  |
    | model | null     | Set the model. Format can differ for task type. Supports `model_name`, `model.yaml` & `model.pt`                                                                                            |
    | data  | null     | Set the data. Format can differ for task type. Supports `data.yaml`, `data_folder`, `dataset_name`|

### Training settings
??? note "Train"
    | Key              | Value  | Description                                                                     |
    |------------------|--------|---------------------------------------------------------------------------------|
    | device          | ''      | cuda device, i.e. 0 or 0,1,2,3 or cpu. `''` selects available cuda 0 device    |
    | epochs          | 100    | Number of epochs to train                                                       |
    | workers         | 8      | Number of cpu workers used per process. Scales automatically with DDP           |
    | batch_size      | 16     | Batch size of the dataloader                                                    |
    | img_size        | 640    | Image size of data in dataloader                                                |
    | optimizer       | SGD    | Optimizer used. Supported optimizer are: `Adam`, `SGD`, `RMSProp`               |
    | single_cls      | False  | Train on multi-class data as single-class                                       |
    | image_weights   | False  | Use weighted image selection for training                                       |
    | rect            | False  | Enable rectangular training                                                     |
    | cos_lr          | False  | Use cosine LR scheduler                        |
    | lr0             | 0.01   | Initial learning rate                          |
    | lrf             | 0.01   | Final OneCycleLR learning rate                 |
    | momentum        | 0.937  | Use as `momentum` for SGD and `beta1` for Adam |
    | weight_decay    | 0.0005 | Optimizer weight decay                         |
    | warmup_epochs   | 3.0    | Warmup epochs. Fractions are ok.               |
    | warmup_momentum | 0.8    | Warmup initial momentum                        |
    | warmup_bias_lr  | 0.1    | Warmup initial bias lr                         |
    | box             | 0.05   | Box loss gain                                  |
    | cls             | 0.5    | cls loss gain                                  |
    | cls_pw          | 1.0    | cls BCELoss positive_weight                    |
    | obj             | 1.0    | bj loss gain (scale with pixels)               |
    | obj_pw          | 1.0    | obj BCELoss positive_weight                    |
    | iou_t           | 0.20   | IOU training threshold                         |
    | anchor_t        | 4.0    | anchor-multiple threshold                      |
    | fl_gamma        | 0.0    | focal loss gamma                               |
    | label_smoothing | 0.0    |                                                |
    | nbs             | 64     | nominal batch size                             |
    | overlap_mask    | `True` | **Segmentation**: Use mask overlapping during training |
    | mask_ratio      | 4      | **Segmentation**: Set mask downsampling         |
    | dropout         | `False`| **Classification**: Use dropout while training   |
### Prediction Settings
??? note "Prediction"
    | Key            | Value                | Description                                        |
    |----------------|----------------------|----------------------------------------------------|
    | source         | `ultralytics/assets` | Input source. Accepts image, folder, video, url    |
    | view_img       | `False`              | View the prediction images                         |
    | save_txt       | `False`              | Save the results in a txt file                     |
    | save_conf      | `False`              | Save the condidence scores                         |
    | save_crop      | `Fasle`              |                                                    |
    | hide_labels    | `False`              | Hide the labels                                    |
    | hide_conf      | `False`              | Hide the confidence scores                         |
    | vid_stride     | `False`              | Input video frame-rate stride                      |
    | line_thickness | `3`                  | Bounding-box thickness (pixels)                    |
    | visualize      | `False`              | Visualize model features                           |
    | augment        | `False`              | Augmented inference                                |
    | agnostic_nms   | `False`              | Class-agnostic NMS                                 |
    | retina_masks   | `False`              | **Segmentation:** High resolution masks            |


### Validation settings
??? note "Validation"
    | Key         | Value   | Description                       |
    |-------------|---------|-----------------------------------|
    | noval       | `False` | ???                               |
    | save_json   | `False` |                                   |
    | save_hybrid | `False` |                                   |
    | conf_thres  | `0.001` | Confidence threshold              |
    | iou_thres   | `0.6`   | IoU threshold                     |
    | max_det     | `300`   | Maximum number of detections      |
    | half        | `True`  | Use .half() mode.                 |
    | dnn         | `False` | Use OpenCV DNN for ONNX inference |
    | plots       | `False` |                                   |

### Augmentation settings
??? note "Augmentation"

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
??? note "files"
    | Key       | Value   | Description                                                                                 |
    |-----------|---------|---------------------------------------------------------------------------------------------|
    | project:  | 'runs'  | The project name                                                                            |
    | name:     | 'exp'   | The run name. `exp` gets automatically incremented if not specified, i.e, `exp`, `exp2` ... |
    | exist_ok: | `False` | ???                                                                                         |
    | plots     | `False` | **Validation**: Save plots while validation                                                 |
    | nosave    | `False` | Don't save any plots, models or files                                                       |