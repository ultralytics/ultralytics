---
comments: true
description: Master YOLOv8 settings and hyperparameters for improved model performance. Learn to use YOLO CLI commands, adjust training settings, and optimize YOLO tasks & modes.
keywords: YOLOv8, settings, hyperparameters, YOLO CLI commands, YOLO tasks, YOLO modes, Ultralytics documentation, model optimization, YOLOv8 training
---

YOLO settings and hyperparameters play a critical role in the model's performance, speed, and accuracy. These settings and hyperparameters can affect the model's behavior at various stages of the model development process, including training, validation, and prediction.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=87"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Mastering Ultralytics YOLOv8: Configuration
</p>

Ultralytics commands use the following syntax:

!!! Example

    === "CLI"

        ```bash
        yolo TASK MODE ARGS
        ```

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLOv8 model from a pre-trained weights file
        model = YOLO('yolov8n.pt')

        # Run MODE mode using the custom arguments ARGS (guess TASK)
        model.MODE(ARGS)
        ```

Where:

- `TASK` (optional) is one of ([detect](../tasks/detect.md), [segment](../tasks/segment.md), [classify](../tasks/classify.md), [pose](../tasks/pose.md))
- `MODE` (required) is one of ([train](../modes/train.md), [val](../modes/val.md), [predict](../modes/predict.md), [export](../modes/export.md), [track](../modes/track.md))
- `ARGS` (optional) are `arg=value` pairs like `imgsz=640` that override defaults.

Default `ARG` values are defined on this page from the `cfg/defaults.yaml` [file](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml).

#### Tasks

YOLO models can be used for a variety of tasks, including detection, segmentation, classification and pose. These tasks differ in the type of output they produce and the specific problem they are designed to solve.

- **Detect**: For identifying and localizing objects or regions of interest in an image or video.
- **Segment**: For dividing an image or video into regions or pixels that correspond to different objects or classes.
- **Classify**: For predicting the class label of an input image.
- **Pose**: For identifying objects and estimating their keypoints in an image or video.

| Key    | Value      | Description                                     |
|--------|------------|-------------------------------------------------|
| `task` | `'detect'` | YOLO task, i.e. detect, segment, classify, pose |

[Tasks Guide](../tasks/index.md){ .md-button }

#### Modes

YOLO models can be used in different modes depending on the specific problem you are trying to solve. These modes include:

- **Train**: For training a YOLOv8 model on a custom dataset.
- **Val**: For validating a YOLOv8 model after it has been trained.
- **Predict**: For making predictions using a trained YOLOv8 model on new images or videos.
- **Export**: For exporting a YOLOv8 model to a format that can be used for deployment.
- **Track**: For tracking objects in real-time using a YOLOv8 model.
- **Benchmark**: For benchmarking YOLOv8 exports (ONNX, TensorRT, etc.) speed and accuracy.

| Key    | Value     | Description                                                   |
|--------|-----------|---------------------------------------------------------------|
| `mode` | `'train'` | YOLO mode, i.e. train, val, predict, export, track, benchmark |

[Modes Guide](../modes/index.md){ .md-button }

## Train

The training settings for YOLO models encompass various hyperparameters and configurations used during the training process. These settings influence the model's performance, speed, and accuracy. Key training settings include batch size, learning rate, momentum, and weight decay. Additionally, the choice of optimizer, loss function, and training dataset composition can impact the training process. Careful tuning and experimentation with these settings are crucial for optimizing performance.

| Key               | Default  | Description                                                                                                                                                                                                          |
|-------------------|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model`           | `None`   | Specifies the model file for training. Accepts a path to either a `.pt` pretrained model or a `.yaml` configuration file. Essential for defining the model structure or initializing weights.                        |
| `data`            | `None`   | Path to the dataset configuration file (e.g., `coco128.yaml`). This file contains dataset-specific parameters, including paths to training and validation data, class names, and number of classes.                  |
| `epochs`          | `100`    | Total number of training epochs. Each epoch represents a full pass over the entire dataset. Adjusting this value can affect training duration and model performance.                                                 |
| `time`            | `None`   | Maximum training time in hours. If set, this overrides the `epochs` argument, allowing training to automatically stop after the specified duration. Useful for time-constrained training scenarios.                  |
| `patience`        | `50`     | Number of epochs to wait without improvement in validation metrics before early stopping the training. Helps prevent overfitting by stopping training when performance plateaus.                                     |
| `batch`           | `16`     | Batch size for training, indicating how many images are processed before the model's internal parameters are updated. AutoBatch (`batch=-1`) dynamically adjusts the batch size based on GPU memory availability.    |
| `imgsz`           | `640`    | Target image size for training. All images are resized to this dimension before being fed into the model. Affects model accuracy and computational complexity.                                                       |
| `save`            | `True`   | Enables saving of training checkpoints and final model weights. Useful for resuming training or model deployment.                                                                                                    |
| `save_period`     | `-1`     | Frequency of saving model checkpoints, specified in epochs. A value of -1 disables this feature. Useful for saving interim models during long training sessions.                                                     |
| `cache`           | `False`  | Enables caching of dataset images in memory (`True`/`ram`), on disk (`disk`), or disables it (`False`). Improves training speed by reducing disk I/O at the cost of increased memory usage.                          |
| `device`          | `None`   | Specifies the computational device(s) for training: a single GPU (`device=0`), multiple GPUs (`device=0,1`), CPU (`device=cpu`), or MPS for Apple silicon (`device=mps`).                                            |
| `workers`         | `8`      | Number of worker threads for data loading (per `RANK` if Multi-GPU training). Influences the speed of data preprocessing and feeding into the model, especially useful in multi-GPU setups.                          |
| `project`         | `None`   | Name of the project directory where training outputs are saved. Allows for organized storage of different experiments.                                                                                               |
| `name`            | `None`   | Name of the training run. Used for creating a subdirectory within the project folder, where training logs and outputs are stored.                                                                                    |
| `exist_ok`        | `False`  | If True, allows overwriting of an existing project/name directory. Useful for iterative experimentation without needing to manually clear previous outputs.                                                          |
| `pretrained`      | `True`   | Determines whether to start training from a pretrained model. Can be a boolean value or a string path to a specific model from which to load weights. Enhances training efficiency and model performance.            |
| `optimizer`       | `'auto'` | Choice of optimizer for training. Options include `SGD`, `Adam`, `AdamW`, `NAdam`, `RAdam`, `RMSProp` etc., or `auto` for automatic selection based on model configuration. Affects convergence speed and stability. |
| `verbose`         | `False`  | Enables verbose output during training, providing detailed logs and progress updates. Useful for debugging and closely monitoring the training process.                                                              |
| `seed`            | `0`      | Sets the random seed for training, ensuring reproducibility of results across runs with the same configurations.                                                                                                     |
| `deterministic`   | `True`   | Forces deterministic algorithm use, ensuring reproducibility but may affect performance and speed due to the restriction on non-deterministic algorithms.                                                            |
| `single_cls`      | `False`  | Treats all classes in multi-class datasets as a single class during training. Useful for binary classification tasks or when focusing on object presence rather than classification.                                 |
| `rect`            | `False`  | Enables rectangular training, optimizing batch composition for minimal padding. Can improve efficiency and speed but may affect model accuracy.                                                                      |
| `cos_lr`          | `False`  | Utilizes a cosine learning rate scheduler, adjusting the learning rate following a cosine curve over epochs. Helps in managing learning rate for better convergence.                                                 |
| `close_mosaic`    | `10`     | Disables mosaic data augmentation in the last N epochs to stabilize training before completion. Setting to 0 disables this feature.                                                                                  |
| `resume`          | `False`  | Resumes training from the last saved checkpoint. Automatically loads model weights, optimizer state, and epoch count, continuing training seamlessly.                                                                |
| `amp`             | `True`   | Enables Automatic Mixed Precision (AMP) training, reducing memory usage and possibly speeding up training with minimal impact on accuracy.                                                                           |
| `fraction`        | `1.0`    | Specifies the fraction of the dataset to use for training. Allows for training on a subset of the full dataset, useful for experiments or when resources are limited.                                                |
| `profile`         | `False`  | Enables profiling of ONNX and TensorRT speeds during training, useful for optimizing model deployment.                                                                                                               |
| `freeze`          | `None`   | Freezes the first N layers of the model or specified layers by index, reducing the number of trainable parameters. Useful for fine-tuning or transfer learning.                                                      |
| `lr0`             | `0.01`   | Initial learning rate (i.e. `SGD=1E-2`, `Adam=1E-3`) . Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.                                              |
| `lrf`             | `0.01`   | Final learning rate as a fraction of the initial rate = (`lr0 * lrf`), used in conjunction with schedulers to adjust the learning rate over time.                                                                    |
| `momentum`        | `0.937`  | Momentum factor for SGD or beta1 for Adam optimizers, influencing the incorporation of past gradients in the current update.                                                                                         |
| `weight_decay`    | `0.0005` | L2 regularization term, penalizing large weights to prevent overfitting.                                                                                                                                             |
| `warmup_epochs`   | `3.0`    | Number of epochs for learning rate warmup, gradually increasing the learning rate from a low value to the initial learning rate to stabilize training early on.                                                      |
| `warmup_momentum` | `0.8`    | Initial momentum for warmup phase, gradually adjusting to the set momentum over the warmup period.                                                                                                                   |
| `warmup_bias_lr`  | `0.1`    | Learning rate for bias parameters during the warmup phase, helping stabilize model training in the initial epochs.                                                                                                   |
| `box`             | `7.5`    | Weight of the box loss component in the loss function, influencing how much emphasis is placed on accurately predicting bounding box coordinates.                                                                    |
| `cls`             | `0.5`    | Weight of the classification loss in the total loss function, affecting the importance of correct class prediction relative to other components.                                                                     |
| `dfl`             | `1.5`    | Weight of the distribution focal loss, used in certain YOLO versions for fine-grained classification.                                                                                                                |
| `pose`            | `12.0`   | Weight of the pose loss in models trained for pose estimation, influencing the emphasis on accurately predicting pose keypoints.                                                                                     |
| `kobj`            | `2.0`    | Weight of the keypoint objectness loss in pose estimation models, balancing detection confidence with pose accuracy.                                                                                                 |
| `label_smoothing` | `0.0`    | Applies label smoothing, softening hard labels to a mix of the target label and a uniform distribution over labels, can improve generalization.                                                                      |
| `nbs`             | `64`     | Nominal batch size for normalization of loss.                                                                                                                                                                        |
| `overlap_mask`    | `True`   | Determines whether segmentation masks should overlap during training, applicable in instance segmentation tasks.                                                                                                     |
| `mask_ratio`      | `4`      | Downsample ratio for segmentation masks, affecting the resolution of masks used during training.                                                                                                                     |
| `dropout`         | `0.0`    | Dropout rate for regularization in classification tasks, preventing overfitting by randomly omitting units during training.                                                                                          |
| `val`             | `True`   | Enables validation during training, allowing for periodic evaluation of model performance on a separate dataset.                                                                                                     |
| `plots`           | `False`  | Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression.                                             |

[Train Guide](../modes/train.md){ .md-button }

## Predict

The prediction settings for YOLO models encompass a range of hyperparameters and configurations that influence the model's performance, speed, and accuracy during inference on new data. Careful tuning and experimentation with these settings are essential to achieve optimal performance for a specific task. Key settings include the confidence threshold, Non-Maximum Suppression (NMS) threshold, and the number of classes considered. Additional factors affecting the prediction process are input data size and format, the presence of supplementary features such as masks or multiple labels per box, and the particular task the model is employed for.

Inference arguments:

| Argument        | Type           | Default                | Description                                                                                                                                                                                                                          |
|-----------------|----------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `source`        | `str`          | `'ultralytics/assets'` | Specifies the data source for inference. Can be an image path, video file, directory, URL, or device ID for live feeds. Supports a wide range of formats and sources, enabling flexible application across different types of input. |
| `conf`          | `float`        | `0.25`                 | Sets the minimum confidence threshold for detections. Objects detected with confidence below this threshold will be disregarded. Adjusting this value can help reduce false positives.                                               |
| `iou`           | `float`        | `0.7`                  | Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Higher values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.                                                |
| `imgsz`         | `int or tuple` | `640`                  | Defines the image size for inference. Can be a single integer `640` for square resizing or a (height, width) tuple. Proper sizing can improve detection accuracy and processing speed.                                               |
| `half`          | `bool`         | `False`                | Enables half-precision (FP16) inference, which can speed up model inference on supported GPUs with minimal impact on accuracy.                                                                                                       |
| `device`        | `str`          | `None`                 | Specifies the device for inference (e.g., `cpu`, `cuda:0` or `0`). Allows users to select between CPU, a specific GPU, or other compute devices for model execution.                                                                 |
| `max_det`       | `int`          | `300`                  | Maximum number of detections allowed per image. Limits the total number of objects the model can detect in a single inference, preventing excessive outputs in dense scenes.                                                         |
| `vid_stride`    | `int`          | `1`                    | Frame stride for video inputs. Allows skipping frames in videos to speed up processing at the cost of temporal resolution. A value of 1 processes every frame, higher values skip frames.                                            |
| `stream_buffer` | `bool`         | `False`                | Determines if all frames should be buffered when processing video streams (`True`), or if the model should return the most recent frame (`False`). Useful for real-time applications.                                                |
| `visualize`     | `bool`         | `False`                | Activates visualization of model features during inference, providing insights into what the model is "seeing". Useful for debugging and model interpretation.                                                                       |
| `augment`       | `bool`         | `False`                | Enables test-time augmentation (TTA) for predictions, potentially improving detection robustness at the cost of inference speed.                                                                                                     |
| `agnostic_nms`  | `bool`         | `False`                | Enables class-agnostic Non-Maximum Suppression (NMS), which merges overlapping boxes of different classes. Useful in multi-class detection scenarios where class overlap is common.                                                  |
| `classes`       | `list[int]`    | `None`                 | Filters predictions to a set of class IDs. Only detections belonging to the specified classes will be returned. Useful for focusing on relevant objects in multi-class detection tasks.                                              |
| `retina_masks`  | `bool`         | `False`                | Uses high-resolution segmentation masks if available in the model. This can enhance mask quality for segmentation tasks, providing finer detail.                                                                                     |
| `embed`         | `list[int]`    | `None`                 | Specifies the layers from which to extract feature vectors or embeddings. Useful for downstream tasks like clustering or similarity search.                                                                                          |

Visualization arguments:

| Argument      | Type          | Default | Description                                                                                                                                                                   |
|---------------|---------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `show`        | `bool`        | `False` | If `True`, displays the annotated images or videos in a window. Useful for immediate visual feedback during development or testing.                                           |
| `save`        | `bool`        | `False` | Enables saving of the annotated images or videos to file. Useful for documentation, further analysis, or sharing results.                                                     |
| `save_frames` | `bool`        | `False` | When processing videos, saves individual frames as images. Useful for extracting specific frames or for detailed frame-by-frame analysis.                                     |
| `save_txt`    | `bool`        | `False` | Saves detection results in a text file, following the format `[class] [x_center] [y_center] [width] [height] [confidence]`. Useful for integration with other analysis tools. |
| `save_conf`   | `bool`        | `False` | Includes confidence scores in the saved text files. Enhances the detail available for post-processing and analysis.                                                           |
| `save_crop`   | `bool`        | `False` | Saves cropped images of detections. Useful for dataset augmentation, analysis, or creating focused datasets for specific objects.                                             |
| `show_labels` | `bool`        | `True`  | Displays labels for each detection in the visual output. Provides immediate understanding of detected objects.                                                                |
| `show_conf`   | `bool`        | `True`  | Displays the confidence score for each detection alongside the label. Gives insight into the model's certainty for each detection.                                            |
| `show_boxes`  | `bool`        | `True`  | Draws bounding boxes around detected objects. Essential for visual identification and location of objects in images or video frames.                                          |
| `line_width`  | `None or int` | `None`  | Specifies the line width of bounding boxes. If `None`, the line width is automatically adjusted based on the image size. Provides visual customization for clarity.           |

[Predict Guide](../modes/predict.md){ .md-button }

## Val

The val (validation) settings for YOLO models involve various hyperparameters and configurations used to evaluate the model's performance on a validation dataset. These settings influence the model's performance, speed, and accuracy. Common YOLO validation settings include batch size, validation frequency during training, and performance evaluation metrics. Other factors affecting the validation process include the validation dataset's size and composition, as well as the specific task the model is employed for. Careful tuning and experimentation with these settings are crucial to ensure optimal performance on the validation dataset and detect and prevent overfitting.

| Key           | Value   | Description                                                        |
|---------------|---------|--------------------------------------------------------------------|
| `data`        | `None`  | path to data file, i.e. coco128.yaml                               |
| `imgsz`       | `640`   | size of input images as integer                                    |
| `batch`       | `16`    | number of images per batch (-1 for AutoBatch)                      |
| `save_json`   | `False` | save results to JSON file                                          |
| `save_hybrid` | `False` | save hybrid version of labels (labels + additional predictions)    |
| `conf`        | `0.001` | object confidence threshold for detection                          |
| `iou`         | `0.6`   | intersection over union (IoU) threshold for NMS                    |
| `max_det`     | `300`   | maximum number of detections per image                             |
| `half`        | `True`  | use half precision (FP16)                                          |
| `device`      | `None`  | device to run on, i.e. cuda device=0/1/2/3 or device=cpu           |
| `dnn`         | `False` | use OpenCV DNN for ONNX inference                                  |
| `plots`       | `False` | save plots and images during train/val                             |
| `rect`        | `False` | rectangular val with each batch collated for minimum padding       |
| `split`       | `val`   | dataset split to use for validation, i.e. 'val', 'test' or 'train' |

[Val Guide](../modes/val.md){ .md-button }

## Export

Export settings for YOLO models encompass configurations and options related to saving or exporting the model for use in different environments or platforms. These settings can impact the model's performance, size, and compatibility with various systems. Key export settings include the exported model file format (e.g., ONNX, TensorFlow SavedModel), the target device (e.g., CPU, GPU), and additional features such as masks or multiple labels per box. The export process may also be affected by the model's specific task and the requirements or constraints of the destination environment or platform. It is crucial to thoughtfully configure these settings to ensure the exported model is optimized for the intended use case and functions effectively in the target environment.

| Key         | Value           | Description                                          |
|-------------|-----------------|------------------------------------------------------|
| `format`    | `'torchscript'` | format to export to                                  |
| `imgsz`     | `640`           | image size as scalar or (h, w) list, i.e. (640, 480) |
| `keras`     | `False`         | use Keras for TF SavedModel export                   |
| `optimize`  | `False`         | TorchScript: optimize for mobile                     |
| `half`      | `False`         | FP16 quantization                                    |
| `int8`      | `False`         | INT8 quantization                                    |
| `dynamic`   | `False`         | ONNX/TensorRT: dynamic axes                          |
| `simplify`  | `False`         | ONNX/TensorRT: simplify model                        |
| `opset`     | `None`          | ONNX: opset version (optional, defaults to latest)   |
| `workspace` | `4`             | TensorRT: workspace size (GB)                        |
| `nms`       | `False`         | CoreML: add NMS                                      |

[Export Guide](../modes/export.md){ .md-button }

## Augmentation

Augmentation settings for YOLO models refer to the various transformations and modifications applied to the training data to increase the diversity and size of the dataset. These settings can affect the model's performance, speed, and accuracy. Some common YOLO augmentation settings include the type and intensity of the transformations applied (e.g. random flips, rotations, cropping, color changes), the probability with which each transformation is applied, and the presence of additional features such as masks or multiple labels per box. Other factors that may affect the augmentation process include the size and composition of the original dataset and the specific task the model is being used for. It is important to carefully tune and experiment with these settings to ensure that the augmented dataset is diverse and representative enough to train a high-performing model.

| Key            | Value           | Description                                                                    |
|----------------|-----------------|--------------------------------------------------------------------------------|
| `hsv_h`        | `0.015`         | image HSV-Hue augmentation (fraction)                                          |
| `hsv_s`        | `0.7`           | image HSV-Saturation augmentation (fraction)                                   |
| `hsv_v`        | `0.4`           | image HSV-Value augmentation (fraction)                                        |
| `degrees`      | `0.0`           | image rotation (+/- deg)                                                       |
| `translate`    | `0.1`           | image translation (+/- fraction)                                               |
| `scale`        | `0.5`           | image scale (+/- gain)                                                         |
| `shear`        | `0.0`           | image shear (+/- deg)                                                          |
| `perspective`  | `0.0`           | image perspective (+/- fraction), range 0-0.001                                |
| `flipud`       | `0.0`           | image flip up-down (probability)                                               |
| `fliplr`       | `0.5`           | image flip left-right (probability)                                            |
| `mosaic`       | `1.0`           | image mosaic (probability)                                                     |
| `mixup`        | `0.0`           | image mixup (probability)                                                      |
| `copy_paste`   | `0.0`           | segment copy-paste (probability)                                               |
| `auto_augment` | `'randaugment'` | auto augmentation policy for classification (randaugment, autoaugment, augmix) |
| `erasing`      | `0.4`           | probability o random erasing during classification training (0-1) training     |

## Logging, checkpoints, plotting and file management

Logging, checkpoints, plotting, and file management are important considerations when training a YOLO model.

- Logging: It is often helpful to log various metrics and statistics during training to track the model's progress and diagnose any issues that may arise. This can be done using a logging library such as TensorBoard or by writing log messages to a file.
- Checkpoints: It is a good practice to save checkpoints of the model at regular intervals during training. This allows you to resume training from a previous point if the training process is interrupted or if you want to experiment with different training configurations.
- Plotting: Visualizing the model's performance and training progress can be helpful for understanding how the model is behaving and identifying potential issues. This can be done using a plotting library such as matplotlib or by generating plots using a logging library such as TensorBoard.
- File management: Managing the various files generated during the training process, such as model checkpoints, log files, and plots, can be challenging. It is important to have a clear and organized file structure to keep track of these files and make it easy to access and analyze them as needed.

Effective logging, checkpointing, plotting, and file management can help you keep track of the model's progress and make it easier to debug and optimize the training process.

| Key        | Value    | Description                                                                                    |
|------------|----------|------------------------------------------------------------------------------------------------|
| `project`  | `'runs'` | project name                                                                                   |
| `name`     | `'exp'`  | experiment name. `exp` gets automatically incremented if not specified, i.e, `exp`, `exp2` ... |
| `exist_ok` | `False`  | whether to overwrite existing experiment                                                       |
| `plots`    | `False`  | save plots during train/val                                                                    |
| `save`     | `False`  | save train checkpoints and predict results                                                     |
