---
comments: true
description: Optimize your YOLO model's performance with the right settings and hyperparameters. Learn about training, validation, and prediction configurations.
keywords: YOLO, hyperparameters, configuration, training, validation, prediction, model settings, Ultralytics, performance optimization, machine learning
---

YOLO settings and hyperparameters play a critical role in the model's performance, speed, and [accuracy](https://www.ultralytics.com/glossary/accuracy). These settings and hyperparameters can affect the model's behavior at various stages of the model development process, including training, validation, and prediction.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=87"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Mastering Ultralytics YOLO: Configuration
</p>

Ultralytics commands use the following syntax:

!!! example

    === "CLI"

        ```bash
        yolo TASK MODE ARGS
        ```

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11 model from a pre-trained weights file
        model = YOLO("yolo11n.pt")

        # Run MODE mode using the custom arguments ARGS (guess TASK)
        model.MODE(ARGS)
        ```

Where:

- `TASK` (optional) is one of ([detect](../tasks/detect.md), [segment](../tasks/segment.md), [classify](../tasks/classify.md), [pose](../tasks/pose.md), [obb](../tasks/obb.md))
- `MODE` (required) is one of ([train](../modes/train.md), [val](../modes/val.md), [predict](../modes/predict.md), [export](../modes/export.md), [track](../modes/track.md), [benchmark](../modes/benchmark.md))
- `ARGS` (optional) are `arg=value` pairs like `imgsz=640` that override defaults.

Default `ARG` values are defined on this page from the `cfg/defaults.yaml` [file](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml).

#### Tasks

YOLO models can be used for a variety of tasks, including detection, segmentation, classification and pose. These tasks differ in the type of output they produce and the specific problem they are designed to solve.

- **Detect**: For identifying and localizing objects or regions of interest in an image or video.
- **Segment**: For dividing an image or video into regions or pixels that correspond to different objects or classes.
- **Classify**: For predicting the class label of an input image.
- **Pose**: For identifying objects and estimating their keypoints in an image or video.
- **OBB**: Oriented (i.e. rotated) bounding boxes suitable for satellite or medical imagery.

| Argument | Default    | Description                                                                                                                                                                                                                                                                                                                                                                  |
| -------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task`   | `'detect'` | Specifies the YOLO task to be executed. Options include `detect` for [object detection](https://www.ultralytics.com/glossary/object-detection), `segment` for segmentation, `classify` for classification, `pose` for pose estimation and `obb` for oriented bounding boxes. Each task is tailored to specific types of output and problems within image and video analysis. |

[Tasks Guide](../tasks/index.md){ .md-button }

#### Modes

YOLO models can be used in different modes depending on the specific problem you are trying to solve. These modes include:

- **Train**: For training a YOLO11 model on a custom dataset.
- **Val**: For validating a YOLO11 model after it has been trained.
- **Predict**: For making predictions using a trained YOLO11 model on new images or videos.
- **Export**: For exporting a YOLO11 model to a format that can be used for deployment.
- **Track**: For tracking objects in real-time using a YOLO11 model.
- **Benchmark**: For benchmarking YOLO11 exports (ONNX, TensorRT, etc.) speed and accuracy.

| Argument | Default   | Description                                                                                                                                                                                                                                                                                                                                                                                   |
| -------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mode`   | `'train'` | Specifies the mode in which the YOLO model operates. Options are `train` for model training, `val` for validation, `predict` for inference on new data, `export` for model conversion to deployment formats, `track` for object tracking, and `benchmark` for performance evaluation. Each mode is designed for different stages of the model lifecycle, from development through deployment. |

[Modes Guide](../modes/index.md){ .md-button }

## Train Settings

The training settings for YOLO models encompass various hyperparameters and configurations used during the training process. These settings influence the model's performance, speed, and accuracy. Key training settings include batch size, [learning rate](https://www.ultralytics.com/glossary/learning-rate), momentum, and weight decay. Additionally, the choice of optimizer, [loss function](https://www.ultralytics.com/glossary/loss-function), and training dataset composition can impact the training process. Careful tuning and experimentation with these settings are crucial for optimizing performance.

{% include "macros/train-args.md" %}

!!! info "Note on Batch-size Settings"

    The `batch` argument can be configured in three ways:

    - **Fixed [Batch Size](https://www.ultralytics.com/glossary/batch-size)**: Set an integer value (e.g., `batch=16`), specifying the number of images per batch directly.
    - **Auto Mode (60% GPU Memory)**: Use `batch=-1` to automatically adjust batch size for approximately 60% CUDA memory utilization.
    - **Auto Mode with Utilization Fraction**: Set a fraction value (e.g., `batch=0.70`) to adjust batch size based on the specified fraction of GPU memory usage.

[Train Guide](../modes/train.md){ .md-button }

## Predict Settings

The prediction settings for YOLO models encompass a range of hyperparameters and configurations that influence the model's performance, speed, and accuracy during inference on new data. Careful tuning and experimentation with these settings are essential to achieve optimal performance for a specific task. Key settings include the confidence threshold, Non-Maximum Suppression (NMS) threshold, and the number of classes considered. Additional factors affecting the prediction process are input data size and format, the presence of supplementary features such as masks or multiple labels per box, and the particular task the model is employed for.

Inference arguments:

{% include "macros/predict-args.md" %}

Visualization arguments:

{% include "macros/visualization-args.md" %}

[Predict Guide](../modes/predict.md){ .md-button }

## Validation Settings

The val (validation) settings for YOLO models involve various hyperparameters and configurations used to evaluate the model's performance on a validation dataset. These settings influence the model's performance, speed, and accuracy. Common YOLO validation settings include batch size, validation frequency during training, and performance evaluation metrics. Other factors affecting the validation process include the validation dataset's size and composition, as well as the specific task the model is employed for.

{% include "macros/validation-args.md" %}

Careful tuning and experimentation with these settings are crucial to ensure optimal performance on the validation dataset and detect and prevent [overfitting](https://www.ultralytics.com/glossary/overfitting).

[Val Guide](../modes/val.md){ .md-button }

## Export Settings

Export settings for YOLO models encompass configurations and options related to saving or exporting the model for use in different environments or platforms. These settings can impact the model's performance, size, and compatibility with various systems. Key export settings include the exported model file format (e.g., ONNX, TensorFlow SavedModel), the target device (e.g., CPU, GPU), and additional features such as masks or multiple labels per box. The export process may also be affected by the model's specific task and the requirements or constraints of the destination environment or platform.

{% include "macros/export-args.md" %}

It is crucial to thoughtfully configure these settings to ensure the exported model is optimized for the intended use case and functions effectively in the target environment.

[Export Guide](../modes/export.md){ .md-button }

## Augmentation Settings

Augmentation techniques are essential for improving the robustness and performance of YOLO models by introducing variability into the [training data](https://www.ultralytics.com/glossary/training-data), helping the model generalize better to unseen data. The following table outlines the purpose and effect of each augmentation argument:

{% include "macros/augmentation-args.md" %}

These settings can be adjusted to meet the specific requirements of the dataset and task at hand. Experimenting with different values can help find the optimal augmentation strategy that leads to the best model performance.

## Logging, Checkpoints and Plotting Settings

Logging, checkpoints, plotting, and file management are important considerations when training a YOLO model.

- Logging: It is often helpful to log various metrics and statistics during training to track the model's progress and diagnose any issues that may arise. This can be done using a logging library such as TensorBoard or by writing log messages to a file.
- Checkpoints: It is a good practice to save checkpoints of the model at regular intervals during training. This allows you to resume training from a previous point if the training process is interrupted or if you want to experiment with different training configurations.
- Plotting: Visualizing the model's performance and training progress can be helpful for understanding how the model is behaving and identifying potential issues. This can be done using a plotting library such as matplotlib or by generating plots using a logging library such as TensorBoard.
- File management: Managing the various files generated during the training process, such as model checkpoints, log files, and plots, can be challenging. It is important to have a clear and organized file structure to keep track of these files and make it easy to access and analyze them as needed.

Effective logging, checkpointing, plotting, and file management can help you keep track of the model's progress and make it easier to debug and optimize the training process.

| Argument   | Default  | Description                                                                                                                                                                                                                                                                                                                         |
| ---------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `project`  | `'runs'` | Specifies the root directory for saving training runs. Each run will be saved in a separate subdirectory within this directory.                                                                                                                                                                                                     |
| `name`     | `'exp'`  | Defines the name of the experiment. If not specified, YOLO automatically increments this name for each run, e.g., `exp`, `exp2`, etc., to avoid overwriting previous experiments.                                                                                                                                                   |
| `exist_ok` | `False`  | Determines whether to overwrite an existing experiment directory if one with the same name already exists. Setting this to `True` allows overwriting, while `False` prevents it.                                                                                                                                                    |
| `plots`    | `False`  | Controls the generation and saving of training and validation plots. Set to `True` to create plots such as loss curves, [precision](https://www.ultralytics.com/glossary/precision)-[recall](https://www.ultralytics.com/glossary/recall) curves, and sample predictions. Useful for visually tracking model performance over time. |
| `save`     | `False`  | Enables the saving of training checkpoints and final model weights. Set to `True` to periodically save model states, allowing training to be resumed from these checkpoints or models to be deployed.                                                                                                                               |

## FAQ

### How do I improve the performance of my YOLO model during training?

Improving YOLO model performance involves tuning hyperparameters like batch size, learning rate, momentum, and weight decay. Adjusting augmentation settings, selecting the right optimizer, and employing techniques like early stopping or [mixed precision](https://www.ultralytics.com/glossary/mixed-precision) can also help. For detailed guidance on training settings, refer to the [Train Guide](../modes/train.md).

### What are the key hyperparameters to consider for YOLO model accuracy?

Key hyperparameters affecting YOLO model accuracy include:

- **Batch Size (`batch`)**: Larger batch sizes can stabilize training but may require more memory.
- **Learning Rate (`lr0`)**: Controls the step size for weight updates; smaller rates offer fine adjustments but slow convergence.
- **Momentum (`momentum`)**: Helps accelerate gradient vectors in the right directions, dampening oscillations.
- **Image Size (`imgsz`)**: Larger image sizes can improve accuracy but increase computational load.

Adjust these values based on your dataset and hardware capabilities. Explore more in the [Train Settings](#train-settings) section.

### How do I set the learning rate for training a YOLO model?

The learning rate (`lr0`) is crucial for optimization. A common starting point is `0.01` for SGD or `0.001` for Adam. It's essential to monitor training metrics and adjust if necessary. Use cosine learning rate schedulers (`cos_lr`) or warmup techniques (`warmup_epochs`, `warmup_momentum`) to dynamically modify the rate during training. Find more details in the [Train Guide](../modes/train.md).

### What are the default inference settings for YOLO models?

Default inference settings include:

- **Confidence Threshold (`conf=0.25`)**: Minimum confidence for detections.
- **IoU Threshold (`iou=0.7`)**: For Non-Maximum Suppression (NMS).
- **Image Size (`imgsz=640`)**: Resizes input images prior to inference.
- **Device (`device=None`)**: Selects CPU or GPU for inference.

For a comprehensive overview, visit the [Predict Settings](#predict-settings) section and the [Predict Guide](../modes/predict.md).

### Why should I use mixed precision training with YOLO models?

Mixed precision training, enabled with `amp=True`, helps reduce memory usage and can speed up training by utilizing the advantages of both FP16 and FP32. This is beneficial for modern GPUs, which support mixed precision natively, allowing more models to fit in memory and enable faster computations without significant loss in accuracy. Learn more about this in the [Train Guide](../modes/train.md).
