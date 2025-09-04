---
comments: true
description: Optimize your Ultralytics YOLO model's performance with the right settings and hyperparameters. Learn about training, validation, and prediction configurations.
keywords: YOLO, hyperparameters, configuration, training, validation, prediction, model settings, Ultralytics, performance optimization, machine learning
---

# Configuration

YOLO settings and hyperparameters play a critical role in the model's performance, speed, and [accuracy](https://www.ultralytics.com/glossary/accuracy). These settings can affect the model's behavior at various stages, including training, validation, and prediction.

**Watch:** Mastering Ultralytics YOLO: Configuration

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

        # Load a YOLO model from a pre-trained weights file
        model = YOLO("yolo11n.pt")

        # Run MODE mode using the custom arguments ARGS (guess TASK)
        model.MODE(ARGS)
        ```

Where:

- `TASK` (optional) is one of ([detect](../tasks/detect.md), [segment](../tasks/segment.md), [classify](../tasks/classify.md), [pose](../tasks/pose.md), [obb](../tasks/obb.md))
- `MODE` (required) is one of ([train](../modes/train.md), [val](../modes/val.md), [predict](../modes/predict.md), [export](../modes/export.md), [track](../modes/track.md), [benchmark](../modes/benchmark.md))
- `ARGS` (optional) are `arg=value` pairs like `imgsz=640` that override defaults.

Default `ARG` values are defined on this page and come from the `cfg/defaults.yaml` [file](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml).

## Tasks

Ultralytics YOLO models can perform a variety of computer vision tasks, including:

- **Detect**: [Object detection](https://docs.ultralytics.com/tasks/detect/) identifies and localizes objects within an image or video.
- **Segment**: [Instance segmentation](https://docs.ultralytics.com/tasks/segment/) divides an image or video into regions corresponding to different objects or classes.
- **Classify**: [Image classification](https://docs.ultralytics.com/tasks/classify/) predicts the class label of an input image.
- **Pose**: [Pose estimation](https://docs.ultralytics.com/tasks/pose/) identifies objects and estimates their keypoints in an image or video.
- **OBB**: [Oriented Bounding Boxes](https://docs.ultralytics.com/tasks/obb/) uses rotated bounding boxes, suitable for satellite or medical imagery.

| Argument | Default    | Description                                                                                                                                                                                                                                                                                                                        |
| -------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task`   | `'detect'` | Specifies the YOLO task: `detect` for [object detection](https://www.ultralytics.com/glossary/object-detection), `segment` for segmentation, `classify` for classification, `pose` for pose estimation, and `obb` for oriented bounding boxes. Each task is tailored to specific outputs and problems in image and video analysis. |

[Tasks Guide](../tasks/index.md){ .md-button }

## Modes

Ultralytics YOLO models operate in different modes, each designed for a specific stage of the model lifecycle:

- **Train**: Train a YOLO model on a custom dataset.
- **Val**: Validate a trained YOLO model.
- **Predict**: Use a trained YOLO model to make predictions on new images or videos.
- **Export**: Export a YOLO model for deployment.
- **Track**: Track objects in real-time using a YOLO model.
- **Benchmark**: Benchmark the speed and accuracy of YOLO exports (ONNX, TensorRT, etc.).

| Argument | Default   | Description                                                                                                                                                                                                                                                                                                        |
| -------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mode`   | `'train'` | Specifies the YOLO model's operating mode: `train` for model training, `val` for validation, `predict` for inference, `export` for converting to deployment formats, `track` for object tracking, and `benchmark` for performance evaluation. Each mode supports different stages, from development to deployment. |

[Modes Guide](../modes/index.md){ .md-button }

## Train Settings

Training settings for YOLO models include hyperparameters and configurations that affect the model's performance, speed, and [accuracy](https://www.ultralytics.com/glossary/accuracy). Key settings include [batch size](https://www.ultralytics.com/glossary/batch-size), [learning rate](https://www.ultralytics.com/glossary/learning-rate), momentum, and weight decay. The choice of optimizer, [loss function](https://www.ultralytics.com/glossary/loss-function), and dataset composition also impact training. Tuning and experimentation are crucial for optimal performance. For more details, see the [Ultralytics entrypoint function](../reference/cfg/__init__.md).

{% include "macros/train-args.md" %}

!!! info "Note on Batch-size Settings"

    The `batch` argument offers three configuration options:

    - **Fixed Batch Size**: Specify the number of images per batch with an integer (e.g., `batch=16`).
    - **Auto Mode (60% GPU Memory)**: Use `batch=-1` for automatic adjustment to approximately 60% CUDA memory utilization.
    - **Auto Mode with Utilization Fraction**: Set a fraction (e.g., `batch=0.70`) to adjust based on a specified GPU memory usage.

[Train Guide](../modes/train.md){ .md-button }

## Predict Settings

Prediction settings for YOLO models include hyperparameters and configurations that influence performance, speed, and [accuracy](https://www.ultralytics.com/glossary/accuracy) during inference. Key settings include the confidence threshold, [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) threshold, and the number of classes. Input data size, format, and supplementary features like masks also affect predictions. Tuning these settings is essential for optimal performance.

Inference arguments:

{% include "macros/predict-args.md" %}

Visualization arguments:

{% from "macros/visualization-args.md" import param_table %} {{ param_table() }}

[Predict Guide](../modes/predict.md){ .md-button }

## Validation Settings

Validation settings for YOLO models involve hyperparameters and configurations to evaluate performance on a [validation dataset](https://www.ultralytics.com/glossary/validation-data). These settings influence performance, speed, and [accuracy](https://www.ultralytics.com/glossary/accuracy). Common settings include batch size, validation frequency, and performance metrics. The validation dataset's size and composition, along with the specific task, also affect the process.

{% include "macros/validation-args.md" %}

Careful tuning and experimentation are crucial to ensure optimal performance and to detect and prevent [overfitting](https://www.ultralytics.com/glossary/overfitting).

[Val Guide](../modes/val.md){ .md-button }

## Export Settings

Export settings for YOLO models include configurations for saving or exporting the model for use in different environments. These settings impact performance, size, and compatibility. Key settings include the exported file format (e.g., ONNX, TensorFlow SavedModel), the target device (e.g., CPU, GPU), and features like masks. The model's task and the destination environment's constraints also affect the export process.

{% include "macros/export-args.md" %}

Thoughtful configuration ensures the exported model is optimized for its use case and functions effectively in the target environment.

[Export Guide](../modes/export.md){ .md-button }

## Solutions Settings

Ultralytics Solutions configuration settings offer flexibility to customize models for tasks like object counting, heatmap creation, workout tracking, data analysis, zone tracking, queue management, and region-based counting. These options allow easy adjustments for accurate and useful results tailored to specific needs.

{% from "macros/solutions-args.md" import param_table %} {{ param_table() }}

[Solutions Guide](../solutions/index.md){ .md-button }

## Augmentation Settings

[Data augmentation](https://www.ultralytics.com/glossary/data-augmentation) techniques are essential for improving YOLO model robustness and performance by introducing variability into the [training data](https://www.ultralytics.com/glossary/training-data), helping the model generalize better to unseen data. The following table outlines each augmentation argument's purpose and effect:

{% include "macros/augmentation-args.md" %}

Adjust these settings to meet dataset and task requirements. Experimenting with different values can help find the optimal augmentation strategy for the best model performance.

[Augmentation Guide](../guides/yolo-data-augmentation.md){ .md-button }

## Logging, Checkpoints and Plotting Settings

Logging, checkpoints, plotting, and file management are important when training a YOLO model:

- **Logging**: Track the model's progress and diagnose issues using libraries like [TensorBoard](https://docs.ultralytics.com/integrations/tensorboard/) or by writing to a file.
- **Checkpoints**: Save the model at regular intervals to resume training or experiment with different configurations.
- **Plotting**: Visualize performance and training progress using libraries like matplotlib or TensorBoard.
- **File management**: Organize files generated during training, such as checkpoints, log files, and plots, for easy access and analysis.

Effective management of these aspects helps track progress and makes debugging and optimization easier.

| Argument   | Default  | Description                                                                                                                                                                                                                                                                                               |
| ---------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `project`  | `'runs'` | Specifies the root directory for saving training runs. Each run is saved in a separate subdirectory.                                                                                                                                                                                                      |
| `name`     | `'exp'`  | Defines the experiment name. If unspecified, YOLO increments this name for each run (e.g., `exp`, `exp2`) to avoid overwriting.                                                                                                                                                                           |
| `exist_ok` | `False`  | Determines whether to overwrite an existing experiment directory. `True` allows overwriting; `False` prevents it.                                                                                                                                                                                         |
| `plots`    | `False`  | Controls the generation and saving of training and validation plots. Set to `True` to create plots like loss curves, [precision](https://www.ultralytics.com/glossary/precision)-[recall](https://www.ultralytics.com/glossary/recall) curves, and sample predictions for visual tracking of performance. |
| `save`     | `False`  | Enables saving training checkpoints and final model weights. Set to `True` to save model states periodically, allowing training resumption or model deployment.                                                                                                                                           |

## FAQ

### How do I improve my YOLO model's performance during training?

Improve performance by tuning hyperparameters like [batch size](https://www.ultralytics.com/glossary/batch-size), [learning rate](https://www.ultralytics.com/glossary/learning-rate), momentum, and weight decay. Adjust [data augmentation](https://www.ultralytics.com/glossary/data-augmentation) settings, select the right optimizer, and use techniques like early stopping or [mixed precision](https://www.ultralytics.com/glossary/mixed-precision). For details, see the [Train Guide](../modes/train.md).

### What are the key hyperparameters for YOLO model accuracy?

Key hyperparameters affecting accuracy include:

- **Batch Size (`batch`)**: Larger sizes can stabilize training but need more memory.
- **Learning Rate (`lr0`)**: Smaller rates offer fine adjustments but slower convergence.
- **Momentum (`momentum`)**: Accelerates gradient vectors, dampening oscillations.
- **Image Size (`imgsz`)**: Larger sizes improve accuracy but increase computational load.

Adjust these based on your dataset and hardware. Learn more in [Train Settings](#train-settings).

### How do I set the learning rate for training a YOLO model?

The learning rate (`lr0`) is crucial; start with `0.01` for SGD or `0.001` for [Adam optimizer](https://www.ultralytics.com/glossary/adam-optimizer). Monitor metrics and adjust as needed. Use cosine learning rate schedulers (`cos_lr`) or warmup (`warmup_epochs`, `warmup_momentum`). Details are in the [Train Guide](../modes/train.md).

### What are the default inference settings for YOLO models?

Default settings include:

- **Confidence Threshold (`conf=0.25`)**: Minimum confidence for detections.
- **IoU Threshold (`iou=0.7`)**: For [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms).
- **Image Size (`imgsz=640`)**: Resizes input images.
- **Device (`device=None`)**: Selects CPU or GPU.

For a full overview, see [Predict Settings](#predict-settings) and the [Predict Guide](../modes/predict.md).

### Why use mixed precision training with YOLO models?

[Mixed precision](https://www.ultralytics.com/glossary/mixed-precision) training (`amp=True`) reduces memory usage and speeds up training using FP16 and FP32. It's beneficial for modern GPUs, allowing larger models and faster computations without significant accuracy loss. Learn more in the [Train Guide](../modes/train.md).
