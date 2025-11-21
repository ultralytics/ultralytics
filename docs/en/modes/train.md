---
comments: true
description: Learn how to efficiently train object detection models using YOLO11 with comprehensive instructions on settings, augmentation, and hardware utilization.
keywords: Ultralytics, YOLO11, model training, deep learning, object detection, GPU training, dataset augmentation, hyperparameter tuning, model performance, apple silicon training
---

# Model Training with Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

Training a [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) model involves feeding it data and adjusting its parameters so that it can make accurate predictions. Train mode in Ultralytics YOLO11 is engineered for effective and efficient training of object detection models, fully utilizing modern hardware capabilities. This guide aims to cover all the details you need to get started with training your own models using YOLO11's robust set of features.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train a YOLO model on Your Custom Dataset in Google Colab.
</p>

## Why Choose Ultralytics YOLO for Training?

Here are some compelling reasons to opt for YOLO11's Train mode:

- **Efficiency:** Make the most out of your hardware, whether you're on a single-GPU setup or scaling across multiple GPUs.
- **Versatility:** Train on custom datasets in addition to readily available ones like COCO, VOC, and ImageNet.
- **User-Friendly:** Simple yet powerful CLI and Python interfaces for a straightforward training experience.
- **Hyperparameter Flexibility:** A broad range of customizable hyperparameters to fine-tune model performance.

### Key Features of Train Mode

The following are some notable features of YOLO11's Train mode:

- **Automatic Dataset Download:** Standard datasets like COCO, VOC, and ImageNet are downloaded automatically on first use.
- **Multi-GPU Support:** Scale your training efforts seamlessly across multiple GPUs to expedite the process.
- **Hyperparameter Configuration:** The option to modify hyperparameters through YAML configuration files or CLI arguments.
- **Visualization and Monitoring:** Real-time tracking of training metrics and visualization of the learning process for better insights.

!!! tip

    * YOLO11 datasets like COCO, VOC, ImageNet and many others automatically download on first use, i.e. `yolo train data=coco.yaml`

## Usage Examples

Train YOLO11n on the COCO8 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 640. The training device can be specified using the `device` argument. If no argument is passed GPU `device=0` will be used if available, otherwise `device='cpu'` will be used. See Arguments section below for a full list of training arguments.

!!! warning "Windows Multi-Processing Error"

    On Windows, you may receive a `RuntimeError` when launching the training as a script. Add a `if __name__ == "__main__":` block before your training code to resolve it.

!!! example "Single-GPU and CPU Training Example"

    Device is determined automatically. If a GPU is available then it will be used (default CUDA device 0), otherwise training will start on CPU.

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.yaml")  # build a new model from YAML
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo detect train data=coco8.yaml model=yolo11n.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo detect train data=coco8.yaml model=yolo11n.yaml pretrained=yolo11n.pt epochs=100 imgsz=640
        ```

### Multi-GPU Training

Multi-GPU training allows for more efficient utilization of available hardware resources by distributing the training load across multiple GPUs. This feature is available through both the Python API and the command-line interface. To enable multi-GPU training, specify the GPU device IDs you wish to use.

!!! example "Multi-GPU Training Example"

    To train with 2 GPUs, CUDA devices 0 and 1 use the following commands. Expand to additional GPUs as required.

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Train the model with 2 GPUs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[0, 1])

        # Train the model with the two most idle GPUs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[-1, -1])
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model using GPUs 0 and 1
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640 device=0,1

        # Use the two most idle GPUs
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640 device=-1,-1
        ```

### Idle GPU Training

Idle GPU Training enables automatic selection of the least utilized GPUs in multi-GPU systems, optimizing resource usage without manual GPU selection. This feature identifies available GPUs based on utilization metrics and VRAM availability.

!!! example "Idle GPU Training Example"

    To automatically select and use the most idle GPU(s) for training, use the `-1` device parameter. This is particularly useful in shared computing environments or servers with multiple users.

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Train using the single most idle GPU
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=-1)

        # Train using the two most idle GPUs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[-1, -1])
        ```

    === "CLI"

        ```bash
        # Start training using the single most idle GPU
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640 device=-1

        # Start training using the two most idle GPUs
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640 device=-1,-1
        ```

The auto-selection algorithm prioritizes GPUs with:

1. Lower current utilization percentages
2. Higher available memory (free VRAM)
3. Lower temperature and power consumption

This feature is especially valuable in shared computing environments or when running multiple training jobs across different models. It automatically adapts to changing system conditions, ensuring optimal resource allocation without manual intervention.

### Apple Silicon MPS Training

With the support for Apple silicon chips integrated in the Ultralytics YOLO models, it's now possible to train your models on devices utilizing the powerful Metal Performance Shaders (MPS) framework. The MPS offers a high-performance way of executing computation and image processing tasks on Apple's custom silicon.

To enable training on Apple silicon chips, you should specify 'mps' as your device when initiating the training process. Below is an example of how you could do this in Python and via the command line:

!!! example "MPS Training Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Train the model with MPS
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device="mps")
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model using MPS
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640 device=mps
        ```

While leveraging the computational power of the Apple silicon chips, this enables more efficient processing of the training tasks. For more detailed guidance and advanced configuration options, please refer to the [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).

### Resuming Interrupted Trainings

Resuming training from a previously saved state is a crucial feature when working with deep learning models. This can come in handy in various scenarios, like when the training process has been unexpectedly interrupted, or when you wish to continue training a model with new data or for more epochs.

When training is resumed, Ultralytics YOLO loads the weights from the last saved model and also restores the optimizer state, [learning rate](https://www.ultralytics.com/glossary/learning-rate) scheduler, and the epoch number. This allows you to continue the training process seamlessly from where it was left off.

You can easily resume training in Ultralytics YOLO by setting the `resume` argument to `True` when calling the `train` method, and specifying the path to the `.pt` file containing the partially trained model weights.

Below is an example of how to resume an interrupted training using Python and via the command line:

!!! example "Resume Training Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/last.pt")  # load a partially trained model

        # Resume training
        results = model.train(resume=True)
        ```

    === "CLI"

        ```bash
        # Resume an interrupted training
        yolo train resume model=path/to/last.pt
        ```

By setting `resume=True`, the `train` function will continue training from where it left off, using the state stored in the 'path/to/last.pt' file. If the `resume` argument is omitted or set to `False`, the `train` function will start a new training session.

Remember that checkpoints are saved at the end of every epoch by default, or at fixed intervals using the `save_period` argument, so you must complete at least 1 epoch to resume a training run.

## Train Settings

The training settings for YOLO models encompass various hyperparameters and configurations used during the training process. These settings influence the model's performance, speed, and [accuracy](https://www.ultralytics.com/glossary/accuracy). Key training settings include batch size, learning rate, momentum, and weight decay. Additionally, the choice of optimizer, [loss function](https://www.ultralytics.com/glossary/loss-function), and training dataset composition can impact the training process. Careful tuning and experimentation with these settings are crucial for optimizing performance.

| Argument          | Type                     | Default  | Description                                                                                                                                                                                                                                                                             |
| ----------------- | ------------------------ | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`           | `str`                    | `None`   | Specifies the model file for training. Accepts a path to either a `.pt` pretrained model or a `.yaml` configuration file. Essential for defining the model structure or initializing weights.                                                                                           |
| `data`            | `str`                    | `None`   | Path to the dataset configuration file (e.g., `coco8.yaml`). This file contains dataset-specific parameters, including paths to training and [validation data](https://www.ultralytics.com/glossary/validation-data), class names, and number of classes.                               |
| `epochs`          | `int`                    | `100`    | Total number of training epochs. Each [epoch](https://www.ultralytics.com/glossary/epoch) represents a full pass over the entire dataset. Adjusting this value can affect training duration and model performance.                                                                      |
| `time`            | `float`                  | `None`   | Maximum training time in hours. If set, this overrides the `epochs` argument, allowing training to automatically stop after the specified duration. Useful for time-constrained training scenarios.                                                                                     |
| `patience`        | `int`                    | `100`    | Number of epochs to wait without improvement in validation metrics before early stopping the training. Helps prevent [overfitting](https://www.ultralytics.com/glossary/overfitting) by stopping training when performance plateaus.                                                    |
| `batch`           | `int` or `float`         | `16`     | [Batch size](https://www.ultralytics.com/glossary/batch-size), with three modes: set as an integer (e.g., `batch=16`), auto mode for 60% GPU memory utilization (`batch=-1`), or auto mode with specified utilization fraction (`batch=0.70`).                                          |
| `imgsz`           | `int`                    | `640`    | Target image size for training. Images are resized to squares with sides equal to the specified value (if `rect=False`), preserving aspect ratio for YOLO models but not RT-DETR. Affects model [accuracy](https://www.ultralytics.com/glossary/accuracy) and computational complexity. |
| `save`            | `bool`                   | `True`   | Enables saving of training checkpoints and final model weights. Useful for resuming training or [model deployment](https://www.ultralytics.com/glossary/model-deployment).                                                                                                              |
| `save_period`     | `int`                    | `-1`     | Frequency of saving model checkpoints, specified in epochs. A value of -1 disables this feature. Useful for saving interim models during long training sessions.                                                                                                                        |
| `cache`           | `bool`                   | `False`  | Enables caching of dataset images in memory (`True`/`ram`), on disk (`disk`), or disables it (`False`). Improves training speed by reducing disk I/O at the cost of increased memory usage.                                                                                             |
| `device`          | `int` or `str` or `list` | `None`   | Specifies the computational device(s) for training: a single GPU (`device=0`), multiple GPUs (`device=[0,1]`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`), or auto-selection of most idle GPU (`device=-1`) or multiple idle GPUs (`device=[-1,-1]`)                      |
| `workers`         | `int`                    | `8`      | Number of worker threads for data loading (per `RANK` if Multi-GPU training). Influences the speed of data preprocessing and feeding into the model, especially useful in multi-GPU setups.                                                                                             |
| `project`         | `str`                    | `None`   | Name of the project directory where training outputs are saved. Allows for organized storage of different experiments.                                                                                                                                                                  |
| `name`            | `str`                    | `None`   | Name of the training run. Used for creating a subdirectory within the project folder, where training logs and outputs are stored.                                                                                                                                                       |
| `exist_ok`        | `bool`                   | `False`  | If True, allows overwriting of an existing project/name directory. Useful for iterative experimentation without needing to manually clear previous outputs.                                                                                                                             |
| `pretrained`      | `bool` or `str`          | `True`   | Determines whether to start training from a pretrained model. Can be a boolean value or a string path to a specific model from which to load weights. Enhances training efficiency and model performance.                                                                               |
| `optimizer`       | `str`                    | `'auto'` | Choice of optimizer for training. Options include `SGD`, `Adam`, `AdamW`, `NAdam`, `RAdam`, `RMSProp` etc., or `auto` for automatic selection based on model configuration. Affects convergence speed and stability.                                                                    |
| `seed`            | `int`                    | `0`      | Sets the random seed for training, ensuring reproducibility of results across runs with the same configurations.                                                                                                                                                                        |
| `deterministic`   | `bool`                   | `True`   | Forces deterministic algorithm use, ensuring reproducibility but may affect performance and speed due to the restriction on non-deterministic algorithms.                                                                                                                               |
| `single_cls`      | `bool`                   | `False`  | Treats all classes in multi-class datasets as a single class during training. Useful for binary classification tasks or when focusing on object presence rather than classification.                                                                                                    |
| `classes`         | `list[int]`              | `None`   | Specifies a list of class IDs to train on. Useful for filtering out and focusing only on certain classes during training.                                                                                                                                                               |
| `rect`            | `bool`                   | `False`  | Enables minimum padding strategy—images in a batch are minimally padded to reach a common size, with the longest side equal to `imgsz`. Can improve efficiency and speed but may affect model accuracy.                                                                                 |
| `multi_scale`     | `bool`                   | `False`  | Enables multi-scale training by increasing/decreasing `imgsz` by up to a factor of `0.5` during training. Trains the model to be more accurate with multiple `imgsz` during inference.                                                                                                  |
| `cos_lr`          | `bool`                   | `False`  | Utilizes a cosine [learning rate](https://www.ultralytics.com/glossary/learning-rate) scheduler, adjusting the learning rate following a cosine curve over epochs. Helps in managing learning rate for better convergence.                                                              |
| `close_mosaic`    | `int`                    | `10`     | Disables mosaic [data augmentation](https://www.ultralytics.com/glossary/data-augmentation) in the last N epochs to stabilize training before completion. Setting to 0 disables this feature.                                                                                           |
| `resume`          | `bool`                   | `False`  | Resumes training from the last saved checkpoint. Automatically loads model weights, optimizer state, and epoch count, continuing training seamlessly.                                                                                                                                   |
| `amp`             | `bool`                   | `True`   | Enables Automatic [Mixed Precision](https://www.ultralytics.com/glossary/mixed-precision) (AMP) training, reducing memory usage and possibly speeding up training with minimal impact on accuracy.                                                                                      |
| `fraction`        | `float`                  | `1.0`    | Specifies the fraction of the dataset to use for training. Allows for training on a subset of the full dataset, useful for experiments or when resources are limited.                                                                                                                   |
| `profile`         | `bool`                   | `False`  | Enables profiling of ONNX and TensorRT speeds during training, useful for optimizing model deployment.                                                                                                                                                                                  |
| `freeze`          | `int` or `list`          | `None`   | Freezes the first N layers of the model or specified layers by index, reducing the number of trainable parameters. Useful for fine-tuning or [transfer learning](https://www.ultralytics.com/glossary/transfer-learning).                                                               |
| `lr0`             | `float`                  | `0.01`   | Initial learning rate (i.e. `SGD=1E-2`, `Adam=1E-3`). Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.                                                                                                                  |
| `lrf`             | `float`                  | `0.01`   | Final learning rate as a fraction of the initial rate = (`lr0 * lrf`), used in conjunction with schedulers to adjust the learning rate over time.                                                                                                                                       |
| `momentum`        | `float`                  | `0.937`  | Momentum factor for SGD or beta1 for [Adam optimizers](https://www.ultralytics.com/glossary/adam-optimizer), influencing the incorporation of past gradients in the current update.                                                                                                     |
| `weight_decay`    | `float`                  | `0.0005` | L2 [regularization](https://www.ultralytics.com/glossary/regularization) term, penalizing large weights to prevent overfitting.                                                                                                                                                         |
| `warmup_epochs`   | `float`                  | `3.0`    | Number of epochs for learning rate warmup, gradually increasing the learning rate from a low value to the initial learning rate to stabilize training early on.                                                                                                                         |
| `warmup_momentum` | `float`                  | `0.8`    | Initial momentum for warmup phase, gradually adjusting to the set momentum over the warmup period.                                                                                                                                                                                      |
| `warmup_bias_lr`  | `float`                  | `0.1`    | Learning rate for bias parameters during the warmup phase, helping stabilize model training in the initial epochs.                                                                                                                                                                      |
| `box`             | `float`                  | `7.5`    | Weight of the box loss component in the [loss function](https://www.ultralytics.com/glossary/loss-function), influencing how much emphasis is placed on accurately predicting [bounding box](https://www.ultralytics.com/glossary/bounding-box) coordinates.                            |
| `cls`             | `float`                  | `0.5`    | Weight of the classification loss in the total loss function, affecting the importance of correct class prediction relative to other components.                                                                                                                                        |
| `dfl`             | `float`                  | `1.5`    | Weight of the distribution focal loss, used in certain YOLO versions for fine-grained classification.                                                                                                                                                                                   |
| `pose`            | `float`                  | `12.0`   | Weight of the pose loss in models trained for pose estimation, influencing the emphasis on accurately predicting pose keypoints.                                                                                                                                                        |
| `kobj`            | `float`                  | `2.0`    | Weight of the keypoint objectness loss in pose estimation models, balancing detection confidence with pose accuracy.                                                                                                                                                                    |
| `nbs`             | `int`                    | `64`     | Nominal batch size for normalization of loss.                                                                                                                                                                                                                                           |
| `overlap_mask`    | `bool`                   | `True`   | Determines whether object masks should be merged into a single mask for training, or kept separate for each object. In case of overlap, the smaller mask is overlaid on top of the larger mask during merge.                                                                            |
| `mask_ratio`      | `int`                    | `4`      | Downsample ratio for segmentation masks, affecting the resolution of masks used during training.                                                                                                                                                                                        |
| `dropout`         | `float`                  | `0.0`    | Dropout rate for regularization in classification tasks, preventing overfitting by randomly omitting units during training.                                                                                                                                                             |
| `val`             | `bool`                   | `True`   | Enables validation during training, allowing for periodic evaluation of model performance on a separate dataset.                                                                                                                                                                        |
| `plots`           | `bool`                   | `False`  | Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression.                                                                                                                |
| `compile`         | `bool` or `str`          | `False`  | Enables PyTorch 2.x `torch.compile` graph compilation with `backend='inductor'`. Accepts `True` → `"default"`, `False` → disables, or a string mode such as `"default"`, `"reduce-overhead"`, `"max-autotune-no-cudagraphs"`. Falls back to eager with a warning if unsupported.        |

!!! info "Note on Batch-size Settings"

    The `batch` argument can be configured in three ways:

    - **Fixed [Batch Size](https://www.ultralytics.com/glossary/batch-size)**: Set an integer value (e.g., `batch=16`), specifying the number of images per batch directly.
    - **Auto Mode (60% GPU Memory)**: Use `batch=-1` to automatically adjust batch size for approximately 60% CUDA memory utilization.
    - **Auto Mode with Utilization Fraction**: Set a fraction value (e.g., `batch=0.70`) to adjust batch size based on the specified fraction of GPU memory usage.

## Augmentation Settings and Hyperparameters

Augmentation techniques are essential for improving the robustness and performance of YOLO models by introducing variability into the [training data](https://www.ultralytics.com/glossary/training-data), helping the model generalize better to unseen data. The following table outlines the purpose and effect of each augmentation argument:

| Argument                                                                                               | Type    | Default                 | Supported Tasks                                | Range         | Description                                                                                                                                                    |
| ------------------------------------------------------------------------------------------------------ | ------- | ----------------------- | ---------------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`hsv_h`](../guides/yolo-data-augmentation.md/#hue-adjustment-hsv_h)                                   | `float` | `0.015`           | `detect`, `segment`, `pose`, `obb`, `classify` | `0.0 - 1.0`   | Adjusts the hue of the image by a fraction of the color wheel, introducing color variability. Helps the model generalize across different lighting conditions. |
| [`hsv_s`](../guides/yolo-data-augmentation.md/#saturation-adjustment-hsv_s)                            | `float` | `0.7`           | `detect`, `segment`, `pose`, `obb`, `classify` | `0.0 - 1.0`   | Alters the saturation of the image by a fraction, affecting the intensity of colors. Useful for simulating different environmental conditions.                 |
| [`hsv_v`](../guides/yolo-data-augmentation.md/#brightness-adjustment-hsv_v)                            | `float` | `0.4`           | `detect`, `segment`, `pose`, `obb`, `classify` | `0.0 - 1.0`   | Modifies the value (brightness) of the image by a fraction, helping the model to perform well under various lighting conditions.                               |
| [`degrees`](../guides/yolo-data-augmentation.md/#rotation-degrees)                                     | `float` | `0.0`         | `detect`, `segment`, `pose`, `obb`             | `0.0 - 180`   | Rotates the image randomly within the specified degree range, improving the model's ability to recognize objects at various orientations.                      |
| [`translate`](../guides/yolo-data-augmentation.md/#translation-translate)                              | `float` | `0.1`       | `detect`, `segment`, `pose`, `obb`             | `0.0 - 1.0`   | Translates the image horizontally and vertically by a fraction of the image size, aiding in learning to detect partially visible objects.                      |
| [`scale`](../guides/yolo-data-augmentation.md/#scale-scale)                                            | `float` | `0.5`           | `detect`, `segment`, `pose`, `obb`, `classify` | `>=0.0`       | Scales the image by a gain factor, simulating objects at different distances from the camera.                                                                  |
| [`shear`](../guides/yolo-data-augmentation.md/#shear-shear)                                            | `float` | `0.0`           | `detect`, `segment`, `pose`, `obb`             | `-180 - +180` | Shears the image by a specified degree, mimicking the effect of objects being viewed from different angles.                                                    |
| [`perspective`](../guides/yolo-data-augmentation.md/#perspective-perspective)                          | `float` | `0.0`     | `detect`, `segment`, `pose`, `obb`             | `0.0 - 0.001` | Applies a random perspective transformation to the image, enhancing the model's ability to understand objects in 3D space.                                     |
| [`flipud`](../guides/yolo-data-augmentation.md/#flip-up-down-flipud)                                   | `float` | `0.0`          | `detect`, `segment`, `pose`, `obb`, `classify` | `0.0 - 1.0`   | Flips the image upside down with the specified probability, increasing the data variability without affecting the object's characteristics.                    |
| [`fliplr`](../guides/yolo-data-augmentation.md/#flip-left-right-fliplr)                                | `float` | `0.5`          | `detect`, `segment`, `pose`, `obb`, `classify` | `0.0 - 1.0`   | Flips the image left to right with the specified probability, useful for learning symmetrical objects and increasing dataset diversity.                        |
| [`bgr`](../guides/yolo-data-augmentation.md/#bgr-channel-swap-bgr)                                     | `float` | `0.0`             | `detect`, `segment`, `pose`, `obb`             | `0.0 - 1.0`   | Flips the image channels from RGB to BGR with the specified probability, useful for increasing robustness to incorrect channel ordering.                       |
| [`mosaic`](../guides/yolo-data-augmentation.md/#mosaic-mosaic)                                         | `float` | `1.0`          | `detect`, `segment`, `pose`, `obb`             | `0.0 - 1.0`   | Combines four training images into one, simulating different scene compositions and object interactions. Highly effective for complex scene understanding.     |
| [`mixup`](../guides/yolo-data-augmentation.md/#mixup-mixup)                                            | `float` | `0.0`           | `detect`, `segment`, `pose`, `obb`             | `0.0 - 1.0`   | Blends two images and their labels, creating a composite image. Enhances the model's ability to generalize by introducing label noise and visual variability.  |
| [`cutmix`](../guides/yolo-data-augmentation.md/#cutmix-cutmix)                                         | `float` | `0.0`          | `detect`, `segment`, `pose`, `obb`             | `0.0 - 1.0`   | Combines portions of two images, creating a partial blend while maintaining distinct regions. Enhances model robustness by creating occlusion scenarios.       |
| [`copy_paste`](../guides/yolo-data-augmentation.md/#copy-paste-copy_paste)                             | `float` | `0.0`      | `segment`                                      | `0.0 - 1.0`   | Copies and pastes objects across images to increase object instances.                                                                                          |
| [`copy_paste_mode`](../guides/yolo-data-augmentation.md/#copy-paste-mode-copy_paste_mode)              | `str`   | `flip` | `segment`                                      | -             | Specifies the `copy-paste` strategy to use. Options include `'flip'` and `'mixup'`.                                                                            |
| [`auto_augment`](../guides/yolo-data-augmentation.md/#auto-augment-auto_augment)                       | `str`   | `randaugment`    | `classify`                                     | -             | Applies a predefined augmentation policy (`'randaugment'`, `'autoaugment'`, or `'augmix'`) to enhance model performance through visual diversity.              |
| [`erasing`](../guides/yolo-data-augmentation.md/#random-erasing-erasing)                               | `float` | `0.4`         | `classify`                                     | `0.0 - 0.9`   | Randomly erases regions of the image during training to encourage the model to focus on less obvious features.                                                 |
| [`augmentations`](../guides/yolo-data-augmentation.md/#custom-albumentations-transforms-augmentations) | `list`  | ``   | `detect`, `segment`, `pose`, `obb`             | -             | Custom Albumentations transforms for advanced data augmentation (Python API only). Accepts a list of transform objects for specialized augmentation needs.     |

These settings can be adjusted to meet the specific requirements of the dataset and task at hand. Experimenting with different values can help find the optimal augmentation strategy that leads to the best model performance.

!!! info

    For more information about training augmentation operations, see the [reference section](../reference/data/augment.md).

## Logging

In training a YOLO11 model, you might find it valuable to keep track of the model's performance over time. This is where logging comes into play. Ultralytics YOLO provides support for three types of loggers - [Comet](../integrations/comet.md), [ClearML](../integrations/clearml.md), and [TensorBoard](../integrations/tensorboard.md).

To use a logger, select it from the dropdown menu in the code snippet above and run it. The chosen logger will be installed and initialized.

### Comet

[Comet](../integrations/comet.md) is a platform that allows data scientists and developers to track, compare, explain and optimize experiments and models. It provides functionalities such as real-time metrics, code diffs, and hyperparameters tracking.

To use Comet:

!!! example

    === "Python"

        ```python
        # pip install comet_ml
        import comet_ml

        comet_ml.init()
        ```

Remember to sign in to your Comet account on their website and get your API key. You will need to add this to your environment variables or your script to log your experiments.

### ClearML

[ClearML](https://clear.ml/) is an open-source platform that automates tracking of experiments and helps with efficient sharing of resources. It is designed to help teams manage, execute, and reproduce their ML work more efficiently.

To use ClearML:

!!! example

    === "Python"

        ```python
        # pip install clearml
        import clearml

        clearml.browser_login()
        ```

After running this script, you will need to sign in to your ClearML account on the browser and authenticate your session.

### TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard) is a visualization toolkit for [TensorFlow](https://www.ultralytics.com/glossary/tensorflow). It allows you to visualize your TensorFlow graph, plot quantitative metrics about the execution of your graph, and show additional data like images that pass through it.

To use TensorBoard in [Google Colab](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb):

!!! example

    === "CLI"

        ```bash
        load_ext tensorboard
        tensorboard --logdir ultralytics/runs # replace with 'runs' directory
        ```

To use TensorBoard locally run the below command and view results at `http://localhost:6006/`.

!!! example

    === "CLI"

        ```bash
        tensorboard --logdir ultralytics/runs # replace with 'runs' directory
        ```

This will load TensorBoard and direct it to the directory where your training logs are saved.

After setting up your logger, you can then proceed with your model training. All training metrics will be automatically logged in your chosen platform, and you can access these logs to monitor your model's performance over time, compare different models, and identify areas for improvement.

## FAQ

### How do I train an [object detection](https://www.ultralytics.com/glossary/object-detection) model using Ultralytics YOLO11?

To train an object detection model using Ultralytics YOLO11, you can either use the Python API or the CLI. Below is an example for both:

!!! example "Single-GPU and CPU Training Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

For more details, refer to the [Train Settings](#train-settings) section.

### What are the key features of Ultralytics YOLO11's Train mode?

The key features of Ultralytics YOLO11's Train mode include:

- **Automatic Dataset Download:** Automatically downloads standard datasets like COCO, VOC, and ImageNet.
- **Multi-GPU Support:** Scale training across multiple GPUs for faster processing.
- **Hyperparameter Configuration:** Customize hyperparameters through YAML files or CLI arguments.
- **Visualization and Monitoring:** Real-time tracking of training metrics for better insights.

These features make training efficient and customizable to your needs. For more details, see the [Key Features of Train Mode](#key-features-of-train-mode) section.

### How do I resume training from an interrupted session in Ultralytics YOLO11?

To resume training from an interrupted session, set the `resume` argument to `True` and specify the path to the last saved checkpoint.

!!! example "Resume Training Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the partially trained model
        model = YOLO("path/to/last.pt")

        # Resume training
        results = model.train(resume=True)
        ```

    === "CLI"

        ```bash
        yolo train resume model=path/to/last.pt
        ```

Check the section on [Resuming Interrupted Trainings](#resuming-interrupted-trainings) for more information.

### Can I train YOLO11 models on Apple silicon chips?

Yes, Ultralytics YOLO11 supports training on Apple silicon chips utilizing the Metal Performance Shaders (MPS) framework. Specify 'mps' as your training device.

!!! example "MPS Training Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo11n.pt")

        # Train the model on Apple silicon chip (M1/M2/M3/M4)
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device="mps")
        ```

    === "CLI"

        ```bash
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640 device=mps
        ```

For more details, refer to the [Apple Silicon MPS Training](#apple-silicon-mps-training) section.

### What are the common training settings, and how do I configure them?

Ultralytics YOLO11 allows you to configure a variety of training settings such as batch size, learning rate, epochs, and more through arguments. Here's a brief overview:

| Argument | Default | Description                                                            |
| -------- | ------- | ---------------------------------------------------------------------- |
| `model`  | `None`  | Path to the model file for training.                                   |
| `data`   | `None`  | Path to the dataset configuration file (e.g., `coco8.yaml`).           |
| `epochs` | `100`   | Total number of training epochs.                                       |
| `batch`  | `16`    | Batch size, adjustable as integer or auto mode.                        |
| `imgsz`  | `640`   | Target image size for training.                                        |
| `device` | `None`  | Computational device(s) for training like `cpu`, `0`, `0,1`, or `mps`. |
| `save`   | `True`  | Enables saving of training checkpoints and final model weights.        |

For an in-depth guide on training settings, check the [Train Settings](#train-settings) section.
