---
comments: true
description: Step-by-step guide to train YOLOv8 models with Ultralytics YOLO including examples of single-GPU and multi-GPU training
keywords: Ultralytics, YOLOv8, YOLO, object detection, train mode, custom dataset, GPU training, multi-GPU, hyperparameters, CLI examples, Python examples
---

# Model Training with Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

Training a deep learning model involves feeding it data and adjusting its parameters so that it can make accurate predictions. Train mode in Ultralytics YOLOv8 is engineered for effective and efficient training of object detection models, fully utilizing modern hardware capabilities. This guide aims to cover all the details you need to get started with training your own models using YOLOv8's robust set of features.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train a YOLOv8 model on Your Custom Dataset in Google Colab.
</p>

## Why Choose Ultralytics YOLO for Training?

Here are some compelling reasons to opt for YOLOv8's Train mode:

- **Efficiency:** Make the most out of your hardware, whether you're on a single-GPU setup or scaling across multiple GPUs.
- **Versatility:** Train on custom datasets in addition to readily available ones like COCO, VOC, and ImageNet.
- **User-Friendly:** Simple yet powerful CLI and Python interfaces for a straightforward training experience.
- **Hyperparameter Flexibility:** A broad range of customizable hyperparameters to fine-tune model performance.

### Key Features of Train Mode

The following are some notable features of YOLOv8's Train mode:

- **Automatic Dataset Download:** Standard datasets like COCO, VOC, and ImageNet are downloaded automatically on first use.
- **Multi-GPU Support:** Scale your training efforts seamlessly across multiple GPUs to expedite the process.
- **Hyperparameter Configuration:** The option to modify hyperparameters through YAML configuration files or CLI arguments.
- **Visualization and Monitoring:** Real-time tracking of training metrics and visualization of the learning process for better insights.

!!! Tip "Tip"

    * YOLOv8 datasets like COCO, VOC, ImageNet and many others automatically download on first use, i.e. `yolo train data=coco.yaml`

## Usage Examples

Train YOLOv8n on the COCO128 dataset for 100 epochs at image size 640. The training device can be specified using the `device` argument. If no argument is passed GPU `device=0` will be used if available, otherwise `device=cpu` will be used. See Arguments section below for a full list of training arguments.

!!! Example "Single-GPU and CPU Training Example"

    Device is determined automatically. If a GPU is available then it will be used, otherwise training will start on CPU.

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.yaml')  # build a new model from YAML
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

        # Train the model
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
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

### Multi-GPU Training

Multi-GPU training allows for more efficient utilization of available hardware resources by distributing the training load across multiple GPUs. This feature is available through both the Python API and the command-line interface. To enable multi-GPU training, specify the GPU device IDs you wish to use.

!!! Example "Multi-GPU Training Example"

    To train with 2 GPUs, CUDA devices 0 and 1 use the following commands. Expand to additional GPUs as required.

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

        # Train the model with 2 GPUs
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device=[0, 1])
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model using GPUs 0 and 1
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0,1
        ```

### Apple M1 and M2 MPS Training

With the support for Apple M1 and M2 chips integrated in the Ultralytics YOLO models, it's now possible to train your models on devices utilizing the powerful Metal Performance Shaders (MPS) framework. The MPS offers a high-performance way of executing computation and image processing tasks on Apple's custom silicon.

To enable training on Apple M1 and M2 chips, you should specify 'mps' as your device when initiating the training process. Below is an example of how you could do this in Python and via the command line:

!!! Example "MPS Training Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

        # Train the model with 2 GPUs
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device='mps')
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model using GPUs 0 and 1
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=mps
        ```

While leveraging the computational power of the M1/M2 chips, this enables more efficient processing of the training tasks. For more detailed guidance and advanced configuration options, please refer to the [PyTorch MPS documentation](https://pytorch.org/docs/stable/notes/mps.html).

### Resuming Interrupted Trainings

Resuming training from a previously saved state is a crucial feature when working with deep learning models. This can come in handy in various scenarios, like when the training process has been unexpectedly interrupted, or when you wish to continue training a model with new data or for more epochs.

When training is resumed, Ultralytics YOLO loads the weights from the last saved model and also restores the optimizer state, learning rate scheduler, and the epoch number. This allows you to continue the training process seamlessly from where it was left off.

You can easily resume training in Ultralytics YOLO by setting the `resume` argument to `True` when calling the `train` method, and specifying the path to the `.pt` file containing the partially trained model weights.

Below is an example of how to resume an interrupted training using Python and via the command line:

!!! Example "Resume Training Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('path/to/last.pt')  # load a partially trained model

        # Resume training
        results = model.train(resume=True)
        ```

    === "CLI"

        ```bash
        # Resume an interrupted training
        yolo train resume model=path/to/last.pt
        ```

By setting `resume=True`, the `train` function will continue training from where it left off, using the state stored in the 'path/to/last.pt' file. If the `resume` argument is omitted or set to `False`, the `train` function will start a new training session.

Remember that checkpoints are saved at the end of every epoch by default, or at fixed interval using the `save_period` argument, so you must complete at least 1 epoch to resume a training run.

## Train Settings

The training settings for YOLO models encompass various hyperparameters and configurations used during the training process. These settings influence the model's performance, speed, and accuracy. Key training settings include batch size, learning rate, momentum, and weight decay. Additionally, the choice of optimizer, loss function, and training dataset composition can impact the training process. Careful tuning and experimentation with these settings are crucial for optimizing performance.

| Argument          | Default  | Description                                                                                                                                                                                                          |
|-------------------|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model`           | `None`   | Specifies the model file for training. Accepts a path to either a `.pt` pretrained model or a `.yaml` configuration file. Essential for defining the model structure or initializing weights.                        |
| `data`            | `None`   | Path to the dataset configuration file (e.g., `coco128.yaml`). This file contains dataset-specific parameters, including paths to training and validation data, class names, and number of classes.                  |
| `epochs`          | `100`    | Total number of training epochs. Each epoch represents a full pass over the entire dataset. Adjusting this value can affect training duration and model performance.                                                 |
| `time`            | `None`   | Maximum training time in hours. If set, this overrides the `epochs` argument, allowing training to automatically stop after the specified duration. Useful for time-constrained training scenarios.                  |
| `patience`        | `100`    | Number of epochs to wait without improvement in validation metrics before early stopping the training. Helps prevent overfitting by stopping training when performance plateaus.                                     |
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

## Augmentation Settings and Hyperparameters

Augmentation techniques are essential for improving the robustness and performance of YOLO models by introducing variability into the training data, helping the model generalize better to unseen data. The following table outlines the purpose and effect of each augmentation argument:

| Argument       | Type    | Default       | Range         | Description                                                                                                                                                               |
|----------------|---------|---------------|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `hsv_h`        | `float` | `0.015`       | `0.0 - 1.0`   | Adjusts the hue of the image by a fraction of the color wheel, introducing color variability. Helps the model generalize across different lighting conditions.            |
| `hsv_s`        | `float` | `0.7`         | `0.0 - 1.0`   | Alters the saturation of the image by a fraction, affecting the intensity of colors. Useful for simulating different environmental conditions.                            |
| `hsv_v`        | `float` | `0.4`         | `0.0 - 1.0`   | Modifies the value (brightness) of the image by a fraction, helping the model to perform well under various lighting conditions.                                          |
| `degrees`      | `float` | `0.0`         | `-180 - +180` | Rotates the image randomly within the specified degree range, improving the model's ability to recognize objects at various orientations.                                 |
| `translate`    | `float` | `0.1`         | `0.0 - 1.0`   | Translates the image horizontally and vertically by a fraction of the image size, aiding in learning to detect partially visible objects.                                 |
| `scale`        | `float` | `0.5`         | `>=0.0`       | Scales the image by a gain factor, simulating objects at different distances from the camera.                                                                             |
| `shear`        | `float` | `0.0`         | `-180 - +180` | Shears the image by a specified degree, mimicking the effect of objects being viewed from different angles.                                                               |
| `perspective`  | `float` | `0.0`         | `0.0 - 0.001` | Applies a random perspective transformation to the image, enhancing the model's ability to understand objects in 3D space.                                                |
| `flipud`       | `float` | `0.0`         | `0.0 - 1.0`   | Flips the image upside down with the specified probability, increasing the data variability without affecting the object's characteristics.                               |
| `fliplr`       | `float` | `0.5`         | `0.0 - 1.0`   | Flips the image left to right with the specified probability, useful for learning symmetrical objects and increasing dataset diversity.                                   |
| `bgr`          | `float` | `0.0`         | `0.0 - 1.0`   | Flips the image channels from RGB to BGR with the specified probability, useful for increasing robustness to incorrect channel ordering.                                  |
| `mosaic`       | `float` | `1.0`         | `0.0 - 1.0`   | Combines four training images into one, simulating different scene compositions and object interactions. Highly effective for complex scene understanding.                |
| `mixup`        | `float` | `0.0`         | `0.0 - 1.0`   | Blends two images and their labels, creating a composite image. Enhances the model's ability to generalize by introducing label noise and visual variability.             |
| `copy_paste`   | `float` | `0.0`         | `0.0 - 1.0`   | Copies objects from one image and pastes them onto another, useful for increasing object instances and learning object occlusion.                                         |
| `auto_augment` | `str`   | `randaugment` | -             | Automatically applies a predefined augmentation policy (`randaugment`, `autoaugment`, `augmix`), optimizing for classification tasks by diversifying the visual features. |
| `erasing`      | `float` | `0.4`         | `0.0 - 1.0`   | Randomly erases a portion of the image during classification training, encouraging the model to focus on less obvious features for recognition.                           |

These settings can be adjusted to meet the specific requirements of the dataset and task at hand. Experimenting with different values can help find the optimal augmentation strategy that leads to the best model performance.

!!! info

    For more information about training augmentation operations, see the [reference section](../reference/data/augment.md).

## Logging

In training a YOLOv8 model, you might find it valuable to keep track of the model's performance over time. This is where logging comes into play. Ultralytics' YOLO provides support for three types of loggers - Comet, ClearML, and TensorBoard.

To use a logger, select it from the dropdown menu in the code snippet above and run it. The chosen logger will be installed and initialized.

### Comet

[Comet](../integrations/comet.md) is a platform that allows data scientists and developers to track, compare, explain and optimize experiments and models. It provides functionalities such as real-time metrics, code diffs, and hyperparameters tracking.

To use Comet:

!!! Example

    === "Python"

        ```python
        # pip install comet_ml
        import comet_ml

        comet_ml.init()
        ```

Remember to sign in to your Comet account on their website and get your API key. You will need to add this to your environment variables or your script to log your experiments.

### ClearML

[ClearML](https://www.clear.ml/) is an open-source platform that automates tracking of experiments and helps with efficient sharing of resources. It is designed to help teams manage, execute, and reproduce their ML work more efficiently.

To use ClearML:

!!! Example

    === "Python"

        ```python
        # pip install clearml
        import clearml

        clearml.browser_login()
        ```

After running this script, you will need to sign in to your ClearML account on the browser and authenticate your session.

### TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard) is a visualization toolkit for TensorFlow. It allows you to visualize your TensorFlow graph, plot quantitative metrics about the execution of your graph, and show additional data like images that pass through it.

To use TensorBoard in [Google Colab](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb):

!!! Example

    === "CLI"

        ```bash
        load_ext tensorboard
        tensorboard --logdir ultralytics/runs  # replace with 'runs' directory
        ```

To use TensorBoard locally run the below command and view results at http://localhost:6006/.

!!! Example

    === "CLI"

        ```bash
        tensorboard --logdir ultralytics/runs  # replace with 'runs' directory
        ```

This will load TensorBoard and direct it to the directory where your training logs are saved.

After setting up your logger, you can then proceed with your model training. All training metrics will be automatically logged in your chosen platform, and you can access these logs to monitor your model's performance over time, compare different models, and identify areas for improvement.
