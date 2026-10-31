---
comments: true
description: Train Ultralytics YOLO26 models with Python or the CLI, covering training arguments, augmentation, multi-GPU, Apple MPS, Ascend NPU, resume, and logging.
keywords: Ultralytics, YOLO26, model training, deep learning, object detection, GPU training, AMD ROCm, Huawei Ascend NPU, data augmentation, hyperparameter tuning, model performance, Apple silicon training
---

# Model Training with Ultralytics YOLO

<img width="1024" src="https://cdn.ul.run/i/f874ab850f33f361d01a01e9a8c98655.avif" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

Training a [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) model involves feeding it data and adjusting its parameters so that it can make accurate predictions. Train mode in Ultralytics YOLO26 trains [object detection](https://www.ultralytics.com/glossary/object-detection) and other vision task models efficiently on modern hardware. This guide covers the details you need to start training your own models with YOLO26. If you haven't installed Ultralytics yet, start with the [Quickstart guide](../quickstart.md), or follow the beginner Ultralytics Academy lesson [Train Your First Model](https://academy.ultralytics.com/courses/train-your-first-yolo/train-your-first-model).

See the [unreleased YOLO27 preview](../models/yolo27.md#usage-examples) for planned training examples.

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

Reasons to use YOLO26's Train mode:

- **Efficiency:** Make the most of your hardware, whether you're on a single-GPU setup or scaling across multiple GPUs.
- **Versatility:** Train on custom datasets in addition to readily available ones like [COCO](https://platform.ultralytics.com/ultralytics/datasets/coco2017), [VOC](https://platform.ultralytics.com/ultralytics/datasets/voc), and ImageNet.
- **User-Friendly:** Simple yet powerful CLI and Python interfaces for a straightforward training experience.
- **Hyperparameter Flexibility:** A broad range of customizable [hyperparameters](../guides/hyperparameter-tuning.md) to fine-tune model performance. For deeper control, you can [customize the trainer](../guides/custom-trainer.md) itself.
- **Cloud Training:** Train on cloud GPUs through [Ultralytics Platform](../platform/train/cloud-training.md) with real-time metrics and automatic checkpointing.

### Key Features of Train Mode

The following are some notable features of YOLO26's Train mode:

- **Automatic Dataset Download:** Dataset configurations with a download source are downloaded automatically on first use, e.g., `yolo train data=coco8.yaml`. See the [Datasets overview](../datasets/index.md) for supported formats and datasets.
- **Multi-GPU Support:** Scale training across multiple GPUs to shorten training time.
- **Hyperparameter Configuration:** The option to modify hyperparameters through YAML configuration files or CLI arguments.
- **Visualization and Monitoring:** Real-time tracking of training metrics and visualization of the learning process for better insights.

## Usage Examples

Train [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n) on the [COCO8](https://platform.ultralytics.com/ultralytics/datasets/coco8) dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 640. The training device can be specified using the `device` argument. If no argument is passed, GPU `device=0` will be used when available; otherwise `device='cpu'` will be used. See [Train Settings](#train-settings) for a full list of training arguments.

!!! warning "Windows Multi-Processing Error"

    On Windows, you may receive a `RuntimeError` when launching the training as a script. Add an `if __name__ == "__main__":` block before your training code to resolve it.

!!! example "Single-GPU and CPU Training Example"

    Device is determined automatically. If a GPU is available, it will be used (default CUDA device 0); otherwise training will start on CPU.

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.yaml")  # build a new model from YAML
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo detect train data=coco8.yaml model=yolo26n.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo detect train data=coco8.yaml model=yolo26n.yaml pretrained=yolo26n.pt epochs=100 imgsz=640
        ```

### Multi-GPU Training

Multi-GPU training makes better use of available hardware by [distributing the training load](https://www.ultralytics.com/glossary/distributed-training) across multiple GPUs. This feature is available through both the Python API and the command-line interface. To enable multi-GPU training, specify the GPU device IDs you wish to use.

!!! example "Multi-GPU Training Example"

    To train with 2 GPUs (CUDA devices 0 and 1), use the following commands. Expand to additional GPUs as required.

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model with 2 GPUs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[0, 1])

        # Train the model with the two most idle GPUs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[-1, -1])
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model using GPUs 0 and 1
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640 device=0,1

        # Use the two most idle GPUs
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640 device=-1,-1
        ```

!!! warning "Windows Multi-GPU Support"

    Multi-GPU training does not work on Windows with the official PyTorch wheels for `torch>=2.4`. Ultralytics launches DDP through `torch.distributed.run`, whose rendezvous step builds a `TCPStore` that has defaulted to the libuv backend since PyTorch 2.4, and the official Windows wheels are built without libuv. Training exits immediately with:

    ```
    RuntimeError: use_libuv was requested but PyTorch was built without libuv support
    ```

    Setting `USE_LIBUV=0` does not resolve this, because the rendezvous constructs the `TCPStore` directly and that code path never reads the variable. Train on Linux or [WSL2](https://learn.microsoft.com/windows/wsl/install) to use multiple GPUs, or train on a single GPU with `device=0` on Windows.

!!! note "Multi-GPU Training with Custom Code"

    When you specify multiple devices (e.g., `device=[0, 1]`), Ultralytics internally spawns a new trainer instance and executes `torch.distributed.run` under the hood. This works for standard CLI usage and unmodified Python scripts.

    However, if your script contains custom components—such as a custom trainer, validator, dataset, or augmentation pipeline—these objects cannot be automatically serialized and transferred to the DDP subprocesses. In this case, you must launch your script directly with `torch.distributed.run`:

    ```bash
    python -m torch.distributed.run --nproc_per_node 2 your_training_script.py
    ```

AMD GPU training uses a PyTorch ROCm build with the standard `device=0` or `device=cuda:0` syntax. See the [AMD integration guide](../integrations/amd.md) for ROCm installation, multi-GPU training, AMP notes, and [MIGraphX inference](../integrations/amd.md#export-and-inference-with-migraphx) of exported models.

Intel GPU training uses `device=xpu:0`, or multiple XPU IDs with a PyTorch build that provides XCCL.

### Huawei Ascend NPU Training

Ultralytics supports training and validation on Huawei Ascend NPUs through [`torch_npu`](https://github.com/Ascend/pytorch). Install mutually compatible CANN, PyTorch, and `torch_npu` versions by following the [Ascend Extension for PyTorch installation guide](https://github.com/Ascend/pytorch#installation), then source the CANN environment before starting Ultralytics:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

!!! example "Ascend NPU Training Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Train on one Ascend NPU
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device="npu:0")

        # Train across two Ascend NPUs with HCCL
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device="npu:0,1")
        ```

    === "CLI"

        ```bash
        # Train on one Ascend NPU
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640 device=npu:0

        # Train across two Ascend NPUs with HCCL
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640 device=npu:0,1
        ```

The standard training features, including AMP, validation, checkpointing, and resume, use the active NPU. AutoBatch is available for single-NPU training, while multiple NPU IDs launch distributed training through HCCL. See the [Huawei Ascend integration guide](../integrations/ascend.md) for model export and deployment after training.

### Idle GPU Training

Idle GPU Training automatically selects the least utilized GPUs in multi-GPU systems, so you don't have to pick devices manually. It ranks available GPUs by utilization and free VRAM.

!!! example "Idle GPU Training Example"

    To automatically select and use the most idle GPU(s) for training, use the `-1` device parameter. This is particularly useful in shared computing environments or servers with multiple users.

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train using the single most idle GPU
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=-1)

        # Train using the two most idle GPUs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[-1, -1])
        ```

    === "CLI"

        ```bash
        # Start training using the single most idle GPU
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640 device=-1

        # Start training using the two most idle GPUs
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640 device=-1,-1
        ```

The auto-selection algorithm prioritizes GPUs with:

1. Lower current utilization percentages
2. Higher available memory (free VRAM)

This is especially useful in shared computing environments or when running multiple training jobs at once, because devices are chosen from current conditions each time a run starts.

### Apple Silicon MPS Training

Ultralytics YOLO models can train on Apple silicon through the Metal Performance Shaders (MPS) framework, which runs computation and image processing tasks on Apple's custom GPUs.

To train on Apple silicon, specify `mps` as your device when starting training, in Python or on the command line:

!!! example "MPS Training Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model with MPS
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device="mps")
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model using MPS
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640 device=mps
        ```

For more detailed guidance and advanced configuration options, see the [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).

### Resuming Interrupted Trainings

Resuming training from a saved checkpoint is useful when a run was interrupted unexpectedly, or when you want to continue training a model with new data or for more epochs.

When training is resumed, Ultralytics YOLO loads the weights from the last saved model and also restores the optimizer state, [learning rate](https://www.ultralytics.com/glossary/learning-rate) scheduler, epoch number, and dataset. Training then continues from where it stopped. Pass an explicit `data=` alongside `resume` to continue on a different dataset instead of the checkpoint's.

Resume training in Ultralytics YOLO by setting the `resume` argument to `True` when calling the `train` method, and specifying the path to the `.pt` file containing the partially trained model weights.

To resume an interrupted training run in Python or on the command line:

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

By setting `resume=True`, the `train` function will continue training from where it left off, using the state stored in the `path/to/last.pt` file. If the `resume` argument is omitted or set to `False`, the `train` function will start a new training session. A run that streams to [Ultralytics Platform](../platform/train/cloud-training.md#resume-remote-training) continues the same Platform model when resumed.

Remember that checkpoints are saved at the end of every epoch by default, or at fixed intervals using the `save_period` argument, so you must complete at least 1 epoch to resume a training run.

## Train Settings

The training settings for YOLO models encompass various hyperparameters and configurations used during the training process. These settings influence the model's performance, speed, and [accuracy](https://www.ultralytics.com/glossary/accuracy). Key training settings include batch size, learning rate, momentum, and weight decay. Additionally, the choice of optimizer, [loss function](https://www.ultralytics.com/glossary/loss-function), and training dataset composition can impact the training process. Careful tuning and experimentation with these settings are key to optimizing performance.

### MuSGD Optimizer

In YOLO26, **MuSGD** is a hybrid optimizer that combines standard **SGD** updates with **Muon-style orthogonalized updates**.

It is **recommended for longer YOLO26 training runs and larger datasets**, where orthogonalized Muon updates can help stabilize optimization.

Only 2D linear weights and 4D convolutional filters (reshaped to 2D) receive the Muon-style update together with SGD, while all other parameters, such as batch normalization weights and bias terms, remain on standard SGD.

When `optimizer=auto` is used, Ultralytics automatically selects **MuSGD** for longer training runs (typically when iterations > 10000). For shorter runs, the trainer falls back to **AdamW**. In `auto` mode, the trainer also selects the learning rate, so a user-provided `lr0` is ignored. To control these settings manually, select an optimizer explicitly, for example `optimizer=AdamW lr0=0.001`.

Example usage:

```bash
yolo train model=yolo26n.pt data=coco8.yaml optimizer=MuSGD
```

See the implementation in `ultralytics/optim/muon.py` and the optimizer auto-selection logic in `BaseTrainer.build_optimizer`. For background on how MuSGD works alongside ProgLoss and STAL, read [YOLO26: Smarter Training with ProgLoss, STAL & MuSGD](https://www.ultralytics.com/blog/how-ultralytics-yolo26-trains-smarter-with-progloss-stal-and-musgd).

{% include "macros/train-args.md" %}

!!! info "Note on Batch-size Settings"

    The `batch` argument accepts three kinds of values, and the trainer adds two safeguards:

    - **Fixed [Batch Size](https://www.ultralytics.com/glossary/batch-size)**: Set an integer value (e.g., `batch=16`), specifying the number of images per batch directly.
    - **Auto Mode (60% GPU Memory)**: Use `batch=-1` to automatically adjust batch size for approximately 60% of the CUDA memory available to training.
    - **Auto Mode with Utilization Fraction**: Set a fraction value (e.g., `batch=0.70`) to adjust batch size based on the specified fraction of available GPU memory.
    - **OOM Auto-Retry**: If a CUDA out-of-memory error occurs during the first epoch, the trainer automatically halves the batch size and retries (up to 3 times). This only applies to single-GPU training; multi-GPU (DDP) training will raise the error immediately.
    - **No Fit Found**: If no candidate batch size produces a usable profile, AutoBatch raises a clear `RuntimeError` instead of silently falling back to an unrelated default.

## Augmentation Settings and Hyperparameters

[Data augmentation](https://www.ultralytics.com/glossary/data-augmentation) techniques are essential for improving the robustness and performance of YOLO models by introducing variability into the [training data](https://www.ultralytics.com/glossary/training-data), helping the model generalize better to unseen data. The following table outlines the purpose and effect of each augmentation argument:

{% include "macros/augmentation-args.md" %}

These settings can be adjusted to meet the specific requirements of the dataset and task at hand. Experimenting with different values can help find the best augmentation strategy; the [YOLO data augmentation guide](../guides/yolo-data-augmentation.md) shows the effect of each setting.

!!! info

    For more information about training augmentation operations, see the [reference section](../reference/data/augment.md).

## Logging

Training metrics, plots, and checkpoints are always written to the run directory, and Ultralytics also streams them to any [experiment tracker](https://www.ultralytics.com/glossary/experiment-tracking) you have installed and enabled: [Comet](../integrations/comet.md), [ClearML](../integrations/clearml.md), [TensorBoard](../integrations/tensorboard.md), [MLflow](../integrations/mlflow.md), [Weights & Biases](../integrations/weights-biases.md), and [DVCLive](../integrations/dvc.md). Each logger is toggled through the [Ultralytics settings](../usage/settings.md), for example `yolo settings tensorboard=True`. The three most common setups are shown below.

### Comet

[Comet](../integrations/comet.md) is a platform that allows data scientists and developers to track, compare, explain and optimize experiments and models. It provides real-time metrics, code diffs, and hyperparameter tracking.

To use Comet:

!!! example

    === "Python"

        ```python
        # pip install comet_ml
        import comet_ml

        comet_ml.login()
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
        %load_ext tensorboard
        tensorboard --logdir ultralytics/runs # replace with 'runs' directory
        ```

To use TensorBoard locally, run the command below and view results at `localhost:6006`.

!!! example

    === "CLI"

        ```bash
        tensorboard --logdir ultralytics/runs # replace with 'runs' directory
        ```

This will load TensorBoard and direct it to the directory where your training logs are saved.

After setting up your logger, start training as usual. Training metrics are logged automatically to your chosen platform, where you can monitor performance over time, compare models, and identify areas for improvement.

## What's Next

[Validate](val.md) your trained model against held-out data to check its real-world accuracy, then [export](export.md) it to ONNX, TensorRT, or another deployment format. Training on your own data instead of COCO8? Format it first with the [Datasets guide](../datasets/index.md).

## FAQ

### Can I train without a local GPU?

Yes. [Ultralytics Platform cloud training](../platform/train/cloud-training.md) includes free credits to get started. Upload your dataset, select a model and GPU, and train directly from the browser.

### How do I train an object detection model using Ultralytics YOLO26?

To train an object detection model using Ultralytics YOLO26, you can either use the Python API or the CLI. Below is an example for both:

!!! example "Single-GPU and CPU Training Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

For more details, refer to the [Train Settings](#train-settings) section.

### What are the key features of Ultralytics YOLO26's Train mode?

The key features of Ultralytics YOLO26's Train mode include:

- **Automatic Dataset Download:** Automatically downloads standard datasets like COCO, VOC, and ImageNet.
- **Multi-GPU Support:** Scale training across multiple GPUs for faster processing.
- **Hyperparameter Configuration:** Customize hyperparameters through YAML files or CLI arguments.
- **Visualization and Monitoring:** Real-time tracking of training metrics for better insights.

These features make training efficient and customizable to your needs. For more details, see the [Key Features of Train Mode](#key-features-of-train-mode) section.

### How do I resume training from an interrupted session in Ultralytics YOLO26?

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

### How do I train a model on an imbalanced dataset?

Class imbalance occurs when some classes have significantly fewer examples than others in your training data. This can cause the model to perform poorly on rare classes. Ultralytics YOLO supports class weighting through the `cls_pw` argument to address this issue.

The `cls_pw` argument controls class weighting power based on inverse class frequency:

- `cls_pw=0.0` (default): Disables class weighting
- `cls_pw=1.0`: Applies full inverse frequency weighting
- Values between `0.0` and `1.0`: Provide partial weighting for moderate imbalance

The class weights are computed as `(1.0 / class_counts) ^ cls_pw` and normalized so their mean equals 1.0. Classes without training labels, such as those excluded by `classes=`, get a weight of 1.0.

!!! example "Training on Imbalanced Dataset"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n.pt")

        # Train with full class weighting for severely imbalanced data
        results = model.train(data="custom.yaml", epochs=100, imgsz=640, cls_pw=1.0)

        # Or use partial weighting (0.25) for moderate imbalance
        results = model.train(data="custom.yaml", epochs=100, imgsz=640, cls_pw=0.25)
        ```

    === "CLI"

        ```bash
        # Train with full inverse frequency weighting
        yolo detect train data=custom.yaml model=yolo26n.pt epochs=100 imgsz=640 cls_pw=1.0

        # Train with partial weighting for moderate imbalance
        yolo detect train data=custom.yaml model=yolo26n.pt epochs=100 imgsz=640 cls_pw=0.25
        ```

!!! tip

    Start with `cls_pw=0.25` for moderately imbalanced datasets and increase to `1.0` if the rare classes still underperform. You can check the computed class weights in the training logs to verify the weight distribution.

### Can I train YOLO26 models on Apple silicon chips?

Yes, Ultralytics YOLO26 supports training on Apple silicon chips utilizing the Metal Performance Shaders (MPS) framework. Specify `mps` as your training device.

!!! example "MPS Training Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n.pt")

        # Train the model on Apple silicon chip (M1/M2/M3/M4)
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device="mps")
        ```

    === "CLI"

        ```bash
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640 device=mps
        ```

For more details, refer to the [Apple Silicon MPS Training](#apple-silicon-mps-training) section.

### What are the common training settings, and how do I configure them?

Ultralytics YOLO26 allows you to configure a variety of training settings such as batch size, learning rate, epochs, and more through arguments. Here's a brief overview:

| Argument | Default | Description                                                                                                            |
| -------- | ------- | ---------------------------------------------------------------------------------------------------------------------- |
| `model`  | `None`  | Path to the model file for training.                                                                                   |
| `data`   | `None`  | Path to the dataset YAML (e.g., `coco8.yaml`), or a dataset directory or name (e.g., `imagenet10`) for classification. |
| `epochs` | `100`   | Total number of training epochs.                                                                                       |
| `batch`  | `16`    | Batch size, adjustable as integer or auto mode.                                                                        |
| `imgsz`  | `640`   | Target image size for training.                                                                                        |
| `device` | `None`  | Computational device(s) for training like `cpu`, `0`, `0,1`, or `mps`.                                                 |
| `save`   | `True`  | Enables saving of training checkpoints and final model weights.                                                        |

For an in-depth guide on training settings, check the [Train Settings](#train-settings) section.
