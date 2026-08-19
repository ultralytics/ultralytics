---
comments: true
description: YOLOE-26 is a real-time open-vocabulary detection and instance segmentation model. Detect and segment any class from a text prompt, a visual example, or a built-in vocabulary, without retraining.
keywords: YOLOE, YOLOE-26, open-vocabulary detection, open-vocabulary segmentation, zero-shot detection, instance segmentation, text prompts, visual prompts, prompt-free detection, YOLO26, real-time object detection
---

# YOLOE: Real-Time Open-Vocabulary Detection and Segmentation

## Introduction

![YOLOE Prompting Options](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/yoloe-visualization.avif)

[YOLOE (Real-Time Seeing Anything)](https://arxiv.org/html/2503.07465v1) is a new advancement in zero-shot, promptable YOLO models, designed for **open-vocabulary** detection and segmentation. Unlike previous YOLO models limited to fixed categories, YOLOE uses text, image, or internal vocabulary prompts, enabling real-time detection of any object class. Built on the Ultralytics YOLO architectures — YOLOv8, [YOLO11](yolo11.md) and [YOLO26](yolo26.md) — and inspired by [YOLO-World](yolo-world.md), YOLOE achieves **state-of-the-art zero-shot performance** with minimal impact on speed and accuracy.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/JcZsqUc8PMM"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to use Ultralytics YOLOE-26 (New) | Open Vocabulary & Real-Time Seeing Anything 🚀
</p>

## Quick Start

Name the classes you want and run. YOLOE-26 returns boxes and [instance segmentation](../tasks/segment.md) masks for categories it was never trained on.

!!! example "Detect anything you can name"

    === "Python"

        ```python
        from ultralytics import YOLOE

        model = YOLOE("yoloe-26s-seg.pt")

        # Any class names work here, not just COCO ones
        model.set_classes(["forklift", "safety helmet", "spilled liquid"])

        results = model.predict("https://ultralytics.com/images/bus.jpg")
        results[0].show()
        ```

    === "CLI"

        ```bash
        yolo predict model=yoloe-26s-seg.pt source="https://ultralytics.com/images/bus.jpg" classes="forklift,safety helmet,spilled liquid"
        ```

The first `set_classes()` call downloads a text encoder — see [Installation and requirements](#installation-and-requirements) before running this offline.

## Choosing a Prompting Mode

YOLOE accepts three kinds of prompt, and the choice decides which checkpoint you load and what your class labels look like. Pick the row that matches what you can supply at inference time.

| Mode              | Checkpoint    | You supply                         | Class names in the results                   | Use it when                                                                  |
| ----------------- | ------------- | ---------------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------- |
| **Text prompt**   | `*-seg.pt`    | Class names as strings             | Exactly the names you passed                 | You can describe the target in words — the usual choice                      |
| **Visual prompt** | `*-seg.pt`    | Example boxes on a reference image | Generic `object0`, `object1`, …              | The target is hard to name: a specific part, logo, or defect                 |
| **Prompt-free**   | `*-seg-pf.pt` | Nothing                            | Names from the built-in 4,585-tag vocabulary | You are cataloguing or exploring and do not know what to look for in advance |

!!! warning "Two behaviours that surprise people"

    - **Visual prompts do not carry your labels.** The class IDs in `visual_prompts` group examples together; the model reports them as `object0`, `object1`, and so on. Map them back to your own names yourself.
    - **Prompt-free checkpoints reject `set_classes()`.** Calling it on a `*-seg-pf.pt` model raises `AssertionError: Prompt-free model does not support setting classes.` Load a `*-seg.pt` checkpoint instead when you need your own classes.

## Installation and Requirements

```bash
pip install -U ultralytics
```

Text prompting needs a text encoder on top of that, and it is fetched on first use rather than at install time:

- The first `set_classes()` call installs [ultralytics/CLIP](https://github.com/ultralytics/CLIP) from GitHub with `pip` (it provides the tokenizer) and downloads a TorchScript text encoder into the Ultralytics weights directory. YOLOE-26 pulls `mobileclip2_b.ts`, about 254 MB; YOLOE-11 and YOLOE-v8 pull `mobileclip_blt.ts`. Both steps need network access, so run one prompted prediction before deploying to an offline or air-gapped machine.
- Visual prompts and prompt-free checkpoints need no text encoder at all.
- Text prompting requires **PyTorch 1.13 or newer**.

To skip the download at inference time entirely, bake the prompts into the weights once and reuse them — see [Reuse prompt embeddings](#export-usage).

The original paper reports, for the v8-scale models it introduced: on LVIS, YOLOE-v8-S beats YOLO-Worldv2-S by **3.5 AP** at a third of the training cost and 1.4× the inference speed; transferred to COCO, YOLOE-v8-L gains **0.6 box AP** and **0.4 mask AP** over closed-set YOLOv8-L with nearly **4× less training time**. The sections below cover the architecture, the checkpoints Ultralytics ships, and how to use them.

## Architecture Overview

<p align="center">
  <img src="https://github.com/THU-MIG/yoloe/raw/main/figures/pipeline.svg" alt="YOLOE Architecture" width=90%>
</p>

YOLOE retains the standard YOLO structure—a convolutional **backbone** (e.g., CSP-Darknet) for feature extraction, a **neck** (e.g., PAN-FPN) for multi-scale fusion, and an **anchor-free, decoupled** detection **head** (as in YOLOv8/YOLO11) predicting classes and boxes independently. YOLOE introduces three novel modules enabling open-vocabulary detection:

- **Re-parameterizable Region-Text Alignment (RepRTA)**: Supports **text-prompted detection** by refining text [embeddings](https://www.ultralytics.com/glossary/embeddings) (e.g., from CLIP) via a small auxiliary network. The auxiliary network is re-parameterized away after training, so prompting costs no extra network at inference — the prompt embeddings are simply compared against region features. YOLOE thus detects arbitrary text-labeled objects, such as an unseen "traffic light".

- **Semantic-Activated Visual Prompt Encoder (SAVPE)**: Enables **visual-prompted detection** via a lightweight embedding branch. Given a reference image, SAVPE encodes semantic and activation features, conditioning the model to detect visually similar objects—a one-shot detection capability useful for logos or specific parts.

- **Lazy Region-Prompt Contrast (LRPC)**: In **prompt-free mode**, YOLOE performs open-set recognition using internal embeddings matched against a built-in vocabulary of 4,585 tag names. Without external prompts or encoders, YOLOE identifies objects via embedding similarity lookup, efficiently handling large label spaces at inference.

Additionally, YOLOE integrates real-time **instance segmentation** by extending the detection head with a mask prediction branch (similar to YOLACT or YOLOv8-Seg), adding minimal overhead.

Crucially, YOLOE's open-world modules introduce **no inference cost** when used as a regular closed-set YOLO. Post-training, YOLOE parameters can be re-parameterized into a standard YOLO head, preserving identical FLOPs and speed (e.g., matching [YOLO11](yolo11.md) exactly).

## Available Models, Supported Tasks, and Operating Modes

This section details the models available with their specific pretrained weights, the tasks they support, and their compatibility with various operating modes such as [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), and [Export](../modes/export.md), denoted by ✅ for supported modes and ❌ for unsupported modes.

### Text/Visual Prompt models

| Model Type | Pretrained Weights                                                                                  | Tasks Supported                              | Training | Validation | Inference | Export |
| ---------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------- | -------- | ---------- | --------- | ------ |
| YOLOE-11S  | [yoloe-11s-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11s-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-11M  | [yoloe-11m-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11m-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-11L  | [yoloe-11l-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11l-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-v8S  | [yoloe-v8s-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-v8s-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-v8M  | [yoloe-v8m-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-v8m-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-v8L  | [yoloe-v8l-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-v8l-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-26N  | [yoloe-26n-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26n-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-26S  | [yoloe-26s-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-26M  | [yoloe-26m-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26m-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-26L  | [yoloe-26l-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-26X  | [yoloe-26x-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |

### Prompt Free models

| Model Type   | Pretrained Weights                                                                                        | Tasks Supported                              | Training | Validation | Inference | Export |
| ------------ | --------------------------------------------------------------------------------------------------------- | -------------------------------------------- | -------- | ---------- | --------- | ------ |
| YOLOE-11S-PF | [yoloe-11s-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11s-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-11M-PF | [yoloe-11m-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11m-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-11L-PF | [yoloe-11l-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11l-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-v8S-PF | [yoloe-v8s-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-v8s-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-v8M-PF | [yoloe-v8m-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-v8m-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-v8L-PF | [yoloe-v8l-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-v8l-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-26N-PF | [yoloe-26n-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26n-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-26S-PF | [yoloe-26s-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-26M-PF | [yoloe-26m-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26m-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-26L-PF | [yoloe-26l-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOE-26X-PF | [yoloe-26x-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅       | ✅         | ✅        | ✅     |

!!! tip "YOLOE-26 Performance"

    === "Text/Visual Prompts"

        | Model         | size<br><sup>(pixels)</sup> | Prompt Type | mAP<sup>minival<br>50-95(e2e)</sup> | mAP<sup>minival<br>50-95</sup> | mAP<sub>r</sub> | mAP<sub>c</sub> | mAP<sub>f</sub> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        |---------------|-----------------------------|-------------|-------------------------------------|----------------------------|-----------------|-----------------|-----------------|--------------------------|-------------------------|
        | [YOLOE-26n-seg](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26n-seg.pt) | 640                         | Text/Visual | 23.7 / 20.9                         | 24.7 / 21.9                | 20.5 / 17.6     | 24.1 / 22.3     | 26.1 / 22.4     | 3.9 / 3.1                | 6.1                     |
        | [YOLOE-26s-seg](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt) | 640                         | Text/Visual | 29.9 / 27.1                         | 30.8 / 28.6                | 23.9 / 25.1     | 29.6 / 27.8     | 33.0 / 29.9     | 10.7 / 11.0              | 21.9                    |
        | [YOLOE-26m-seg](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26m-seg.pt) | 640                         | Text/Visual | 35.4 / 31.3                         | 35.4 / 33.9                | 31.1 / 33.4     | 34.7 / 34.0     | 36.9 / 33.8     | 21.3 / 25.1              | 70.6                    |
        | [YOLOE-26l-seg](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg.pt) | 640                         | Text/Visual | 36.8 / 33.7                         | 37.8 / 36.3                | 35.1 / 37.6     | 37.6 / 36.2     | 38.5 / 36.1     | 25.5 / 29.3              | 89.0                    |
        | [YOLOE-26x-seg](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt) | 640                         | Text/Visual | 39.5 / 36.2                         | 40.6 / 38.5                | 37.4 / 35.3     | 40.9 / 38.8     | 41.0 / 38.8     | 55.2 / 65.2              | 197.7                   |

    === "Prompt-free"

        | Model            | size<br><sup>(pixels)</sup> | mAP<sup>minival<br>50-95(e2e)</sup> | mAP<sup>minival<br>50(e2e)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
        |------------------|-----------------------------|-------------------------------------|------------------------------|--------------------------|-------------------------|
        | [YOLOE-26n-seg-pf](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26n-seg-pf.pt) | 640                         | 16.6                                | 22.7                         | 2.3                      | 5.3                     |
        | [YOLOE-26s-seg-pf](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg-pf.pt) | 640                         | 21.4                                | 28.6                         | 9.0                      | 20.8                    |
        | [YOLOE-26m-seg-pf](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26m-seg-pf.pt) | 640                         | 25.7                                | 33.6                         | 19.4                     | 68.4                    |
        | [YOLOE-26l-seg-pf](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg-pf.pt) | 640                         | 27.2                                | 35.4                         | 23.6                     | 86.8                    |
        | [YOLOE-26x-seg-pf](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg-pf.pt) | 640                         | 29.9                                | 38.7                         | 53.1                     | 194.4                   |

## Usage Examples

The YOLOE models are easy to integrate into your Python applications. Ultralytics provides user-friendly [Python API](../usage/python.md) and [CLI commands](../usage/cli.md) to streamline development.

### Train Usage

#### Fine-Tuning on custom dataset

You can fine-tune any [pretrained YOLOE model](#textvisual-prompt-models) on your custom YOLO dataset for both detection and instance segmentation tasks.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/vnn90bEyk0w"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train YOLOE on Car Parts Segmentation Dataset | Open-Vocabulary Model, Prediction & Export 🚀
</p>

!!! example

    === "Fine-Tuning"

        **Instance segmentation**

        Fine-tuning a YOLOE pretrained checkpoint mostly follows the [standard YOLO training procedure](../modes/train.md). The key difference is explicitly passing `YOLOEPESegTrainer` as the `trainer` parameter to `model.train()`:

        ```python
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

        model = YOLOE("yoloe-26s-seg.pt")

        # Fine-tune on your segmentation dataset
        results = model.train(
            data="coco128-seg.yaml",  # Segmentation dataset
            epochs=80,
            patience=10,
            trainer=YOLOEPESegTrainer,  # <- Important: use segmentation trainer
        )
        ```

        **Object detection**

        All [pretrained YOLOE models](#textvisual-prompt-models) perform instance segmentation by default. To use these pretrained checkpoints for training a detection model, initialize a detection model from scratch using the YAML configuration, then load the pretrained segmentation checkpoint of the same scale. Note that we use `YOLOEPETrainer` instead of `YOLOEPESegTrainer` since we're training a detection model:

        ```python
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEPETrainer

        # Initialize a detection model from a config
        model = YOLOE("yoloe-26s.yaml")

        # Load weights from a pretrained segmentation checkpoint (same scale)
        model.load("yoloe-26s-seg.pt")

        # Fine-tune on your detection dataset
        results = model.train(
            data="coco128.yaml",  # Detection dataset
            epochs=80,
            patience=10,
            trainer=YOLOEPETrainer,  # <- Important: use detection trainer
        )
        ```

    === "Linear Probing"

        Linear probing fine-tunes only the classification branch while freezing the rest of the model. This approach is useful when working with limited data, as it prevents overfitting by leveraging previously learned features while adapting only the classification head.

        **Instance segmentation**

        ```python
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

        # Load a pretrained segmentation model
        model = YOLOE("yoloe-26s-seg.pt")

        # Identify the head layer index
        head_index = len(model.model.model) - 1

        # Freeze all backbone and neck layers (i.e., everything before the head)
        freeze = [str(i) for i in range(head_index)]

        # Freeze parts of the segmentation head, keeping only the classification branch trainable
        for name, child in model.model.model[-1].named_children():
            if "cv3" not in name:
                freeze.append(f"{head_index}.{name}")

        # Freeze detection branch components
        freeze.extend(
            [
                f"{head_index}.cv3.0.0",
                f"{head_index}.cv3.0.1",
                f"{head_index}.cv3.1.0",
                f"{head_index}.cv3.1.1",
                f"{head_index}.cv3.2.0",
                f"{head_index}.cv3.2.1",
            ]
        )

        # Train only the classification branch
        results = model.train(
            data="coco128-seg.yaml",  # Segmentation dataset
            epochs=80,
            patience=10,
            trainer=YOLOEPESegTrainer,  # <- Important: use segmentation trainer
            freeze=freeze,
        )
        ```

        **Object detection**

        For object detection task, the training process is almost the same as the instance segmentation example above but we use `YOLOEPETrainer` instead of `YOLOEPESegTrainer`, and initialize the object detection model using the YAML and then load the weights from the pretrained instance segmentation checkpoint.

        ```python
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEPETrainer

        # Initialize a detection model from a config
        model = YOLOE("yoloe-26s.yaml")

        # Load weights from a pretrained segmentation checkpoint (same scale)
        model.load("yoloe-26s-seg.pt")

        # Identify the head layer index
        head_index = len(model.model.model) - 1

        # Freeze all backbone and neck layers (i.e., everything before the head)
        freeze = [str(i) for i in range(head_index)]

        # Freeze parts of the segmentation head, keeping only the classification branch trainable
        for name, child in model.model.model[-1].named_children():
            if "cv3" not in name:
                freeze.append(f"{head_index}.{name}")

        # Freeze detection branch components
        freeze.extend(
            [
                f"{head_index}.cv3.0.0",
                f"{head_index}.cv3.0.1",
                f"{head_index}.cv3.1.0",
                f"{head_index}.cv3.1.1",
                f"{head_index}.cv3.2.0",
                f"{head_index}.cv3.2.1",
            ]
        )

        # Train only the classification branch
        results = model.train(
            data="coco128.yaml",  # Detection dataset
            epochs=80,
            patience=10,
            trainer=YOLOEPETrainer,  # <- Important: use detection trainer
            freeze=freeze,
        )
        ```

### Predict Usage

YOLOE supports both text-based and visual prompting. Using prompts is straightforward—just pass them through the `predict` method as shown below:

!!! example

    === "Text Prompt"

        Text prompts allow you to specify the classes that you wish to detect through textual descriptions. The following code shows how you can use YOLOE to detect people and buses in an image:

        ```python
        from ultralytics import YOLOE

        # Initialize a YOLOE model
        model = YOLOE("yoloe-26l-seg.pt")  # or yoloe-26s/m-seg.pt for different sizes

        # Set text prompt to detect person and bus. You only need to do this once after you load the model.
        model.set_classes(["person", "bus"])

        # Run detection on the given image
        results = model.predict("path/to/image.jpg")

        # Show results
        results[0].show()
        ```

    === "Visual Prompt"

        Visual prompts allow you to guide the model by showing it visual examples of the target classes, rather than describing them in text.

        The `visual_prompts` argument takes a dictionary with two keys: `bboxes` and `cls`. Each bounding box in `bboxes` should tightly enclose an example of the object you want the model to detect, and the corresponding entry in `cls` specifies the class label for that box. This pairing tells the model, "This is what class X looks like—now find more like it."

        Class IDs (`cls`) in `visual_prompts` are used to associate each bounding box with a specific category within your prompt. They aren't fixed labels, but temporary identifiers you assign to each example. The only requirement is that class IDs must be sequential, starting from 0. This helps the model correctly associate each box with its respective class.

        You can provide visual prompts directly within the same image you want to run inference on. For example:

        ```python
        import numpy as np

        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        # Initialize a YOLOE model
        model = YOLOE("yoloe-26l-seg.pt")

        # Define visual prompts using bounding boxes and their corresponding class IDs.
        # Each box highlights an example of the object you want the model to detect.
        visual_prompts = {
            "bboxes": np.array(
                [
                    [221.52, 405.8, 344.98, 857.54],  # Box enclosing person
                    [120, 425, 160, 445],  # Box enclosing glasses
                ],
            ),
            "cls": np.array(
                [
                    0,  # ID to be assigned for person
                    1,  # ID to be assigned for glasses
                ]
            ),
        }

        # Run inference on an image, using the provided visual prompts as guidance
        results = model.predict(
            "ultralytics/assets/bus.jpg",
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
        )

        # Show results
        results[0].show()
        ```

        Or you can provide examples from a separate reference image using the `refer_image` argument. In that case, the `bboxes` and `cls` in `visual_prompts` should describe objects in the reference image, not the target image you're making predictions on:

        !!! note

            If `source` is a video or stream, the model automatically uses the first frame as the `refer_image`. This means your `visual_prompts` are applied to that initial frame to help the model understand what to look for in the rest of the video. Alternatively, you can explicitly pass any specific frame as the `refer_image` to control which visual examples the model uses as reference.

        ```python
        import numpy as np

        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        # Initialize a YOLOE model
        model = YOLOE("yoloe-26l-seg.pt")

        # Define visual prompts based on a separate reference image
        visual_prompts = {
            "bboxes": np.array([[221.52, 405.8, 344.98, 857.54]]),  # Box enclosing person
            "cls": np.array([0]),  # ID to be assigned for person
        }

        # Run prediction on a different image, using reference image to guide what to look for
        results = model.predict(
            "ultralytics/assets/zidane.jpg",  # Target image for detection
            refer_image="ultralytics/assets/bus.jpg",  # Reference image used to get visual prompts
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
        )

        # Show results
        results[0].show()
        ```

        Using `refer_image` also sets the classes permanently, so you can run predictions without having to supply the same visual prompts again, and export the model while retaining the ability to still detect the same classes after export:
        ```python
        # After making prediction with `refer_image`, you can run predictions without passing visual_prompts again and still get the same classes back
        results = model("ultralytics/assets/bus.jpg")

        # Or export it to a different format while retaining the classes
        model.export(format="onnx")
        ```

        You can also use PyTorch tensors directly as both the source and `refer_image`, which is useful when images are already in tensor format from an existing pipeline:

        ```python
        import numpy as np
        import torch

        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        model = YOLOE("yoloe-11l-seg.pt")

        # (1, 3, H, W) float tensor in [0, 1], e.g. from an existing preprocessing pipeline
        img_tensor = torch.rand(1, 3, 480, 480)

        # Visual prompts in the tensor's pixel coordinates
        visual_prompts = {
            "bboxes": np.array([[10, 10, 50, 50]]),
            "cls": np.array([0]),
        }

        results = model.predict(
            img_tensor,
            refer_image=img_tensor,
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
            imgsz=640,
        )
        ```

        You can also pass multiple target images to run prediction on:

        ```python
        import numpy as np

        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        # Initialize a YOLOE model
        model = YOLOE("yoloe-26l-seg.pt")

        # Define visual prompts using bounding boxes and their corresponding class IDs.
        # Each box highlights an example of the object you want the model to detect.
        visual_prompts = {
            "bboxes": [
                np.array(
                    [
                        [221.52, 405.8, 344.98, 857.54],  # Box enclosing person
                        [120, 425, 160, 445],  # Box enclosing glasses
                    ],
                ),
                np.array([[150, 200, 1150, 700]]),
            ],
            "cls": [
                np.array(
                    [
                        0,  # ID to be assigned for person
                        1,  # ID to be assigned for glasses
                    ]
                ),
                np.array([0]),
            ],
        }

        # Run inference on multiple images, using the provided visual prompts as guidance
        results = model.predict(
            ["ultralytics/assets/bus.jpg", "ultralytics/assets/zidane.jpg"],
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
        )

        # Show results
        results[0].show()
        ```

    === "Prompt free"

        YOLOE also includes prompt-free variants that come with a built-in vocabulary. These models don't require any prompts and work like traditional YOLO models. Instead of relying on user-provided labels or visual examples, they detect objects from a [predefined list of 4,585 classes](https://github.com/xinyu1205/recognize-anything/blob/main/ram/data/ram_tag_list.txt) based on the tag set used by the [Recognize Anything Model Plus (RAM++)](https://arxiv.org/abs/2310.15200).

        ```python
        from ultralytics import YOLOE

        # Initialize a YOLOE model
        model = YOLOE("yoloe-26l-seg-pf.pt")

        # Run prediction. No prompts required.
        results = model.predict("path/to/image.jpg")

        # Show results
        results[0].show()
        ```

### Val Usage

Model validation on a dataset is streamlined as follows:

!!! example

    === "Text Prompt"

        ```python
        from ultralytics import YOLOE

        # Create a YOLOE model
        model = YOLOE("yoloe-26l-seg.pt")  # or yoloe-26s/m-seg.pt for different sizes

        # Conduct model validation on the COCO128-seg example dataset
        metrics = model.val(data="coco128-seg.yaml")
        ```

    === "Visual Prompt"

        By default it's using the provided dataset to extract visual embeddings for each category.

        ```python
        from ultralytics import YOLOE

        # Create a YOLOE model
        model = YOLOE("yoloe-26l-seg.pt")  # or yoloe-26s/m-seg.pt for different sizes

        # Conduct model validation on the COCO128-seg example dataset
        metrics = model.val(data="coco128-seg.yaml", load_vp=True)
        ```

        Alternatively we could use another dataset as a reference dataset to extract visual embeddings for each category.
        Note this reference dataset should have exactly the same categories as provided dataset.

        ```python
        from ultralytics import YOLOE

        # Create a YOLOE model
        model = YOLOE("yoloe-26l-seg.pt")  # or select yoloe-26s/m-seg.pt for different sizes

        # Conduct model validation on the COCO128-seg example dataset
        metrics = model.val(data="coco128-seg.yaml", load_vp=True, refer_data="coco.yaml")
        ```


    === "Prompt Free"

        ```python
        from ultralytics import YOLOE

        # Create a YOLOE model
        model = YOLOE("yoloe-26l-seg-pf.pt")  # or yoloe-26s/m-seg-pf.pt for different sizes

        # Conduct model validation on the COCO128-seg example dataset
        metrics = model.val(data="coco128-seg.yaml", single_cls=True)
        ```

### Export Usage

The export process is similar to other YOLO models, with the added flexibility of handling text and visual prompts:

!!! warning "Exported models are static"

    Classes configured with `set_classes()` (or via `refer_image` for visual prompts) are baked into the exported weights. Once exported, the model can no longer accept new prompts: calling `set_classes()` or passing `visual_prompts=...` to `predict()` on a loaded export will fail. To change the detected classes, re-export from the original `.pt` checkpoint with the new prompts configured. The exported file behaves like a standard YOLO detector and can also be loaded with `YOLO()` instead of `YOLOE()`.

Prompt embeddings can be saved once and reused when producing static exports such as ONNX, OpenVINO, TensorRT, CoreML, LiteRT, and RKNN. The NPZ profile is loaded by the original PyTorch model before export; it is not an additional runtime input, and the exported model does not require the NPZ file.

!!! example "Reuse prompt embeddings"

    === "Text Prompt"

        ```python
        from ultralytics import YOLOE

        model = YOLOE("yoloe-26n-seg.pt")
        model.set_classes(["person", "bus"])
        model.save_prompt_embeddings("person-bus.npz")

        # The profile is bound to the source checkpoint and can be reused for later exports.
        model = YOLOE("yoloe-26n-seg.pt")
        model.load_prompt_embeddings("person-bus.npz")
        model.export(format="onnx")
        ```

    === "Visual Prompt"

        ```python
        import numpy as np

        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        model = YOLOE("yoloe-26n-seg.pt")
        model.predict(
            "path/to/image.jpg",
            refer_image="path/to/reference.jpg",
            visual_prompts={"bboxes": np.array([[50, 80, 180, 260]]), "cls": np.array([0])},
            predictor=YOLOEVPSegPredictor,
        )
        model.save_prompt_embeddings("visual-object.npz")

        model = YOLOE("yoloe-26n-seg.pt")
        model.load_prompt_embeddings("visual-object.npz")
        model.export(format="engine")
        ```

The same prompt profile can also configure a detection-only model built from the matching YOLOE architecture. This removes the mask branch while retaining the prompted classes:

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-26n.yaml").load("yoloe-26n-seg.pt")
model.load_prompt_embeddings("person-bus.npz")
model.export(format="rknn", name="rk3588", quantize=16)
```

!!! example

    ```python
    from ultralytics import YOLOE

    # Select yoloe-26s/m-seg.pt for different sizes
    model = YOLOE("yoloe-26l-seg.pt")

    # Configure the set_classes() before exporting the model
    model.set_classes(["person", "bus"])

    export_model = model.export(format="onnx")
    model = YOLOE(export_model)

    # Run detection on the given image
    results = model.predict("path/to/image.jpg")

    # Show results
    results[0].show()
    ```

### Track Usage

Prompted classes carry straight into [tracking](../modes/track.md), so you can follow objects the tracker was never trained on:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLOE

        model = YOLOE("yoloe-26s-seg.pt")
        model.set_classes(["forklift", "pallet"])

        # persist=True keeps track IDs stable across frames
        for result in model.track("path/to/video.mp4", stream=True, persist=True):
            print(result.boxes.id)
        ```

    === "CLI"

        ```bash
        yolo track model=yoloe-26s-seg-pf.pt source="path/to/video.mp4"
        ```

## YOLOE Performance Comparison

Zero-shot results on LVIS minival for every prompt-based scale, from the [Ultralytics YOLO26 paper](https://arxiv.org/abs/2606.03748). Each cell is reported as **text prompt / visual prompt**; parameters and FLOPs are for the detection configuration the paper evaluates.

| Model     | mAP<sub>50-95</sub> | mAP<sub>r</sub> | mAP<sub>c</sub> | mAP<sub>f</sub> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| --------- | ------------------- | --------------- | --------------- | --------------- | ------------------------ | ----------------------- |
| YOLOE-v8s | 27.9 / 26.2         | 22.3 / 21.3     | 27.8 / 27.7     | 29.0 / 25.7     | 12.3 / 12.6              | 29.8                    |
| YOLOE-v8m | 32.6 / 31.0         | 26.9 / 27.0     | 31.9 / 31.7     | 34.4 / 31.1     | 26.4 / 28.4              | 80.7                    |
| YOLOE-v8l | 35.9 / 34.2         | 33.2 / 33.2     | 34.8 / 34.6     | 37.3 / 34.1     | 43.5 / 47.3              | 167.6                   |
| YOLOE-11s | 27.5 / 26.3         | 21.4 / 22.5     | 26.8 / 27.1     | 29.3 / 26.4     | 10.7 / 10.9              | 22.7                    |
| YOLOE-11m | 33.0 / 31.4         | 26.9 / 27.1     | 32.5 / 31.9     | 34.5 / 31.7     | 21.0 / 24.8              | 70.4                    |
| YOLOE-11l | 35.2 / 33.7         | 29.1 / 28.1     | 35.0 / 34.6     | 36.5 / 33.8     | 26.0 / 29.8              | 89.5                    |
| YOLOE-26n | 24.7 / 21.9         | 20.5 / 17.6     | 24.1 / 22.3     | 26.1 / 22.4     | 3.9 / 3.1                | 6.1                     |
| YOLOE-26s | 30.8 / 28.6         | 23.9 / 25.1     | 29.6 / 27.8     | 33.0 / 29.9     | 10.7 / 11.0              | 21.9                    |
| YOLOE-26m | 35.4 / 33.9         | 31.1 / 33.4     | 34.7 / 34.0     | 36.9 / 33.8     | 21.3 / 25.1              | 70.6                    |
| YOLOE-26l | 37.8 / 36.3         | 35.1 / 37.6     | 37.6 / 36.2     | 38.5 / 36.1     | 25.5 / 29.3              | 89.0                    |
| YOLOE-26x | 40.6 / 38.5         | 37.4 / 35.3     | 40.9 / 38.8     | 41.0 / 38.8     | 55.2 / 65.2              | 197.7                   |

At every scale the YOLOE-26 models lead their YOLOE-11 and YOLOE-v8 counterparts on mAP<sub>50-95</sub> while staying below the v8 line on parameters and FLOPs. On the same split the paper reports YOLO-Worldv2 at 24.4 (S), 32.4 (M) and 35.5 (L), and the transformer-based detectors GLIP-T at 26.0, GDINO-T at 27.4 and DetCLIP-T at 34.4, each carrying 155 to 232 M parameters.

!!! note "These are not the counts for the checkpoint you download"

    The paper evaluates a detection configuration. The released weights are **segmentation** checkpoints and carry a mask branch, SAVPE and the text projection on top, so a loaded `yoloe-26l-seg.pt` reports 35.4 M and 142.0 B rather than the 25.5 M and 89.0 B above. In both cases the FLOPs figure excludes the region-text similarity, so real cost grows with the size of the prompt set even though the column does not move; see [Deployment Notes](#deployment-notes).

## Comparison with Previous Models

- **vs closed-set YOLO.** Once the prompts are set, YOLOE predicts through the ordinary detect/segment path and exports like any other model. What it adds is the ability to change the class list at inference time instead of retraining. What it costs is zero-shot accuracy well below a model trained on your own classes; see [Limitations](#limitations).
- **vs YOLOE on YOLO11.** YOLOE-26 inherits [YOLO26](yolo26.md)'s NMS-free end-to-end head and covers five scales (n/s/m/l/x) against the earlier three (s/m/l). Its per-scale LVIS minival figures are in [Available Models](#available-models-supported-tasks-and-operating-modes).
- **vs transformer-based open-vocabulary detectors.** GLIP and OWL-ViT run a vision-language transformer at inference. YOLOE encodes the prompts once and then compares them against region features inside a convolutional head.

## Use Cases and Applications

Open-vocabulary detection removes the retrain-per-class step, which matters most where the target list is not known up front:

- **Open-world detection** — [robotics](https://www.ultralytics.com/blog/understanding-the-integration-of-computer-vision-in-robotics) and [security systems](https://www.ultralytics.com/blog/computer-vision-for-theft-prevention-enhancing-security) that meet objects nobody enumerated at training time.
- **One-shot detection from an example** — visual prompts pick up a specific part, logo or defect from a single reference box, useful in [industrial inspection](https://www.ultralytics.com/blog/computer-vision-in-manufacturing-improving-production-and-quality).
- **Long-tail cataloguing** — the prompt-free vocabulary covers 4,585 tag names, enough for [biodiversity monitoring](https://www.ultralytics.com/blog/ai-in-wildlife-conservation) or [retail inventory](https://www.ultralytics.com/blog/ai-for-smarter-retail-inventory-management) sweeps.
- **Dataset bootstrapping** — pre-label images with boxes and masks before human review, then train a fast closed-set model on the result.
- **Segmentation of arbitrary targets** — the released `*-seg.pt` checkpoints return a mask with every prediction, so [medical imaging](https://www.ultralytics.com/blog/ai-and-radiology-a-new-era-of-precision-and-efficiency) and [satellite analysis](https://www.ultralytics.com/blog/using-computer-vision-to-analyze-satellite-imagery) get pixel-precise output without a second model.

A common pattern combines two modes: run prompt-free once to discover what is present, then switch to text prompts for the categories that matter.

## Limitations

- **Zero-shot accuracy is well below a model trained on your classes.** The prompted checkpoints land roughly in the 25-40 mAP band on LVIS minival (see the per-scale table above); a closed-set YOLO trained on your own data will beat that on those classes. Reach for YOLOE to cover classes you cannot train for, not to replace training.
- **Rare categories are the weak spot.** The `mAP_r` column reports accuracy on LVIS's rare classes specifically, and it sits below the common and frequent columns across the range. Check it rather than the headline mAP when your targets are unusual.
- **A prompt describes appearance, not relationships.** Detection works by comparing region features against the prompt embedding, so prompts that depend on state, context or comparison — "damaged", "left-most", "the one being carried" — have no reliable handle to match on.
- **Large prompt sets cost latency.** Measured on `yoloe-26s-seg`, a CPU forward pass grows about 19% from 80 to 1,203 classes and about 89% at the full 4,585-name vocabulary. Reported FLOPs stay flat, so the profile will not warn you.
- **Exported models are frozen.** Classes are baked in at export time; changing them means re-exporting from the `.pt` checkpoint.

## YOLOE vs SAM 3 vs YOLO-World

All three take a text prompt, but only YOLOE and SAM 3 return masks, and they answer different questions:

|                  | [YOLOE](#quick-start)                                 | [SAM 3](sam-3.md)                                                     | [YOLO-World](yolo-world.md)                          |
| ---------------- | ----------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------- |
| Built for        | Real-time detection and segmentation of named classes | Concept segmentation and promptable tracking                          | Real-time open-vocabulary detection                  |
| Masks            | Yes, with the `*-seg.pt` checkpoints                  | Yes                                                                   | No, boxes only                                       |
| Visual prompts   | Yes (SAVPE)                                           | Yes                                                                   | No                                                   |
| Prompt-free mode | Yes, 4,585-name vocabulary                            | No                                                                    | No                                                   |
| Pick it when     | You need throughput and can name the classes          | You need the strongest concept segmentation and can spend the compute | You are already on it — see the migration note below |

**Coming from YOLO-World?** The API is the same shape: swap `YOLOWorld` for `YOLOE`, load a `*-seg.pt` checkpoint, and keep your `set_classes()` call as it is. You gain masks and visual prompts; the [export note](#export-usage) about frozen classes applies to both.

## Deployment Notes

- **Hardware.** Inference wants an NVIDIA GPU with 4-8 GB of VRAM; the `n` and `s` scales run on edge GPUs such as [Jetson](../guides/nvidia-jetson.md) or on CPU at reduced resolution. Fine-tuning needs a single GPU. The authors' open-vocabulary pre-training used 8x RTX 4090.
- **Class names are placeholders until you prompt.** A freshly loaded `*-seg.pt` checkpoint reports `nc=80` with numeric names (`"0"`, `"1"`, …), so call `set_classes()` before reading labels. Prompt-free checkpoints ship the full 4,585-name vocabulary already populated.
- **Prompt cost scales with the number of classes.** The prompt embeddings are computed once and stored on the model, but they are compared against region features on every forward pass, so a short prompt list is close to free and a very large one is not. Measured on `yoloe-26s-seg` on CPU, a forward pass grows about 19% going from 80 to 1,203 classes and about 89% going to the full 4,585. Reported FLOPs do not move at all, because the region-text similarity is not counted.
- **NMS behavior.** YOLOE automatically uses `agnostic_nms=True` during prediction, suppressing lower-scoring overlapping boxes across different classes rather than only within the same class. This prevents duplicate detections when the same object matches multiple categories in YOLOE's large vocabulary. On end-to-end YOLOE-26 models it only prevents the same detection from appearing under multiple class labels (IoU=1.0 duplicates) and performs no IoU-threshold suppression between distinct boxes. You can override this by passing `agnostic_nms=False` explicitly.
- **Batch inference** works directly (`model.predict([img1, img2])`), and visual prompts can differ per image in the same call by passing one `bboxes`/`cls` array per source image.

!!! tip

    Fine-tune from a provided checkpoint rather than training from scratch, and prefer prompt wording close to everyday category names — rare phrasings are where open-vocabulary accuracy falls off.

## Training the Official Models from Scratch

Most readers never need this. It reproduces the published open-vocabulary checkpoints from Objects365, GQA and Flickr30k — hundreds of thousands of images on 8x RTX 4090 — and is unrelated to fine-tuning on your own data, which is covered under [Train Usage](#train-usage) above.

!!! warning

    These trainers inherit `YOLOETrainer`, which refuses `compile=True`. Pass `compile=False` (the default) here. The fine-tuning trainers above do not carry that restriction.

### Prepare datasets

!!! note

    Training official YOLOE models needs segment annotations for train data, here's [the script provided by official team](https://github.com/THU-MIG/yoloe/blob/main/tools/generate_sam_masks.py) that converts datasets to segment annotations, powered by [SAM2.1 models](./sam-2.md). Or you can directly download the provided `Processed Segment Annotations` in following table provided by official team.

- Train data

| Dataset                                                           | Type                                                        | Samples | Boxes | Raw Detection Annotations                                                                                                                     | Processed Segment Annotations                                                                                                                   |
| ----------------------------------------------------------------- | ----------------------------------------------------------- | ------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| [Objects365v1](https://opendatalab.com/OpenDataLab/Objects365_v1) | Detection                                                   | 609k    | 9621k | [objects365_train.json](https://opendatalab.com/OpenDataLab/Objects365_v1)                                                                    | [objects365_train_segm.json](https://huggingface.co/datasets/jameslahm/yoloe/resolve/main/objects365_train_segm.json)                           |
| [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html)      | [Grounding](https://www.ultralytics.com/glossary/grounding) | 621k    | 3681k | [final_mixed_train_no_coco.json](https://huggingface.co/GLIPModel/GLIP/resolve/main/mdetr_annotations/final_mixed_train_no_coco.json)         | [final_mixed_train_no_coco_segm.json](https://huggingface.co/datasets/jameslahm/yoloe/resolve/main/final_mixed_train_no_coco_segm.json)         |
| [Flickr30k](https://github.com/BryanPlummer/flickr30k_entities)   | Grounding                                                   | 149k    | 641k  | [final_flickr_separateGT_train.json](https://huggingface.co/GLIPModel/GLIP/resolve/main/mdetr_annotations/final_flickr_separateGT_train.json) | [final_flickr_separateGT_train_segm.json](https://huggingface.co/datasets/jameslahm/yoloe/resolve/main/final_flickr_separateGT_train_segm.json) |

- Val data

| Dataset                                                                                                 | Type      | Annotation Files                                                                                       |
| ------------------------------------------------------------------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------ |
| [LVIS minival](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml) | Detection | [minival.txt](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml) |

### Launching training from scratch

!!! note

    `Visual Prompt` models are fine-tuned based on trained-well `Text Prompt` models.

!!! example

    === "Text Prompt"

        ```python
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOESegTrainerFromScratch

        # Option 1: Use Python dictionary
        data = {
            "train": {
                "yolo_data": ["Objects365.yaml"],
                "grounding_data": [
                    {
                        "img_path": "flickr/full_images/",
                        "json_file": "flickr/annotations/final_flickr_separateGT_train_segm.json",
                    },
                    {
                        "img_path": "mixed_grounding/gqa/images",
                        "json_file": "mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
                    },
                ],
            },
            "val": {"yolo_data": ["lvis.yaml"]},
        }

        # Option 2: Use YAML file (yoloe_data.yaml)
        # train:
        #   yolo_data:
        #     - Objects365.yaml
        #   grounding_data:
        #     - img_path: flickr/full_images/
        #       json_file: flickr/annotations/final_flickr_separateGT_train_segm.json
        #     - img_path: mixed_grounding/gqa/images
        #       json_file: mixed_grounding/annotations/final_mixed_train_no_coco_segm.json
        # val:
        #   yolo_data:
        #     - lvis.yaml

        model = YOLOE("yoloe-26l-seg.yaml")
        model.train(
            data=data,  # or data="yoloe_data.yaml" if using YAML file
            batch=128,
            epochs=30,
            close_mosaic=2,
            optimizer="AdamW",
            lr0=2e-3,
            warmup_bias_lr=0.0,
            weight_decay=0.025,
            momentum=0.9,
            workers=4,
            trainer=YOLOESegTrainerFromScratch,
            device="0,1,2,3,4,5,6,7",
        )
        ```

    === "Visual Prompt"

        Since only the `SAVPE` module needs to be updated during training.
        Converting trained-well Text-prompt model to detection model and adopt detection pipeline with less training cost.
        Note this step is optional, you can directly start from segmentation as well.

        ```python
        from ultralytics import YOLOE
        from ultralytics.utils.patches import torch_load

        det_model = YOLOE("yoloe-26l.yaml")
        state = torch_load("yoloe-26l-seg.pt")
        det_model.load(state["model"])
        det_model.save("yoloe-26l-seg-det.pt")
        ```

        Start training:

        ```python
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOESegVPTrainer

        data = {
            "train": {
                "yolo_data": ["Objects365.yaml"],
                "grounding_data": [
                    {
                        "img_path": "flickr/full_images/",
                        "json_file": "flickr/annotations/final_flickr_separateGT_train_segm.json",
                    },
                    {
                        "img_path": "mixed_grounding/gqa/images",
                        "json_file": "mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
                    },
                ],
            },
            "val": {"yolo_data": ["lvis.yaml"]},
        }

        model = YOLOE("yoloe-26l-seg.pt")
        # replace to yoloe-26l-seg-det.pt if converted to detection model
        # model = YOLOE("yoloe-26l-seg-det.pt")

        # freeze every layer except of the savpe module.
        head_index = len(model.model.model) - 1
        freeze = list(range(head_index))
        for name, child in model.model.model[-1].named_children():
            if "savpe" not in name:
                freeze.append(f"{head_index}.{name}")

        model.train(
            data=data,
            batch=128,
            epochs=2,
            close_mosaic=2,
            optimizer="AdamW",
            lr0=16e-3,
            warmup_bias_lr=0.0,
            weight_decay=0.025,
            momentum=0.9,
            workers=4,
            trainer=YOLOESegVPTrainer,  # use YOLOEVPTrainer if converted to detection model
            device="0,1,2,3,4,5,6,7",
            freeze=freeze,
        )
        ```

        Convert back to segmentation model after training. Only needed if you converted segmentation model to detection model before training.

        ```python
        from copy import deepcopy

        from ultralytics import YOLOE

        model = YOLOE("yoloe-26l-seg.yaml")
        model.load("yoloe-26l-seg.pt")

        # Weights written by the SAVPE training run above. Each rerun creates a new
        # directory (train-2, train-3, ...), so take the path the run printed.
        vp_model = YOLOE("runs/segment/train/weights/best.pt")
        model.model.model[-1].savpe = deepcopy(vp_model.model.model[-1].savpe)
        model.eval()
        model.save("yoloe-26l-seg-vp.pt")  # never overwrite the released checkpoint
        ```

    === "Prompt Free"

        Similar to visual prompt training, for prompt-free model there's only the specialized prompt embedding needs to be updated during training.
        Converting trained-well Text-prompt model to detection model and adopt detection pipeline with less training cost.
        Note this step is optional, you can directly start from segmentation as well.

        ```python
        from ultralytics import YOLOE
        from ultralytics.utils.patches import torch_load

        det_model = YOLOE("yoloe-26l.yaml")
        state = torch_load("yoloe-26l-seg.pt")
        det_model.load(state["model"])
        det_model.save("yoloe-26l-seg-det.pt")
        ```
        Start training:
        ```python
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEPEFreeTrainer

        data = {
            "train": {
                "yolo_data": ["Objects365.yaml"],
                "grounding_data": [
                    {
                        "img_path": "flickr/full_images/",
                        "json_file": "flickr/annotations/final_flickr_separateGT_train_segm.json",
                    },
                    {
                        "img_path": "mixed_grounding/gqa/images",
                        "json_file": "mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
                    },
                ],
            },
            "val": {"yolo_data": ["lvis.yaml"]},
        }

        model = YOLOE("yoloe-26l-seg.pt")
        # replace to yoloe-26l-seg-det.pt if converted to detection model
        # model = YOLOE("yoloe-26l-seg-det.pt")

        # freeze layers.
        head_index = len(model.model.model) - 1
        freeze = [str(f) for f in range(head_index)]
        for name, child in model.model.model[-1].named_children():
            if "cv3" not in name:
                freeze.append(f"{head_index}.{name}")

        freeze.extend(
            [
                f"{head_index}.cv3.0.0",
                f"{head_index}.cv3.0.1",
                f"{head_index}.cv3.1.0",
                f"{head_index}.cv3.1.1",
                f"{head_index}.cv3.2.0",
                f"{head_index}.cv3.2.1",
            ]
        )

        model.train(
            data=data,
            batch=128,
            epochs=1,
            close_mosaic=1,
            optimizer="AdamW",
            lr0=2e-3,
            warmup_bias_lr=0.0,
            weight_decay=0.025,
            momentum=0.9,
            workers=4,
            trainer=YOLOEPEFreeTrainer,
            device="0,1,2,3,4,5,6,7",
            freeze=freeze,
            single_cls=True,  # this is needed
        )
        ```

        Convert back to segmentation model after training. Only needed if you converted segmentation model to detection model before training.

        ```python
        from copy import deepcopy

        from ultralytics import YOLOE

        model = YOLOE("yoloe-26l-seg.pt")
        model.eval()

        pf_model = YOLOE("yoloe-26l-seg-pf.pt")
        names = ["object"]
        tpe = model.get_text_pe(names)
        model.set_classes(names, tpe)
        model.model.model[-1].fuse(model.model.pe)

        model.model.model[-1].cv3[0][2] = deepcopy(pf_model.model.model[-1].cv3[0][2]).requires_grad_(True)
        model.model.model[-1].cv3[1][2] = deepcopy(pf_model.model.model[-1].cv3[1][2]).requires_grad_(True)
        model.model.model[-1].cv3[2][2] = deepcopy(pf_model.model.model[-1].cv3[2][2]).requires_grad_(True)
        del model.model.pe
        model.save("yoloe-26l-seg-pf-custom.pt")  # never overwrite the released checkpoint
        ```

## Citations and Acknowledgments

If YOLOE has contributed to your research or project, please cite the original paper by **Ao Wang, Lihao Liu, Hui Chen, Zijia Lin, Jungong Han, and Guiguang Ding** from **Tsinghua University**:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{wang2025yoloerealtimeseeing,
              title={YOLOE: Real-Time Seeing Anything},
              author={Ao Wang and Lihao Liu and Hui Chen and Zijia Lin and Jungong Han and Guiguang Ding},
              year={2025},
              eprint={2503.07465},
              archivePrefix={arXiv},
              primaryClass={cs.CV},
              url={https://arxiv.org/abs/2503.07465},
        }
        ```

For further reading, the original YOLOE paper is available on [arXiv](https://arxiv.org/html/2503.07465v1). The project's source code and additional resources can be accessed via their [GitHub repository](https://github.com/THU-MIG/yoloe).

## FAQ

### How does YOLOE differ from YOLO-World?

Both do open-vocabulary detection; YOLOE adds two things YOLO-World does not have. It accepts **visual prompts** (an example box instead of a class name) and ships **prompt-free** checkpoints with a built-in 4,585-name vocabulary, where YOLO-World is text-only. Every prediction from the released `*-seg.pt` checkpoints also carries an [instance segmentation](https://www.ultralytics.com/blog/what-is-instance-segmentation-a-quick-guide) mask. On accuracy, the paper reports YOLOE-v8-S ahead of YOLO-Worldv2-S by 3.5 AP on LVIS, at a third of the training cost and 1.4× the inference speed. Migrating is a one-line change — see [YOLOE vs SAM 3 vs YOLO-World](#yoloe-vs-sam-3-vs-yolo-world).

### Can I use YOLOE as a regular YOLO model?

Yes. `set_classes()` stores the prompt embeddings on the model, and inference then follows the ordinary detect/segment path and exports like any other model. Classes only become static at export time. Two caveats worth knowing: the released YOLOE checkpoints are **segmentation** models, so compare them against `yolo26*-seg`, not against the detection variants; and a freshly loaded checkpoint has numeric placeholder class names until you call `set_classes()`.

### What types of prompts can I use with YOLOE?

YOLOE supports three types of prompts:

1. **Text prompts**: Specify object classes using natural language (e.g., "person", "traffic light", "bird scooter")
2. **Visual prompts**: Provide reference images of objects you want to detect
3. **Internal vocabulary**: Use the prompt-free checkpoints' built-in vocabulary of 4,585 tag names without external prompts

This flexibility allows you to adapt YOLOE to various scenarios without retraining the model, making it particularly useful for dynamic environments where detection requirements change frequently.

### How does YOLOE handle instance segmentation?

YOLOE integrates instance segmentation directly into its architecture by extending the detection head with a mask prediction branch. This approach is similar to YOLOv8-Seg but works for any prompted object class. Segmentation masks are automatically included in inference results and can be accessed via `results[0].masks`. This unified approach eliminates the need for separate detection and segmentation models, streamlining workflows for applications requiring pixel-precise object boundaries.

### How does YOLOE handle inference with custom prompts?

Similar to [YOLO-World](yolo-world.md), YOLOE supports a "prompt-then-detect" strategy that utilizes an offline vocabulary to enhance efficiency. Custom prompts like captions or specific object categories are pre-encoded and stored as offline vocabulary embeddings. This approach streamlines the detection process without requiring retraining. You can dynamically set these prompts within the model to tailor it to specific detection tasks:

```python
from ultralytics import YOLO

# Initialize a YOLOE model
model = YOLO("yoloe-26s-seg.pt")

# Define custom classes
model.set_classes(["person", "bus"])

# Execute prediction on an image
results = model.predict("path/to/image.jpg")

# Show results
results[0].show()
```

### Why are my detections labelled object0 and object1 instead of my class names?

That is visual-prompt mode. The class IDs you pass in `visual_prompts["cls"]` only group the example boxes into temporary classes; they do not carry your names. The model reports them as `object0`, `object1`, and so on, in the order of the IDs you assigned, so map them back to your own labels on the result. If you want your names in the output, use a text prompt instead.

### Why does set_classes() fail on a prompt-free checkpoint?

Prompt-free checkpoints (`*-seg-pf.pt`) resolve classes through their own built-in vocabulary and reject external prompts with `AssertionError: Prompt-free model does not support setting classes.` Load a `*-seg.pt` checkpoint when you need your own class list. See [Choosing a Prompting Mode](#choosing-a-prompting-mode).

### What does YOLOE download the first time I run a text prompt?

It installs [ultralytics/CLIP](https://github.com/ultralytics/CLIP) from GitHub with `pip` and fetches a TorchScript text encoder into the Ultralytics weights directory — see [Installation and Requirements](#installation-and-requirements) for the exact asset per model family. Visual prompts and prompt-free checkpoints need neither. To avoid the download on the target machine, set the prompts once and save them with `save_prompt_embeddings()`, or export the model with the classes already configured.

### Can I change the classes of an exported YOLOE model?

No. Classes are baked into the weights at export time, and calling `set_classes()` or passing `visual_prompts=` to a loaded export fails. Re-export from the original `.pt` checkpoint with the new prompts configured. The exported file behaves like a standard YOLO model and can be loaded with `YOLO()` as well as `YOLOE()`.

### Should I use YOLOE or SAM 3?

Use YOLOE when you need real-time throughput and can name the classes, and [SAM 3](sam-3.md) when segmentation quality on a concept matters more than speed. Both accept visual examples; only YOLOE has a prompt-free mode. The full comparison is in [YOLOE vs SAM 3 vs YOLO-World](#yoloe-vs-sam-3-vs-yolo-world).
