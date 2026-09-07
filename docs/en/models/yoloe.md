---
title: YOLOE Open-Vocabulary Detection & Segmentation
comments: true
description: Ultralytics YOLOE detects and segments any object class from a text prompt, a visual example, or a built-in vocabulary, in real time and with no retraining.
keywords: YOLOE, YOLOE-26, open-vocabulary detection, open-vocabulary segmentation, zero-shot detection, instance segmentation, text prompts, visual prompts, prompt-free detection, YOLO26, real-time object detection
---

# YOLOE: Real-Time Open-Vocabulary Detection and Segmentation

**Ultralytics YOLOE** (Real-Time Seeing Anything) is an **open-vocabulary** detection and [instance segmentation](../tasks/segment.md) model: instead of a class list fixed at training time, it takes the categories you want at inference time, as a text prompt, a visual example, or a built-in 4,585-name vocabulary. Built on the Ultralytics YOLO architectures — YOLOv8, [YOLO11](yolo11.md) and [YOLO26](yolo26.md) — and inspired by [YOLO-World](yolo-world.md), YOLOE reaches state-of-the-art zero-shot accuracy at close to closed-set YOLO speed.

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

        # "double-decker bus" is not a COCO class; YOLOE resolves it from the words alone
        model.set_classes(["double-decker bus", "person"])

        results = model.predict("https://ultralytics.com/images/bus.jpg")
        results[0].show()
        ```

    === "CLI"

        ```bash
        yolo predict model=yoloe-26s-seg.pt source="https://ultralytics.com/images/bus.jpg" classes="double-decker bus,person"
        ```

The first `set_classes()` call downloads a text encoder; see [Installation and Requirements](#installation-and-requirements) before deploying to a machine without network access.

## Choosing a Prompting Mode

YOLOE supports three prompting modes, and the choice decides which checkpoint you load and what your class labels look like. Pick the row that matches what you can supply at inference time.

![YOLOE detecting and segmenting objects from text prompts, visual prompts, and its prompt-free vocabulary](https://cdn.ul.run/i/b144f94979e9ca2ae30bcbb41c69816d.avif)

| Mode              | Checkpoint    | You supply                         | Class names in the results                    | Use it when                                                                 |
| ----------------- | ------------- | ---------------------------------- | --------------------------------------------- | --------------------------------------------------------------------------- |
| **Text prompt**   | `*-seg.pt`    | Class names as strings             | Exactly the names you passed                  | You can describe the target in words — the usual choice                     |
| **Visual prompt** | `*-seg.pt`    | Example boxes on a reference image | Generic `object0`, `object1`, …               | You cannot put the target into words: a specific part, logo, or defect      |
| **Prompt-free**   | `*-seg-pf.pt` | Nothing                            | Names from the built-in 4,585-name vocabulary | You are cataloging or exploring and do not know what to look for in advance |

!!! warning "Two Common Surprises"

    - **Visual prompts do not carry your labels.** The class IDs in `visual_prompts` group examples together; the model reports them as `object0`, `object1`, and so on. Map them back to your own names yourself.
    - **Prompt-free checkpoints reject `set_classes()`.** Calling it on a `*-seg-pf.pt` model raises `AssertionError: Prompt-free model does not support setting classes. Please try with Text/Visual prompt models.` Load a `*-seg.pt` checkpoint instead when you need your own classes.

## Installation and Requirements

YOLOE ships in the main Ultralytics package:

```bash
pip install -U ultralytics
```

Text prompting needs a text encoder on top of that, and it is fetched on first use rather than at install time:

- The first `set_classes()` call installs [ultralytics/CLIP](https://github.com/ultralytics/CLIP) from GitHub with `pip` (it provides the tokenizer) and downloads a TorchScript text encoder **into the current working directory**. YOLOE-26 pulls `mobileclip2_b.ts`, about 254 MB; YOLOE-11 and YOLOE-v8 pull `mobileclip_blt.ts`. Run the download once from the directory you will run from, or copy the file there, otherwise it is fetched again.
- Both steps need network access, so run one prompted prediction before deploying to an offline or air-gapped machine.
- Visual prompts and prompt-free checkpoints need no text encoder at all.

The YOLOE-26 checkpoints require `ultralytics` **8.4.0** or newer; the YOLOE-11 and YOLOE-v8 families are available in earlier releases. One text-prompted prediction exercises the whole path — checkpoint download, CLIP install, text encoder, inference:

```bash
yolo predict model=yoloe-26s-seg.pt source="https://ultralytics.com/images/bus.jpg" classes="person"
```

To skip the text-encoder download at inference time entirely, bake the prompts into the weights once and reuse them — see [Reuse prompt embeddings](#export-usage).

## Architecture Overview

<p align="center">
  <img src="https://github.com/THU-MIG/yoloe/raw/main/figures/pipeline.svg" alt="YOLOE Architecture" width=90%>
</p>

YOLOE keeps the standard YOLO structure — a convolutional **backbone** for feature extraction, a **neck** for multi-scale fusion, and an **anchor-free, decoupled head** predicting classes and boxes — and adds three modules, one per prompting mode:

- **Re-parameterizable Region-Text Alignment (RepRTA)** refines text [embeddings](https://www.ultralytics.com/glossary/embeddings) from CLIP through a small auxiliary network. That network runs once per `set_classes()` call and is folded away at export, so it costs nothing per frame. What does run on every forward pass is the comparison of the stored prompt embeddings against region features; see [Limitations](#limitations) for what that costs with a large prompt set.
- **Semantic-Activated Visual Prompt Encoder (SAVPE)** encodes semantic and activation features from an example box, conditioning the model on objects that look like it. This is the one-shot path for targets that are hard to name, such as a logo or a specific part.
- **Lazy Region-Prompt Contrast (LRPC)** matches region embeddings against a built-in 4,585-name vocabulary, so prompt-free checkpoints recognize objects with no external prompt and no text encoder.

[Instance segmentation](../tasks/segment.md) comes from a mask branch on the detection head, as in YOLOv8-Seg, and every prediction carries a mask on `results[0].masks`. Once the model is [exported](#export-usage), the open-world modules are re-parameterized into a standard YOLO head, so the exported file runs the ordinary detect/segment path.

## Available Models

Every checkpoint below is an [instance segmentation](../tasks/segment.md) model and supports [val](../modes/val.md), [predict](../modes/predict.md), [export](../modes/export.md) and [track](../modes/track.md). Load a `*-seg.pt` file for text or visual prompting and a `*-seg-pf.pt` file for prompt-free inference; they are not interchangeable, see [Choosing a Prompting Mode](#choosing-a-prompting-mode). Only the `*-seg.pt` files support [train](../modes/train.md); a prompt-free checkpoint is produced from a trained text-prompt model, see [Training the Official Models from Scratch](#training-the-official-models-from-scratch).

| Model     | Text / visual prompt                                                                                | Prompt-free                                                                                               |
| --------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| YOLOE-26n | [yoloe-26n-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26n-seg.pt) | [yoloe-26n-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26n-seg-pf.pt) |
| YOLOE-26s | [yoloe-26s-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt) | [yoloe-26s-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg-pf.pt) |
| YOLOE-26m | [yoloe-26m-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26m-seg.pt) | [yoloe-26m-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26m-seg-pf.pt) |
| YOLOE-26l | [yoloe-26l-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg.pt) | [yoloe-26l-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg-pf.pt) |
| YOLOE-26x | [yoloe-26x-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt) | [yoloe-26x-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg-pf.pt) |
| YOLOE-11s | [yoloe-11s-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11s-seg.pt) | [yoloe-11s-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11s-seg-pf.pt) |
| YOLOE-11m | [yoloe-11m-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11m-seg.pt) | [yoloe-11m-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11m-seg-pf.pt) |
| YOLOE-11l | [yoloe-11l-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11l-seg.pt) | [yoloe-11l-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11l-seg-pf.pt) |
| YOLOE-v8s | [yoloe-v8s-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-v8s-seg.pt) | [yoloe-v8s-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-v8s-seg-pf.pt) |
| YOLOE-v8m | [yoloe-v8m-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-v8m-seg.pt) | [yoloe-v8m-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-v8m-seg-pf.pt) |
| YOLOE-v8l | [yoloe-v8l-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-v8l-seg.pt) | [yoloe-v8l-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-v8l-seg-pf.pt) |

## YOLOE Performance on LVIS

Zero-shot results on [LVIS](../datasets/detect/lvis.md) minival at 640 pixels, from the [Ultralytics YOLO26 paper](https://arxiv.org/abs/2606.03748).

### Text and visual prompts

Each accuracy and parameter cell reads **text prompt / visual prompt**; FLOPs are given once. Parameters and FLOPs are for the detection configuration the paper evaluates. Accuracy is the paper's Non-E2E figure, the only protocol it reports for every model in the comparison; the YOLOE-26 end-to-end head trails it by at most 1.1 AP under text prompts and 2.6 AP under visual prompts.

| Model     | mAP<sub>50-95</sub> | mAP<sub>r</sub> | mAP<sub>c</sub> | mAP<sub>f</sub> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| --------- | ------------------- | --------------- | --------------- | --------------- | ------------------------ | ----------------------- |
| YOLOE-26n | 24.7 / 21.9         | 20.5 / 17.6     | 24.1 / 22.3     | 26.1 / 22.4     | 3.9 / 3.1                | 6.1                     |
| YOLOE-26s | 30.8 / 28.6         | 23.9 / 25.1     | 29.6 / 27.8     | 33.0 / 29.9     | 10.7 / 11.0              | 21.9                    |
| YOLOE-26m | 35.4 / 33.9         | 31.1 / 33.4     | 34.7 / 34.0     | 36.9 / 33.8     | 21.3 / 25.1              | 70.6                    |
| YOLOE-26l | 37.8 / 36.3         | 35.1 / 37.6     | 37.6 / 36.2     | 38.5 / 36.1     | 25.5 / 29.3              | 89.0                    |
| YOLOE-26x | 40.6 / 38.5         | 37.4 / 35.3     | 40.9 / 38.8     | 41.0 / 38.8     | 55.2 / 65.2              | 197.7                   |
| YOLOE-11s | 27.5 / 26.3         | 21.4 / 22.5     | 26.8 / 27.1     | 29.3 / 26.4     | 10.7 / 10.9              | 22.7                    |
| YOLOE-11m | 33.0 / 31.4         | 26.9 / 27.1     | 32.5 / 31.9     | 34.5 / 31.7     | 21.0 / 24.8              | 70.4                    |
| YOLOE-11l | 35.2 / 33.7         | 29.1 / 28.1     | 35.0 / 34.6     | 36.5 / 33.8     | 26.0 / 29.8              | 89.5                    |
| YOLOE-v8s | 27.9 / 26.2         | 22.3 / 21.3     | 27.8 / 27.7     | 29.0 / 25.7     | 12.3 / 12.6              | 29.8                    |
| YOLOE-v8m | 32.6 / 31.0         | 26.9 / 27.0     | 31.9 / 31.7     | 34.4 / 31.1     | 26.4 / 28.4              | 80.7                    |
| YOLOE-v8l | 35.9 / 34.2         | 33.2 / 33.2     | 34.8 / 34.6     | 37.3 / 34.1     | 43.5 / 47.3              | 167.6                   |

### Prompt-free

The prompt-free checkpoints answer from their built-in vocabulary with no prompt supplied. Each accuracy cell reads **end-to-end / Non-E2E**, the two protocols the paper scores YOLOE-26 under; the [YOLO26 page](yolo26.md#yoloe-26-open-vocabulary-detection-and-segmentation) quotes the Non-E2E column.

| Model        | mAP<sub>50-95</sub> | mAP<sub>r</sub> | mAP<sub>c</sub> | mAP<sub>f</sub> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ------------ | ------------------- | --------------- | --------------- | --------------- | ------------------------ | ----------------------- |
| YOLOE-26n-pf | 16.6 / 17.7         | 15.7 / 15.8     | 15.3 / 16.4     | 17.9 / 19.2     | 2.3                      | 5.3                     |
| YOLOE-26s-pf | 21.4 / 22.6         | 16.2 / 20.2     | 20.1 / 20.9     | 23.5 / 24.5     | 9.0                      | 20.8                    |
| YOLOE-26m-pf | 25.7 / 26.4         | 26.7 / 24.5     | 24.0 / 25.0     | 26.9 / 27.9     | 19.4                     | 68.4                    |
| YOLOE-26l-pf | 27.2 / 28.0         | 26.3 / 25.7     | 25.7 / 26.8     | 28.7 / 29.5     | 23.6                     | 86.8                    |
| YOLOE-26x-pf | 29.9 / 31.1         | 27.5 / 28.9     | 29.1 / 30.7     | 31.1 / 31.7     | 53.1                     | 194.4                   |

Under text and visual prompts the YOLOE-26 models lead their YOLOE-11 and YOLOE-v8 counterparts at every matching scale on mAP<sub>50-95</sub>, while staying below the v8 line on parameters and FLOPs. On the same split the paper reports YOLO-Worldv2 at 24.4 (S), 32.4 (M) and 35.5 (L), and the transformer-based detectors GLIP-T at 26.0, GDINO-T at 27.4 and DetCLIP-T at 34.4, each carrying 155 to 232 M parameters. The original YOLOE paper adds two results for the v8-scale models it introduced. On LVIS, YOLOE-v8s beats YOLO-Worldv2-S by **3.5 AP** at a third of the training cost and 1.4× the inference speed. Transferred to COCO, YOLOE-v8l gains **0.6 box AP** and **0.4 mask AP** over closed-set YOLOv8-L with nearly **4× less training time**.

!!! note "Paper Counts vs Released Checkpoints"

    The YOLO26 paper evaluates a detection configuration. The released weights are **segmentation** checkpoints and carry a mask branch, SAVPE and the text projection on top, so a loaded `yoloe-26l-seg.pt` reports 35.4 M and 142.0 B rather than the 25.5 M and 89.0 B above. In both cases the FLOPs figure excludes the region-text similarity, so real cost grows with the size of the prompt set even though the column does not move; see [Limitations](#limitations).

## Usage Examples

Every YOLOE example below runs from the [Python API](../usage/python.md). Text-prompt prediction, validation, export, tracking and plain training also work from the [CLI](../usage/cli.md); the fine-tuning recipes and visual prompting pass a trainer or predictor class as an argument, which only the Python API accepts.

### Train Usage

Fine-tune any released `*-seg.pt` checkpoint on your own YOLO dataset. This mostly follows the [standard YOLO training procedure](../modes/train.md); the difference is which trainer you pass. `YOLOEPESegTrainer` fuses your class names into the head and fine-tunes from there, which is what you want on your own labels; the default trainer does not train against your class names.

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

        ```python
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

        model = YOLOE("yoloe-26s-seg.pt")

        results = model.train(
            data="coco128-seg.yaml",
            epochs=80,
            patience=10,
            trainer=YOLOEPESegTrainer,  # <- Important: the fine-tuning trainer, not the default
        )
        ```

    === "Linear Probing"

        Linear probing trains only the classification branch and freezes everything else. It is the right choice on a small dataset, where a full fine-tune overfits: the learned features are reused as they are and only the class decision adapts.

        ```python
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

        model = YOLOE("yoloe-26s-seg.pt")

        # Freeze the backbone and neck, i.e. everything before the head
        head_index = len(model.model.model) - 1
        freeze = [str(i) for i in range(head_index)]

        # Freeze the whole head except the terminal conv of every classification tower
        for name, _ in model.model.model[-1].named_children():
            if "cv3" in name:  # cv3, plus one2one_cv3 on end-to-end YOLOE-26 configs
                freeze.extend(f"{head_index}.{name}.{i}.{j}" for i in range(3) for j in (0, 1))
            else:
                freeze.append(f"{head_index}.{name}")

        results = model.train(
            data="coco128-seg.yaml",
            epochs=80,
            patience=10,
            trainer=YOLOEPESegTrainer,
            freeze=freeze,
        )
        ```

!!! tip "Training a detection model instead"

    Every released checkpoint is a segmentation model. To train a detector, build the model from the matching YAML, load the segmentation weights of the same scale, and swap in the detection trainer. Everything else is unchanged.

    ```python
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPETrainer

    model = YOLOE("yoloe-26s.yaml").load("yoloe-26s-seg.pt")

    results = model.train(data="coco128.yaml", epochs=80, patience=10, trainer=YOLOEPETrainer)
    ```

### Predict Usage

The text-prompt call is the one shown in [Quick Start](#quick-start). The remaining two modes each need an extra argument:

!!! example

    === "Visual Prompt"

        Visual prompts show the model an example instead of describing one. `visual_prompts` takes a `bboxes` array of example boxes and a `cls` array of class IDs, one per box. The IDs are temporary groupings, not labels — they must be sequential from 0, and the results come back as `object0`, `object1`, … rather than under names you choose.

        The example boxes can sit on the image you are predicting on:

        ```python
        import numpy as np

        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        model = YOLOE("yoloe-26l-seg.pt")

        # One example box per target, each with its own class ID
        visual_prompts = {
            "bboxes": np.array([[221.52, 405.8, 344.98, 857.54], [120, 425, 160, 445]]),  # person, glasses
            "cls": np.array([0, 1]),
        }

        results = model.predict(
            "ultralytics/assets/bus.jpg",
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
        )
        results[0].show()
        ```

        Or on a separate reference image passed as `refer_image`, in which case `bboxes` and `cls` describe objects in that reference, not in the target:

        ```python
        import numpy as np

        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        model = YOLOE("yoloe-26l-seg.pt")

        visual_prompts = {"bboxes": np.array([[221.52, 405.8, 344.98, 857.54]]), "cls": np.array([0])}  # person

        results = model.predict(
            "ultralytics/assets/zidane.jpg",  # Target image
            refer_image="ultralytics/assets/bus.jpg",  # Where the example boxes live
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
        )
        results[0].show()

        # refer_image also sets the classes permanently, so later calls need no prompts at all
        results = model("ultralytics/assets/bus.jpg")
        model.export(format="onnx")  # And the export keeps them
        ```

        !!! note

            When `source` is a video or stream, the first frame becomes the `refer_image` automatically, so the prompts you pass are applied to that frame and carried through the rest of the video. Pass `refer_image` explicitly to choose a different frame.

        Both `source` and `refer_image` accept `torch` tensors directly, which is useful when the images already come from an existing pipeline. Give the boxes in the tensor's own pixel coordinates:

        ```python
        import numpy as np
        import torch

        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        model = YOLOE("yoloe-11l-seg.pt")

        img_tensor = torch.rand(1, 3, 480, 480)  # (1, 3, H, W) float tensor in [0, 1]
        visual_prompts = {"bboxes": np.array([[10, 10, 50, 50]]), "cls": np.array([0])}

        results = model.predict(
            img_tensor,
            refer_image=img_tensor,
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
            imgsz=640,
        )
        ```

        To predict on several images at once, nest the prompts one level deeper: one `bboxes` array and one `cls` array **per source image**, in the same order as the sources.

        ```python
        import numpy as np

        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        model = YOLOE("yoloe-26l-seg.pt")

        visual_prompts = {
            "bboxes": [
                np.array([[221.52, 405.8, 344.98, 857.54], [120, 425, 160, 445]]),  # bus.jpg: person, glasses
                np.array([[150, 200, 1150, 700]]),  # zidane.jpg: person
            ],
            "cls": [np.array([0, 1]), np.array([0])],
        }

        results = model.predict(
            ["ultralytics/assets/bus.jpg", "ultralytics/assets/zidane.jpg"],
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
        )
        results[0].show()
        ```

    === "Prompt-Free"

        Prompt-free checkpoints carry their own vocabulary and behave like an ordinary YOLO model: no prompts, no text encoder, no `set_classes()`. They report objects from a [built-in 4,585-name vocabulary](https://github.com/xinyu1205/recognize-anything/blob/main/ram/data/ram_tag_list.txt), the tag set used by [Recognize Anything Plus (RAM++)](https://arxiv.org/abs/2310.15200).

        ```python
        from ultralytics import YOLOE

        model = YOLOE("yoloe-26l-seg-pf.pt")

        results = model.predict("path/to/image.jpg")
        results[0].show()
        ```

### Val Usage

Validation runs like any other model on a segmentation dataset:

!!! example

    ```python
    from ultralytics import YOLOE

    model = YOLOE("yoloe-26l-seg.pt")  # or yoloe-26s/m-seg.pt for other sizes

    metrics = model.val(data="coco128-seg.yaml")
    ```

Two variants of the same call cover the other prompting modes:

- **Visual prompts** — `model.val(data="coco128-seg.yaml", load_vp=True)` extracts a visual embedding per category from the dataset itself. Add `refer_data="coco.yaml"` to take the embeddings from a different dataset, which must carry exactly the same categories.
- **Prompt-free** — load a `*-seg-pf.pt` checkpoint and pass `single_cls=True`.

### Export Usage

Prompt embeddings can be saved once and reused when producing static exports such as ONNX, OpenVINO, TensorRT, CoreML, LiteRT, and RKNN. The NPZ profile is loaded by the original PyTorch model before export; it is not an additional runtime input, and the exported model does not require the NPZ file.

!!! warning "Exported models are static"

    Classes configured with `set_classes()` (or via `refer_image` for visual prompts) are baked into the exported weights. Once exported, the model can no longer accept new prompts: calling `set_classes()` or passing `visual_prompts=...` to `predict()` on a loaded export will fail. To change the detected classes, re-export from the original `.pt` checkpoint with the new prompts configured. The exported file behaves like a standard YOLO model and can also be loaded with `YOLO()` instead of `YOLOE()`.

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

### Track Usage

Prompted classes carry straight into [tracking](../modes/track.md), so you can follow objects the tracker was never trained on:

!!! example

    ```python
    from ultralytics import YOLOE

    model = YOLOE("yoloe-26s-seg.pt")
    model.set_classes(["forklift", "pallet"])

    # persist=True keeps track IDs stable across frames
    for result in model.track("path/to/video.mp4", stream=True, persist=True):
        print(result.boxes.id)
    ```

## How YOLOE Compares

YOLOE sits between a closed-set detector and a heavyweight open-vocabulary model. Three comparisons decide whether it is the right choice:

- **Against a closed-set YOLO.** Once the prompts are set, YOLOE predicts through the ordinary detect/segment path and exports like any other model. What it adds is the ability to change the class list at inference time instead of retraining; what it costs is zero-shot accuracy well below a model trained on your own classes.
- **Against the earlier YOLOE families.** YOLOE-26 inherits [YOLO26](yolo26.md)'s NMS-free end-to-end head and covers five scales (n/s/m/l/x) against the earlier three (s/m/l), and leads at every matching scale in [Performance](#yoloe-performance-on-lvis).
- **Against transformer-based open-vocabulary detectors.** GLIP and OWL-ViT run a vision-language transformer at inference. YOLOE encodes the prompts once and then compares them against region features inside a convolutional head.

The nearest alternatives all take a text prompt, but only YOLOE and SAM 3 return masks, and the three answer different questions:

|                  | [YOLOE](#quick-start)                                 | [SAM 3](sam-3.md)                                                     | [YOLO-World](yolo-world.md)                          |
| ---------------- | ----------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------- |
| Built for        | Real-time detection and segmentation of named classes | Concept segmentation and promptable tracking                          | Real-time open-vocabulary detection                  |
| Masks            | Yes, with the `*-seg.pt` checkpoints                  | Yes                                                                   | No, boxes only                                       |
| Visual prompts   | Yes (SAVPE)                                           | Yes                                                                   | No                                                   |
| Prompt-free mode | Yes, 4,585-name vocabulary                            | No                                                                    | No                                                   |
| Pick it when     | You need throughput and can name the classes          | You need the strongest concept segmentation and can spend the compute | You are already on it — see the migration note below |

**Coming from YOLO-World?** The API is the same shape: swap `YOLOWorld` for `YOLOE`, load a `*-seg.pt` checkpoint, and keep your `set_classes()` call as it is. You gain masks and visual prompts; the [export note](#export-usage) about frozen classes applies to both.

## Use Cases and Applications

Open-vocabulary detection removes the retrain-per-class step, which matters most where the target list is not known up front:

- **Open-world detection** — [robotics](https://www.ultralytics.com/blog/understanding-the-integration-of-computer-vision-in-robotics) and [security systems](https://www.ultralytics.com/blog/computer-vision-for-theft-prevention-enhancing-security) that meet objects nobody enumerated at training time.
- **One-shot detection from an example** — visual prompts pick up a specific part, logo or defect from a single reference box, useful in [industrial inspection](https://www.ultralytics.com/blog/computer-vision-in-manufacturing-improving-production-and-quality).
- **Long-tail cataloging** — the built-in 4,585-name vocabulary is broad enough for [biodiversity monitoring](https://www.ultralytics.com/blog/ai-in-wildlife-conservation) or [retail inventory](https://www.ultralytics.com/blog/ai-for-smarter-retail-inventory-management) sweeps.
- **Dataset bootstrapping** — pre-label images with boxes and masks before human review, then train a fast closed-set model on the result.
- **Segmentation of arbitrary targets** — the released `*-seg.pt` checkpoints return a mask with every prediction, so [medical imaging](https://www.ultralytics.com/blog/ai-and-radiology-a-new-era-of-precision-and-efficiency) and [satellite analysis](https://www.ultralytics.com/blog/using-computer-vision-to-analyze-satellite-imagery) get pixel-precise output without a second model.

A common pattern combines two modes: run prompt-free once to discover what is present, then switch to text prompts for the categories that matter.

## Limitations

YOLOE trades accuracy for the ability to change classes at inference time. The consequences worth knowing before you commit:

- **Zero-shot accuracy is well below a model trained on your classes.** The prompted checkpoints land roughly in the 22-40 mAP band on LVIS minival; a closed-set YOLO trained on your own data will beat that on those classes. Reach for YOLOE to cover classes you cannot train for, not to replace training.
- **Rare categories are the weak spot.** The mAP<sub>r</sub> column in [Performance](#yoloe-performance-on-lvis) reports accuracy on LVIS's rare classes specifically, and under text prompting it sits below the common and frequent columns in every row. Check it rather than the headline mAP when your targets are unusual.
- **A prompt describes appearance, not relationships.** Detection works by comparing region features against the prompt embedding, so prompts that depend on state, context or comparison — "damaged", "left-most", "the one being carried" — have no reliable handle to match on. Prefer wording close to everyday category names.
- **Large prompt sets cost latency.** The prompt embeddings are computed once, but they are compared against region features on every forward pass. Measured on CPU with `yoloe-26s-seg.pt`, a forward pass grows about 19% going from 80 to 1,203 classes and about 89% at the full 4,585-name vocabulary. Reported FLOPs do not move at all, because the region-text similarity is not counted, so the profile will not warn you.
- **Class names are placeholders until you prompt.** A freshly loaded `*-seg.pt` checkpoint reports `nc=80` with numeric names (`"0"`, `"1"`, …), so call `set_classes()` before reading labels. Prompt-free checkpoints ship the full vocabulary already populated.

## Deployment Notes

- **Hardware.** Inference needs an NVIDIA GPU with 4-8 GB of VRAM; the `n` and `s` scales run on edge GPUs such as [Jetson](../guides/nvidia-jetson.md) or on CPU at reduced resolution. Fine-tuning needs a single GPU.
- **NMS is class-agnostic by default.** YOLOE predicts with `agnostic_nms=True`. By default this suppresses lower-scoring overlapping boxes across different classes rather than only within the same class, which prevents duplicates when one object matches several categories. With `nms=False`, YOLOE-26 applies no IoU suppression; agnostic mode only keeps the single best class per anchor instead of letting one anchor emit several class labels. Pass `agnostic_nms=False` to override.
- **Batching.** [Batch inference](../modes/predict.md) works directly, and visual prompts can differ per image in the same call.

## Training the Official Models from Scratch

Most readers never need this. It reproduces the published open-vocabulary checkpoints from Objects365, GQA and Flickr30k — about 1.4 M training samples on 8× RTX 4090 — and is unrelated to fine-tuning on your own data, which is covered under [Train Usage](#train-usage) above.

!!! warning

    Every trainer that inherits `YOLOETrainer` refuses `compile=True`, including the default `YOLOESegTrainer` and all the from-scratch trainers below. Pass `compile=False` (the default). The two fine-tuning trainers used above, `YOLOEPESegTrainer` and `YOLOEPETrainer`, do not carry that restriction.

Training needs segment annotations. Either download the processed files below, or generate your own with [the script provided by the official team](https://github.com/THU-MIG/yoloe/blob/main/tools/generate_sam_masks.py), which is powered by [SAM 2.1](sam-2.md). Validation uses [LVIS minival](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml).

| Dataset                                                           | Type                                                        | Samples | Boxes | Processed segment annotations                                                                                                                   |
| ----------------------------------------------------------------- | ----------------------------------------------------------- | ------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| [Objects365v1](https://opendatalab.com/OpenDataLab/Objects365_v1) | Detection                                                   | 609k    | 9621k | [objects365_train_segm.json](https://huggingface.co/datasets/jameslahm/yoloe/resolve/main/objects365_train_segm.json)                           |
| [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html)      | [Grounding](https://www.ultralytics.com/glossary/grounding) | 621k    | 3681k | [final_mixed_train_no_coco_segm.json](https://huggingface.co/datasets/jameslahm/yoloe/resolve/main/final_mixed_train_no_coco_segm.json)         |
| [Flickr30k](https://github.com/BryanPlummer/flickr30k_entities)   | Grounding                                                   | 149k    | 641k  | [final_flickr_separateGT_train_segm.json](https://huggingface.co/datasets/jameslahm/yoloe/resolve/main/final_flickr_separateGT_train_segm.json) |

The text-prompt model is trained first, and the other two prompting modes are refinements of it:

```python
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOESegTrainerFromScratch

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

model = YOLOE("yoloe-26l-seg.yaml")
model.train(
    data=data,  # or the path to a YAML file holding the same structure
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

The visual-prompt and prompt-free checkpoints start from that trained text-prompt model and update one module each. `YOLOESegVPTrainer` is the visual-prompt recipe and `YOLOEPEFreeTrainer` the prompt-free one, but neither class freezes anything by itself: the selective training comes from the `freeze` list you pass alongside it, which names every head child except `savpe` (resp. every classification tower), and the prompt-free run additionally needs `single_cls=True`. The [upstream YOLOE repository](https://github.com/THU-MIG/yoloe) carries the full recipes for the v8-scale models.

A finished prompt-free run is re-parameterized into a checkpoint that reports its names with no prompt at inference, using [`get_vocab` and `set_vocab`](../reference/models/yolo/model.md):

```python
from ultralytics import YOLOE

# Weights written by the prompt-free run and by the text-prompt run it started from. Each
# rerun creates a new directory (train-2, train-3, ...), so take the paths the runs printed.
model = YOLOE("runs/segment/train-2/weights/best.pt")  # prompt-free run, its head is already fused
text_model = YOLOE("runs/segment/train/weights/best.pt")  # text-prompt run, its head is still unfused

names = list(YOLOE("yoloe-26l-seg-pf.pt").model.names.values())  # the 4,585-name vocabulary, or your own list
vocab = text_model.get_vocab(names)

model.set_vocab(vocab, names)
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

Ultralytics YOLOE adds two capabilities YOLO-World does not have: **visual prompts**, where an example box replaces the class name, and **prompt-free** checkpoints that answer from a built-in 4,585-name vocabulary with no prompt at all. Every prediction from the released `*-seg.pt` checkpoints also carries an [instance segmentation](https://www.ultralytics.com/blog/what-is-instance-segmentation-a-quick-guide) mask. On accuracy, the original YOLOE paper puts YOLOE-v8s ahead of YOLO-Worldv2-S by 3.5 AP on LVIS, at a third of the training cost and 1.4× the inference speed. Migrating is a one-line change — see [How YOLOE Compares](#how-yoloe-compares).

### What types of prompts can I use with YOLOE?

Ultralytics YOLOE supports three prompting modes. A **text prompt** is class names as strings on a `*-seg.pt` checkpoint, and is the usual choice. A **visual prompt** is one or more example boxes on a reference image, for targets that are hard to put into words. **Prompt-free** inference uses a separate `*-seg-pf.pt` checkpoint that answers from a built-in 4,585-name vocabulary with nothing supplied. Text and visual prompts share the same checkpoints; prompt-free ones are different files and reject `set_classes()`. See [Choosing a Prompting Mode](#choosing-a-prompting-mode) for the full comparison.

### Which YOLOE model should I use?

Start with `yoloe-26s-seg.pt`: the YOLOE-26 family leads YOLOE-11 and YOLOE-v8 at every matching scale, and the `s` scale is the smallest one above 30 mAP on LVIS minival. Move up to `m`, `l` or `x` when accuracy on rare categories matters more than latency — the mAP<sub>r</sub> column in [Performance](#yoloe-performance-on-lvis) is the one to compare. Drop to `n` only for edge deployment. Load the `*-seg-pf.pt` file of the same scale instead when you want the built-in vocabulary rather than your own class names.

### Why are my detections labeled object0 and object1 instead of my class names?

Labels like `object0` and `object1` mean the prediction came from a visual prompt, which groups the example boxes into temporary numbered classes instead of carrying your names. The class IDs you pass in `visual_prompts["cls"]` only do that grouping. The model reports them as `object0`, `object1`, and so on, in the order of the IDs you assigned, so map them back to your own labels on the result. If you want your names in the output, use a text prompt instead.

### Why does set_classes() fail on a prompt-free checkpoint?

Prompt-free checkpoints (`*-seg-pf.pt`) resolve classes through their own built-in vocabulary and reject external prompts with `AssertionError: Prompt-free model does not support setting classes. Please try with Text/Visual prompt models.` Load a `*-seg.pt` checkpoint when you need your own class list. See [Choosing a Prompting Mode](#choosing-a-prompting-mode).

### What does YOLOE download the first time I run a text prompt?

The first text prompt makes Ultralytics YOLOE install [ultralytics/CLIP](https://github.com/ultralytics/CLIP) from GitHub with `pip` and download a TorchScript text encoder into the current working directory — about 254 MB for YOLOE-26; see [Installation and Requirements](#installation-and-requirements) for the exact asset per model family. Visual prompts and prompt-free checkpoints need neither. To avoid the download on the target machine, set the prompts once and save them with `save_prompt_embeddings()`, or export the model with the classes already configured.

### Can I change the classes of an exported YOLOE model?

No — YOLOE bakes the prompted classes into the weights at export time, so a loaded export rejects both `set_classes()` and `visual_prompts=`. Re-export from the original `.pt` checkpoint with the new prompts configured. The exported file behaves like a standard YOLO model and can be loaded with `YOLO()` as well as `YOLOE()`.

### Should I use YOLOE or SAM 3?

Use YOLOE when you need real-time throughput and can name the classes, and [SAM 3](sam-3.md) when segmentation quality on a concept matters more than speed. Both accept visual examples; only YOLOE has a prompt-free mode. The full comparison is in [How YOLOE Compares](#how-yoloe-compares).
