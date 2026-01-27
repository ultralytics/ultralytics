---
comments: true
description: YOLOE is a real-time open-vocabulary detection and segmentation model that extends YOLO with text, image, or internal vocabulary prompts, enabling detection of any object class with state-of-the-art zero-shot performance.
keywords: YOLOE, open-vocabulary detection, real-time object detection, instance segmentation, YOLO, text prompts, visual prompts, zero-shot detection
---

# YOLOE: Real-Time Seeing Anything

## Introduction

![YOLOE Prompting Options](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/yoloe-visualization.avif)

[YOLOE (Real-Time Seeing Anything)](https://arxiv.org/html/2503.07465v1) is a new advancement in zero-shot, promptable YOLO models, designed for **open-vocabulary** detection and segmentation. Unlike previous YOLO models limited to fixed categories, YOLOE uses text, image, or internal vocabulary prompts, enabling real-time detection of any object class. Built upon YOLOv10 and inspired by [YOLO-World](yolo-world.md), YOLOE achieves **state-of-the-art zero-shot performance** with minimal impact on speed and accuracy.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/HMOoM2NwFIQ"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to use YOLOE with Ultralytics Python package: Open Vocabulary & Real-Time Seeing Anything 🚀
</p>

Compared to earlier YOLO models, YOLOE significantly boosts efficiency and accuracy. It improves by **+3.5 AP** over YOLO-Worldv2 on LVIS while using just a third of the training resources and achieving 1.4× faster inference speeds. Fine-tuned on COCO, YOLOE-v8-large surpasses YOLOv8-L by **0.1 mAP**, using nearly **4× less training time**. This demonstrates YOLOE's exceptional balance of accuracy, efficiency, and versatility. The sections below explore YOLOE's architecture, benchmark comparisons, and integration with the [Ultralytics](https://www.ultralytics.com/) framework.

## Architecture Overview

<p align="center">
  <img src="https://github.com/THU-MIG/yoloe/raw/main/figures/pipeline.svg" alt="YOLOE Architecture" width=90%>
</p>

YOLOE retains the standard YOLO structure—a convolutional **backbone** (e.g., CSP-Darknet) for feature extraction, a **neck** (e.g., PAN-FPN) for multi-scale fusion, and an **anchor-free, decoupled** detection **head** (as in YOLOv8/YOLO11) predicting objectness, classes, and boxes independently. YOLOE introduces three novel modules enabling open-vocabulary detection:

- **Re-parameterizable Region-Text Alignment (RepRTA)**: Supports **text-prompted detection** by refining text [embeddings](https://www.ultralytics.com/glossary/embeddings) (e.g., from CLIP) via a small auxiliary network. At inference, this network is folded into the main model, ensuring zero overhead. YOLOE thus detects arbitrary text-labeled objects (e.g., unseen "traffic light") without runtime penalties.

- **Semantic-Activated Visual Prompt Encoder (SAVPE)**: Enables **visual-prompted detection** via a lightweight embedding branch. Given a reference image, SAVPE encodes semantic and activation features, conditioning the model to detect visually similar objects—a one-shot detection capability useful for logos or specific parts.

- **Lazy Region-Prompt Contrast (LRPC)**: In **prompt-free mode**, YOLOE performs open-set recognition using internal embeddings trained on large vocabularies (1200+ categories from LVIS and Objects365). Without external prompts or encoders, YOLOE identifies objects via embedding similarity lookup, efficiently handling large label spaces at inference.

Additionally, YOLOE integrates real-time **instance segmentation** by extending the detection head with a mask prediction branch (similar to YOLACT or YOLOv8-Seg), adding minimal overhead.

Crucially, YOLOE's open-world modules introduce **no inference cost** when used as a regular closed-set YOLO. Post-training, YOLOE parameters can be re-parameterized into a standard YOLO head, preserving identical FLOPs and speed (e.g., matching [YOLO11](yolo11.md) exactly).

## Available Models, Supported Tasks, and Operating Modes

This section details the models available with their specific pretrained weights, the tasks they support, and their compatibility with various operating modes such as [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), and [Export](../modes/export.md), denoted by ✅ for supported modes and ❌ for unsupported modes.

### Text/Visual Prompt models

| Model Type | Pretrained Weights                                                                                  | Tasks Supported                              | Inference | Validation | Training | Export |
| ---------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------- | --------- | ---------- | -------- | ------ |
| YOLOE-11S  | [yoloe-11s-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11s-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-11M  | [yoloe-11m-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11m-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-11L  | [yoloe-11l-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11l-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-v8S  | [yoloe-v8s-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-v8s-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-v8M  | [yoloe-v8m-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-v8m-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-v8L  | [yoloe-v8l-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-v8l-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-26N  | [yoloe-26n-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26n-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-26S  | [yoloe-26s-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-26M  | [yoloe-26m-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26m-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-26L  | [yoloe-26l-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-26X  | [yoloe-26x-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |

### Prompt Free models

| Model Type   | Pretrained Weights                                                                                        | Tasks Supported                              | Inference | Validation | Training | Export |
| ------------ | --------------------------------------------------------------------------------------------------------- | -------------------------------------------- | --------- | ---------- | -------- | ------ |
| YOLOE-11S-PF | [yoloe-11s-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11s-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-11M-PF | [yoloe-11m-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11m-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-11L-PF | [yoloe-11l-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11l-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-v8S-PF | [yoloe-v8s-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-v8s-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-v8M-PF | [yoloe-v8m-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-v8m-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-v8L-PF | [yoloe-v8l-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-v8l-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-26N-PF | [yoloe-26n-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26n-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-26S-PF | [yoloe-26s-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-26M-PF | [yoloe-26m-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26m-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-26L-PF | [yoloe-26l-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOE-26X-PF | [yoloe-26x-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg-pf.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ✅         | ✅       | ✅     |

!!! tip "YOLOE-26 Performance"

    For detailed performance benchmarks of YOLOE-26 models, see the [YOLO26 Documentation](yolo26.md#yoloe-26-open-vocabulary-instance-segmentation).

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
        freeze = [str(i) for i in range(0, head_index)]

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
        freeze = [str(i) for i in range(0, head_index)]

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
        names = ["person", "bus"]
        model.set_classes(names, model.get_text_pe(names))

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
        visual_prompts = dict(
            bboxes=np.array(
                [
                    [221.52, 405.8, 344.98, 857.54],  # Box enclosing person
                    [120, 425, 160, 445],  # Box enclosing glasses
                ],
            ),
            cls=np.array(
                [
                    0,  # ID to be assigned for person
                    1,  # ID to be assigned for glasses
                ]
            ),
        )

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
        visual_prompts = dict(
            bboxes=np.array([[221.52, 405.8, 344.98, 857.54]]),  # Box enclosing person
            cls=np.array([0]),  # ID to be assigned for person
        )

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

        You can also pass multiple target images to run prediction on:

        ```python
        import numpy as np

        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        # Initialize a YOLOE model
        model = YOLOE("yoloe-26l-seg.pt")

        # Define visual prompts using bounding boxes and their corresponding class IDs.
        # Each box highlights an example of the object you want the model to detect.
        visual_prompts = dict(
            bboxes=[
                np.array(
                    [
                        [221.52, 405.8, 344.98, 857.54],  # Box enclosing person
                        [120, 425, 160, 445],  # Box enclosing glasses
                    ],
                ),
                np.array([[150, 200, 1150, 700]]),
            ],
            cls=[
                np.array(
                    [
                        0,  # ID to be assigned for person
                        1,  # ID to be assigned for glasses
                    ]
                ),
                np.array([0]),
            ],
        )

        # Run inference on multiple image, using the provided visual prompts as guidance
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

!!! example

    ```python
    from ultralytics import YOLOE

    # Select yoloe-26s/m-seg.pt for different sizes
    model = YOLOE("yoloe-26l-seg.pt")

    # Configure the set_classes() before exporting the model
    names = ["person", "bus"]
    model.set_classes(names, model.get_text_pe(names))

    export_model = model.export(format="onnx")
    model = YOLOE(export_model)

    # Run detection on the given image
    results = model.predict("path/to/image.jpg")

    # Show results
    results[0].show()
    ```

### Train Official Models

#### Prepare datasets

!!! note

    Training official YOLOE models needs segment annotations for train data, here's [the script provided by official team](https://github.com/THU-MIG/yoloe/blob/main/tools/generate_sam_masks.py) that converts datasets to segment annotations, powered by [SAM2.1 models](./sam-2.md). Or you can directly download the provided `Processed Segment Annotations` in following table provided by official team.

- Train data

| Dataset                                                           | Type                                                        | Samples | Boxes | Raw Detection Annotations                                                                                                                  | Processed Segment Annotations                                                                                                                |
| ----------------------------------------------------------------- | ----------------------------------------------------------- | ------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| [Objects365v1](https://opendatalab.com/OpenDataLab/Objects365_v1) | Detection                                                   | 609k    | 9621k | [objects365_train.json](https://opendatalab.com/OpenDataLab/Objects365_v1)                                                                 | [objects365_train_segm.json](https://huggingface.co/datasets/jameslahm/yoloe/blob/main/objects365_train_segm.json)                           |
| [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html)      | [Grounding](https://www.ultralytics.com/glossary/grounding) | 621k    | 3681k | [final_mixed_train_no_coco.json](https://huggingface.co/GLIPModel/GLIP/blob/main/mdetr_annotations/final_mixed_train_no_coco.json)         | [final_mixed_train_no_coco_segm.json](https://huggingface.co/datasets/jameslahm/yoloe/blob/main/final_mixed_train_no_coco_segm.json)         |
| [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)     | Grounding                                                   | 149k    | 641k  | [final_flickr_separateGT_train.json](https://huggingface.co/GLIPModel/GLIP/blob/main/mdetr_annotations/final_flickr_separateGT_train.json) | [final_flickr_separateGT_train_segm.json](https://huggingface.co/datasets/jameslahm/yoloe/blob/main/final_flickr_separateGT_train_segm.json) |

- Val data

| Dataset                                                                                                 | Type      | Annotation Files                                                                                       |
| ------------------------------------------------------------------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------ |
| [LVIS minival](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml) | Detection | [minival.txt](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml) |

#### Launching training from scratch

!!! note

    `Visual Prompt` models are fine-tuned based on trained-well `Text Prompt` models.

!!! example

    === "Text Prompt"

        ```python
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOESegTrainerFromScratch

        # Option 1: Use Python dictionary
        data = dict(
            train=dict(
                yolo_data=["Objects365.yaml"],
                grounding_data=[
                    dict(
                        img_path="flickr/full_images/",
                        json_file="flickr/annotations/final_flickr_separateGT_train_segm.json",
                    ),
                    dict(
                        img_path="mixed_grounding/gqa/images",
                        json_file="mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
                    ),
                ],
            ),
            val=dict(yolo_data=["lvis.yaml"]),
        )

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

        Since only the `SAVPE` module needs to be updating during training.
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

        data = dict(
            train=dict(
                yolo_data=["Objects365.yaml"],
                grounding_data=[
                    dict(
                        img_path="flickr/full_images/",
                        json_file="flickr/annotations/final_flickr_separateGT_train_segm.json",
                    ),
                    dict(
                        img_path="mixed_grounding/gqa/images",
                        json_file="mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
                    ),
                ],
            ),
            val=dict(yolo_data=["lvis.yaml"]),
        )

        model = YOLOE("yoloe-26l-seg.pt")
        # replace to yoloe-26l-seg-det.pt if converted to detection model
        # model = YOLOE("yoloe-26l-seg-det.pt")

        # freeze every layer except of the savpe module.
        head_index = len(model.model.model) - 1
        freeze = list(range(0, head_index))
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

        vp_model = YOLOE("yoloe-11l-vp.pt")
        model.model.model[-1].savpe = deepcopy(vp_model.model.model[-1].savpe)
        model.eval()
        model.save("yoloe-26l-seg.pt")
        ```

    === "Prompt Free"

        Similar to visual prompt training, for prompt-free model there's only the specialized prompt embedding needs to be updating during training.
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

        data = dict(
            train=dict(
                yolo_data=["Objects365.yaml"],
                grounding_data=[
                    dict(
                        img_path="flickr/full_images/",
                        json_file="flickr/annotations/final_flickr_separateGT_train_segm.json",
                    ),
                    dict(
                        img_path="mixed_grounding/gqa/images",
                        json_file="mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
                    ),
                ],
            ),
            val=dict(yolo_data=["lvis.yaml"]),
        )

        model = YOLOE("yoloe-26l-seg.pt")
        # replace to yoloe-26l-seg-det.pt if converted to detection model
        # model = YOLOE("yoloe-26l-seg-det.pt")

        # freeze layers.
        head_index = len(model.model.model) - 1
        freeze = [str(f) for f in range(0, head_index)]
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
        model.save("yoloe-26l-seg-pf.pt")
        ```

## YOLOE Performance Comparison

YOLOE matches or exceeds the accuracy of closed-set YOLO models on standard benchmarks like COCO and LVIS, without compromising speed or model size. The table below compares YOLOE-L (built on YOLO11) and YOLOE26-L (built on [YOLO26](yolo26.md)) against corresponding closed-set models:

| Model                      | COCO mAP<sub>50-95</sub> | LVIS mAP<sub>50-95</sub> | Inference Speed (T4)  | Parameters | GFLOPs (640px)     |
| -------------------------- | ------------------------ | ------------------------ | --------------------- | ---------- | ------------------ |
| **YOLOv8-L** (closed-set)  | 52.9%                    | -                        | **9.06 ms** (110 FPS) | 43.7 M     | 165.2 B            |
| **YOLO11-L** (closed-set)  | 53.5%                    | -                        | **6.2 ms** (161 FPS)  | 26.2 M     | 86.9 B             |
| **YOLOE-L** (open-vocab)   | 52.6%                    | 35.2%                    | **6.2 ms** (161 FPS)  | 26.2 M     | 86.9 B<sup>†</sup> |
| **YOLOE26-L** (open-vocab) | -                        | 36.8%                    | **6.2 ms** (161 FPS)  | 32.3 M     | 88.3 B<sup>†</sup> |

<sup>†</sup> _YOLOE-L shares YOLO11-L's architecture and YOLOE26-L shares YOLO26-L's architecture, resulting in similar inference speed and GFLOPs._

YOLOE26-L achieves **36.8% LVIS mAP** with **32.3M parameters** and **88.3B FLOPs**, processing 640×640 images at **6.2 ms (161 FPS)** on T4 GPU. This improves over YOLOE-L's **35.2% LVIS mAP** while maintaining the same inference speed. Crucially, YOLOE's open-vocabulary modules incur **no inference cost**, demonstrating a **"no free lunch trade-off"** design.

For zero-shot tasks, YOLOE26 significantly outperforms prior open-vocabulary detectors: on LVIS, YOLOE26-S achieves **29.9% mAP**, surpassing YOLO-World-S by **+11.4 AP**, while YOLOE26-L achieves **36.8% mAP**, exceeding YOLO-World-L by **+10.0 AP**. YOLOE26 maintains efficient inference at **161 FPS** on T4 GPU, ideal for real-time open-vocabulary applications.

!!! note

    **Benchmark conditions:** YOLOE results are from models pretrained on Objects365, GoldG, and LVIS, then fine-tuned or evaluated on COCO. YOLOE's slight mAP advantage over YOLOv8 comes from extensive pre-training. Without this open-vocab training, YOLOE matches similar-sized YOLO models, affirming its SOTA accuracy and open-world flexibility without performance penalties.

## Comparison with Previous Models

YOLOE introduces notable advancements over prior YOLO models and open-vocabulary detectors:

- **YOLOE vs YOLOv5:**
  [YOLOv5](yolov5.md) offered good speed-accuracy balance but required retraining for new classes and used anchor-based heads. In contrast, YOLOE is **anchor-free** and dynamically detects new classes. YOLOE, building on YOLOv8's improvements, achieves higher accuracy (52.6% vs. YOLOv5's ~50% mAP on COCO) and integrates instance segmentation, unlike YOLOv5.

- **YOLOE vs YOLOv8:**
  YOLOE extends [YOLOv8](yolov8.md)'s redesigned architecture, achieving similar or superior accuracy (**52.6% mAP with ~26M parameters** vs. YOLOv8-L's **52.9% with ~44M parameters**). It significantly reduces training time due to stronger pre-training. The key advancement is YOLOE's **open-world capability**, detecting unseen objects (e.g., "**bird scooter**" or "**peace symbol**") via prompts, unlike YOLOv8's closed-set design.

- **YOLOE vs YOLO11:**
  [YOLO11](yolo11.md) improves upon YOLOv8 with enhanced efficiency and fewer parameters (~22% reduction). YOLOE inherits these gains directly, matching YOLO11's inference speed and parameter count (~26M parameters), while adding **open-vocabulary detection and segmentation**. In closed-set scenarios, YOLOE is equivalent to YOLO11, but crucially adds adaptability to detect unseen classes, achieving **YOLO11 + open-world capability** without compromising speed.

- **YOLOE26 vs YOLOE (YOLO11-based):**
  YOLOE26 builds upon [YOLO26](yolo26.md)'s architecture, inheriting its NMS-free end-to-end design for faster inference. On LVIS, YOLOE26-L achieves **36.8% mAP**, improving over YOLOE-L's **35.2% mAP**. YOLOE26 offers all five model scales (N/S/M/L/X) compared to YOLOE's three (S/M/L), providing more flexibility for different deployment scenarios.

- **YOLOE26 vs previous open-vocabulary detectors:**
  Earlier open-vocab models (GLIP, OWL-ViT, [YOLO-World](yolo-world.md)) relied heavily on vision-language [transformers](https://www.ultralytics.com/glossary/transformer), leading to slow inference. On LVIS, YOLOE26-S achieves **29.9% mAP** (**+11.4 AP** over YOLO-World-S) and YOLOE26-L achieves **36.8% mAP** (**+10.0 AP** over YOLO-World-L), while maintaining real-time inference at **161 FPS** on T4 GPU. Compared to transformer-based approaches (e.g., GLIP), YOLOE26 offers orders-of-magnitude faster inference, effectively bridging the accuracy-efficiency gap in open-set detection.

In summary, YOLOE and YOLOE26 maintain YOLO's renowned speed and efficiency, surpass predecessors in accuracy, integrate segmentation, and introduce powerful open-world detection. YOLOE26 further advances the architecture with NMS-free end-to-end inference from YOLO26, making it ideal for real-time open-vocabulary applications.

## Use Cases and Applications

YOLOE's open-vocabulary detection and segmentation enable diverse applications beyond traditional fixed-class models:

- **Open-World Object Detection:**
  Ideal for dynamic scenarios like [robotics](https://www.ultralytics.com/blog/understanding-the-integration-of-computer-vision-in-robotics), where robots recognize previously unseen objects using prompts, or [security systems](https://www.ultralytics.com/blog/computer-vision-for-theft-prevention-enhancing-security) quickly adapting to new threats (e.g., hazardous items) without retraining.

- **Few-Shot and One-Shot Detection:**
  Using visual prompts (SAVPE), YOLOE rapidly learns new objects from single reference images—perfect for [industrial inspection](https://www.ultralytics.com/blog/computer-vision-in-manufacturing-improving-production-and-quality) (identifying parts or defects instantly) or **custom surveillance**, enabling visual searches with minimal setup.

- **Large-Vocabulary & Long-Tail Recognition:**
  Equipped with a vocabulary of 1000+ classes, YOLOE excels in tasks like [biodiversity monitoring](https://www.ultralytics.com/blog/ai-in-wildlife-conservation) (detecting rare species), **museum collections**, [retail inventory](https://www.ultralytics.com/blog/ai-for-smarter-retail-inventory-management), or **e-commerce**, reliably identifying many classes without extensive per-class training.

- **Interactive Detection and Segmentation:**
  YOLOE supports real-time interactive applications such as **searchable video/image retrieval**, **augmented reality (AR)**, and intuitive **image editing**, driven by natural inputs (text or visual prompts). Users can dynamically isolate, identify, or edit objects precisely using segmentation masks.

- **Automated Data Labeling and Bootstrapping:**
  YOLOE facilitates rapid dataset creation by providing initial bounding box and segmentation annotations, significantly reducing human labeling efforts. Particularly valuable in **analytics of large media collections**, where it can auto-identify objects present, assisting in building specialized models faster.

- **Segmentation for Any Object:**
  Extends segmentation capabilities to arbitrary objects through prompts—particularly beneficial for [medical imaging](https://www.ultralytics.com/blog/ai-and-radiology-a-new-era-of-precision-and-efficiency), **microscopy**, or [satellite imagery analysis](https://www.ultralytics.com/blog/using-computer-vision-to-analyze-satellite-imagery), automatically identifying and precisely segmenting structures without specialized pretrained models. Unlike models like [SAM](sam.md), YOLOE simultaneously recognizes and segments objects automatically, aiding in tasks like **content creation** or **scene understanding**.

Across all these use cases, YOLOE's core advantage is **versatility**, providing a unified model for detection, recognition, and segmentation across dynamic scenarios. Its efficiency ensures real-time performance on resource-constrained devices, ideal for robotics, [autonomous driving](https://www.ultralytics.com/blog/ai-in-self-driving-cars), defense, and beyond.

!!! tip

    Choose YOLOE's mode based on your needs:

    - **Closed-set mode:** For fixed-class tasks (max speed and accuracy).
    - **Prompted mode:** Add new objects quickly via text or visual prompts.
    - **Prompt-free open-set mode:** General detection across many categories (ideal for cataloging and discovery).

    Often, combining modes—such as prompt-free discovery followed by targeted prompts—leverages YOLOE's full potential.

## Training and Inference

YOLOE integrates seamlessly with the [Ultralytics Python API](../usage/python.md) and [CLI](../usage/cli.md), similar to other YOLO models (YOLOv8, YOLO-World). Here's how to quickly get started:

!!! example "Training and inference with YOLOE"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load pretrained YOLOE model and train on custom data
        model = YOLO("yoloe-26s-seg.pt")
        model.train(data="path/to/data.yaml", epochs=50, imgsz=640)

        # Run inference using text prompts ("person", "bus")
        model.set_classes(["person", "bus"])
        results = model.predict(source="test_images/street.jpg")
        results[0].save()  # save annotated output
        ```

        Here, YOLOE behaves like a standard detector by default but easily switches to prompted detection by specifying classes (`set_classes`). Results contain bounding boxes, masks, and labels.

    === "CLI"

        ```bash
        # Training YOLOE on custom dataset
        yolo train model=yoloe-26s-seg.pt data=path/to/data.yaml epochs=50 imgsz=640

        # Inference with text prompts
        yolo predict model=yoloe-26s-seg.pt source="test_images/street.jpg" classes="person,bus"
        ```

        CLI prompts (`classes`) guide YOLOE similarly to Python's `set_classes`. Visual prompting (image-based queries) currently requires the Python API.

### Other Supported Tasks

- **Validation:** Evaluate accuracy easily with `model.val()` or `yolo val`.
- **Export:** Export YOLOE models (`model.export()`) to ONNX, TensorRT, etc., facilitating deployment.
- **Tracking:** YOLOE supports object tracking (`yolo track`) when integrated, useful for tracking prompted classes in videos.

!!! note

    YOLOE automatically includes **segmentation masks** in inference results (`results[0].masks`), simplifying pixel-precise tasks like object extraction or measurement without needing separate models.

## Getting Started

Quickly set up YOLOE with Ultralytics by following these steps:

1. **Installation**:
   Install or update the Ultralytics package:

    ```bash
    pip install -U ultralytics
    ```

2. **Download YOLOE Weights**:
   Pretrained YOLOE models (e.g., YOLOE-v8-S/L, YOLOE-11 variants) are available from the YOLOE GitHub releases. Simply download your desired `.pt` file to load into the Ultralytics YOLO class.

3. **Hardware Requirements**:
    - **Inference**: Recommended GPU (NVIDIA with ≥4-8GB VRAM). Small models run efficiently on edge GPUs (e.g., [Jetson](../guides/nvidia-jetson.md)) or CPUs at lower resolutions. For high-performance inference on compact workstations, see our [NVIDIA DGX Spark](../guides/nvidia-dgx-spark.md) guide.
    - **Training**: Fine-tuning YOLOE on custom data typically requires just one GPU. Extensive open-vocabulary pre-training (LVIS/Objects365) used by authors required substantial compute (8× RTX 4090 GPUs).

4. **Configuration**:
   YOLOE configurations use standard Ultralytics YAML files. Default configs (e.g., `yoloe-26s-seg.yaml`) typically suffice, but you can modify backbone, classes, or image size as needed.

5. **Running YOLOE**:
    - **Quick inference** (prompt-free):
        ```bash
        yolo predict model=yoloe-26s-seg-pf.pt source="image.jpg"
        ```
    - **Prompted detection** (text prompt example):

        ```python
        from ultralytics import YOLO

        model = YOLO("yoloe-26s-seg.pt")
        names = ["bowl", "apple"]
        model.set_classes(names, model.get_text_pe(names))
        results = model.predict("kitchen.jpg")
        results[0].save()
        ```

6. **Integration Tips**:
    - **Class names**: Default YOLOE outputs use LVIS categories; use `set_classes()` to specify your own labels.
    - **Speed**: YOLOE has no overhead unless using prompts. Text prompts have minimal impact; visual prompts slightly more.
    - **Batch inference**: Supported directly (`model.predict([img1, img2])`). For image-specific prompts, run images individually.

The [Ultralytics documentation](https://docs.ultralytics.com/) provides further resources. YOLOE lets you easily explore powerful open-world capabilities within the familiar YOLO ecosystem.

!!! tip

    **Pro Tip:**
    To maximize YOLOE's zero-shot accuracy, fine-tune from provided checkpoints rather than training from scratch. Use prompt words aligning with common training labels (see LVIS categories) to improve detection accuracy.

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

While both YOLOE and [YOLO-World](yolo-world.md) enable open-vocabulary detection, YOLOE offers several advantages. YOLOE achieves +3.5 AP higher accuracy on LVIS while using 3× less training resources and running 1.4× faster than YOLO-Worldv2. YOLOE also supports three prompting modes (text, visual, and internal vocabulary), whereas YOLO-World primarily focuses on text prompts. Additionally, YOLOE includes built-in [instance segmentation](https://www.ultralytics.com/blog/what-is-instance-segmentation-a-quick-guide) capabilities, providing pixel-precise masks for detected objects without additional overhead.

### Can I use YOLOE as a regular YOLO model?

Yes, YOLOE can function exactly like a standard YOLO model with no performance penalty. When used in closed-set mode (without prompts), YOLOE's open-vocabulary modules are re-parameterized into the standard detection head, resulting in identical speed and accuracy to equivalent YOLO11 models. This makes YOLOE extremely versatile—you can use it as a traditional detector for maximum speed and then switch to open-vocabulary mode only when needed.

### What types of prompts can I use with YOLOE?

YOLOE supports three types of prompts:

1. **Text prompts**: Specify object classes using natural language (e.g., "person", "traffic light", "bird scooter")
2. **Visual prompts**: Provide reference images of objects you want to detect
3. **Internal vocabulary**: Use YOLOE's built-in vocabulary of 1200+ categories without external prompts

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
names = ["person", "bus"]
model.set_classes(names, model.get_text_pe(names))

# Execute prediction on an image
results = model.predict("path/to/image.jpg")

# Show results
results[0].show()
```
