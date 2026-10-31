---
comments: true
description: Meta's Segment Anything Model (SAM) performs promptable, zero-shot image segmentation. Learn its architecture, SA-1B dataset, auto-annotation and Ultralytics usage.
keywords: Segment Anything, SAM, SAM-1, image segmentation, promptable segmentation, zero-shot segmentation, instance segmentation, auto-annotation, SAM vs YOLO, SAM vs SAM 2, SA-1B dataset, Meta, Ultralytics
---

# Segment Anything Model (SAM)

!!! note "SAM Evolution"

    This is the original SAM model from Meta. For improved capabilities, see [SAM 2](sam-2.md) for video segmentation or [SAM 3](sam-3.md) for Promptable Concept Segmentation with text and image exemplar prompts.

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/inference-with-meta-sam-and-sam2-using-ultralytics-python-package.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="How to use Segment Anything In Colab"></a>

The Segment Anything Model (SAM) from Meta introduced promptable [image segmentation](https://www.ultralytics.com/glossary/image-segmentation): given a point, box, or mask prompt, it returns a mask for the indicated object, with a lightweight decoder fast enough for interactive use.

## Introduction to SAM: The Segment Anything Model

The Segment Anything Model, or SAM, is an image segmentation model built for promptable segmentation across a wide range of image analysis tasks. SAM is the core of Meta's Segment Anything project, which introduced a new model, task, and dataset for image segmentation.

SAM's design allows it to adapt to new image distributions and tasks without prior knowledge, a capability known as [zero-shot](https://www.ultralytics.com/glossary/zero-shot-learning) transfer. Trained on the [SA-1B dataset](https://arxiv.org/abs/2304.02643), which contains more than 1 billion masks across 11 million curated images, SAM shows strong zero-shot performance that is often competitive with or better than prior fully supervised results.

![SA-1B dataset sample with automatic segmentation masks](https://cdn.ul.run/i/620ba43b239b65fda908547da826ac16.avif) **SA-1B example images.** Images with overlaid masks from the SA-1B dataset. SA-1B contains 11M diverse, high-resolution, licensed, and privacy-protecting images and 1.1B high-quality segmentation masks. These masks were annotated fully automatically by SAM, and as verified by human ratings and numerous experiments, are of high quality and diversity. Images are grouped by number of masks per image for visualization (there are ∼100 masks per image on average).

## Key Features of the Segment Anything Model (SAM)

- **Promptable Segmentation Task:** SAM was designed with a promptable segmentation task in mind, allowing it to generate valid segmentation masks from any given prompt, such as spatial or text clues identifying an object.
- **Advanced Architecture:** The Segment Anything Model uses a heavyweight image encoder, a prompt encoder, and a lightweight mask decoder. This architecture enables flexible prompting, real-time mask computation, and ambiguity awareness in segmentation tasks.
- **The SA-1B Dataset:** Introduced by the Segment Anything project, the SA-1B dataset features over 1 billion masks on 11 million images. As the largest segmentation dataset at its release, it provides SAM with a diverse and large-scale training data source.
- **Zero-Shot Performance:** SAM displays strong zero-shot performance across various segmentation tasks, making it a ready-to-use tool for diverse applications with minimal need for [prompt engineering](https://www.ultralytics.com/glossary/prompt-engineering).

For an in-depth look at the Segment Anything Model and the SA-1B dataset, please visit the [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything) and check out the research paper [Segment Anything](https://arxiv.org/abs/2304.02643).

!!! tip "SAM on Ultralytics Platform"

    SAM-family models (SAM 2.1, SAM 3, and SAM 3.1) power the [smart annotation feature](../platform/data/annotation.md#smart-annotation) on [Ultralytics Platform](https://platform.ultralytics.com), enabling click-based masking for fast dataset labeling.

## Available Models, Supported Tasks, and Operating Modes

This table presents the available models with their specific pretrained weights, the tasks they support, and their compatibility with different operating modes like [Training](../modes/train.md), [Validation](../modes/val.md), [Inference](../modes/predict.md), and [Export](../modes/export.md). Supported modes are indicated by ✅ and unsupported modes by ❌.

| Model Type | Pretrained Weights                                                                  | Tasks Supported                              | Training | Validation | Inference | Export |
| ---------- | ----------------------------------------------------------------------------------- | -------------------------------------------- | -------- | ---------- | --------- | ------ |
| SAM base   | [sam_b.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam_b.pt) | [Instance Segmentation](../tasks/segment.md) | ❌       | ❌         | ✅        | ❌     |
| SAM large  | [sam_l.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/sam_l.pt) | [Instance Segmentation](../tasks/segment.md) | ❌       | ❌         | ✅        | ❌     |

## How to Use SAM: Versatility and Power in Image Segmentation

The Segment Anything Model can be used for many downstream tasks beyond its training data, including edge detection, object proposal generation, [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation), and preliminary text-to-mask prediction. With prompt engineering, SAM can adapt to new tasks and data distributions in a zero-shot manner.

### SAM prediction example

!!! example "Segment with prompts"

    Segment image with given prompts.

    === "Python"

        ```python
        from ultralytics import SAM

        # Load a model
        model = SAM("sam_b.pt")

        # Display model information (optional)
        model.info()

        # Run inference with bboxes prompt
        results = model("ultralytics/assets/zidane.jpg", bboxes=[439, 437, 524, 709])

        # Run inference with single point
        results = model(points=[900, 370], labels=[1])

        # Run inference with multiple points
        results = model(points=[[400, 370], [900, 370]], labels=[1, 1])

        # Run inference with multiple points prompt per object
        results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

        # Run inference with negative points prompt
        results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 0]])
        ```

!!! example "Segment everything"

    Segment the whole image.

    === "Python"

        ```python
        from ultralytics import SAM

        # Load a model
        model = SAM("sam_b.pt")

        # Display model information (optional)
        model.info()

        # Run inference
        model("path/to/image.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with a SAM model
        yolo predict model=sam_b.pt source=path/to/image.jpg
        ```

- SAM segments the whole image when you don't pass any prompts (bboxes/points/masks).

!!! example "SAMPredictor example"

    Set the image once, then run prompt inference multiple times without re-running the image encoder.

    === "Prompt inference"

        ```python
        import cv2

        from ultralytics.models.sam import Predictor as SAMPredictor

        # Create SAMPredictor
        overrides = {"conf": 0.25, "task": "segment", "mode": "predict", "imgsz": 1024, "model": "mobile_sam.pt"}
        predictor = SAMPredictor(overrides=overrides)

        # Set image
        predictor.set_image("ultralytics/assets/zidane.jpg")  # set with image file
        predictor.set_image(cv2.imread("ultralytics/assets/zidane.jpg"))  # set with np.ndarray
        results = predictor(bboxes=[439, 437, 524, 709])

        # Run inference with single point prompt
        results = predictor(points=[900, 370], labels=[1])

        # Run inference with multiple points prompt
        results = predictor(points=[[400, 370], [900, 370]], labels=[1, 1])

        # Run inference with negative points prompt
        results = predictor(points=[[[400, 370], [900, 370]]], labels=[[1, 0]])

        # Reset image
        predictor.reset_image()
        ```

    Segment everything with additional args.

    === "Segment everything"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # Create SAMPredictor
        overrides = {"conf": 0.25, "task": "segment", "mode": "predict", "imgsz": 1024, "model": "mobile_sam.pt"}
        predictor = SAMPredictor(overrides=overrides)

        # Segment with additional args
        results = predictor(source="ultralytics/assets/zidane.jpg", crop_n_layers=1, points_stride=64, min_mask_region_area=100)
        ```

!!! note

    All the returned `results` in the above examples are [Results](../modes/predict.md#working-with-results) objects which provide access to the predicted masks and the source image.

- For more `Segment everything` args, see the [`Predictor/generate` Reference](../reference/models/sam/predict.md).

## SAM Comparison vs YOLO

Here we compare Meta's SAM-b model with Ultralytics segmentation models, including [YOLO26](yolo26.md):

| Model                                                                                          | Size<br><sup>(MB)</sup> | Parameters<br><sup>(M)</sup> | Speed (CPU)<br><sup>(ms/im)</sup> |
| ---------------------------------------------------------------------------------------------- | ----------------------- | ---------------------------- | --------------------------------- |
| Meta SAM-b                                                                                     | 375                     | 93.7                         | 41703                             |
| [MobileSAM](mobile-sam.md)                                                                     | 40.7                    | 10.1                         | 23802                             |
| [FastSAM-s](fast-sam.md) with YOLOv8 [backbone](https://www.ultralytics.com/glossary/backbone) | 23.9                    | 11.8                         | 58.0                              |
| Ultralytics [YOLOv8n-seg](https://platform.ultralytics.com/ultralytics/yolov8/yolov8n-seg)     | 7.1 (52.8x smaller)     | 3.4 (27.6x less)             | 24.8 (1682x faster)               |
| Ultralytics [YOLO11n-seg](https://platform.ultralytics.com/ultralytics/yolo11/yolo11n-seg)     | **6.2** (60.5x smaller) | 2.9 (32.3x less)             | **24.3** (1716x faster)           |
| Ultralytics [YOLO26n-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-seg)     | 6.7 (56.0x smaller)     | **2.7** (34.7x less)         | 25.2 (1655x faster)               |

This comparison demonstrates the substantial differences in model sizes and speeds between SAM variants and YOLO segmentation models. While SAM provides unique automatic segmentation capabilities, YOLO models, particularly YOLOv8n-seg, YOLO11n-seg and YOLO26n-seg, are significantly smaller, faster, and more computationally efficient.

SAM speeds measured with PyTorch, YOLO speeds measured with ONNX Runtime. Tests run on a 2025 Apple M4 Air with 16GB of RAM using `torch==2.10.0`, `ultralytics==8.4.31`, and `onnxruntime==1.24.4`. To reproduce this test:

!!! example

    === "Python"

        ```python
        from ultralytics import ASSETS, SAM, YOLO, FastSAM

        # Profile SAM-b, MobileSAM
        for file in ["sam_b.pt", "mobile_sam.pt"]:
            model = SAM(file)
            model.info()
            model(ASSETS)

        # Profile FastSAM-s
        model = FastSAM("FastSAM-s.pt")
        model.info()
        model(ASSETS)

        # Profile YOLO models (ONNX)
        for file_name in ["yolov8n-seg.pt", "yolo11n-seg.pt", "yolo26n-seg.pt"]:
            model = YOLO(file_name)
            model.info()
            onnx_path = model.export(format="onnx", dynamic=True, nms=False)  # YOLO26 NMS-free head; no-op for YOLOv8/YOLO11
            model = YOLO(onnx_path)
            model(ASSETS)
        ```

## Auto-Annotation: A Quick Path to Segmentation Datasets

Auto-annotation is a key feature of SAM, allowing users to generate a [segmentation dataset](../datasets/segment/index.md) using a pretrained detection model. This feature enables rapid annotation of large numbers of images, bypassing time-consuming manual labeling.

### Generate Your Segmentation Dataset Using a Detection Model

To auto-annotate your dataset with the Ultralytics framework, use the `auto_annotate` function, which pairs a detector such as [YOLO26x](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x) with SAM:

!!! example

    === "Python"

        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data="path/to/images", det_model="yolo26x.pt", sam_model="sam_b.pt")
        ```

{% include "macros/sam-auto-annotate.md" %}

The `auto_annotate` function takes the path to your images, with optional arguments for specifying the pretrained detection and SAM segmentation models, the device to run the models on, and the output directory for saving the annotated results.

Auto-annotation with pretrained models greatly reduces the time and effort required to create segmentation datasets, letting teams with large image collections focus on model development and evaluation rather than manual annotation.

## Citations and Acknowledgments

If you find SAM useful in your research or development work, please consider citing the paper:

!!! quote "Citation"

    === "BibTeX"

        ```bibtex
        @misc{kirillov2023segment,
              title={Segment Anything},
              author={Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and Laura Gustafson and Tete Xiao and Spencer Whitehead and Alexander C. Berg and Wan-Yen Lo and Piotr Dollár and Ross Girshick},
              year={2023},
              eprint={2304.02643},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

We would like to express our gratitude to Meta AI for creating and maintaining this valuable resource for the [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) community.

## FAQ

### What is the Segment Anything Model (SAM)?

Meta's Segment Anything Model (SAM) is an image segmentation model designed for promptable segmentation tasks. It combines image and prompt encoders with a lightweight mask decoder to generate segmentation masks from prompts such as points, boxes, or masks. Trained on the SA-1B dataset, SAM performs well zero-shot, adapting to new image distributions and tasks without prior knowledge.

### How can I use the Segment Anything Model (SAM) for image segmentation?

You can use the Segment Anything Model (SAM) for image segmentation by running inference with various prompts such as bounding boxes or points. Here's an example using Python:

```python
from ultralytics import SAM

# Load a model
model = SAM("sam_b.pt")

# Segment with bounding box prompt
model("ultralytics/assets/zidane.jpg", bboxes=[439, 437, 524, 709])

# Segment with points prompt
model("ultralytics/assets/zidane.jpg", points=[900, 370], labels=[1])

# Segment with multiple points prompt
model("ultralytics/assets/zidane.jpg", points=[[400, 370], [900, 370]], labels=[1, 1])

# Segment with multiple points prompt per object
model("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

# Segment with negative points prompt.
model("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 0]])
```

Alternatively, you can run inference with SAM in the command line interface (CLI):

```bash
yolo predict model=sam_b.pt source=path/to/image.jpg
```

### How do SAM and YOLO models compare in terms of performance?

Compared to YOLO models, SAM variants like SAM-b, MobileSAM, and FastSAM-s are typically larger and slower but offer unique zero-shot segmentation capabilities. For example, [YOLO26n-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-seg) is **56x smaller** and over **1650x faster** than Meta's original SAM-b model on CPU. This makes YOLO models ideal for applications requiring rapid, lightweight, and computationally efficient segmentation, while SAM models excel in flexible, promptable, and zero-shot segmentation tasks.

### How can I auto-annotate my dataset using SAM?

Ultralytics' SAM offers an auto-annotation feature that allows generating segmentation datasets using a pretrained detection model. Here's an example in Python:

```python
from ultralytics.data.annotator import auto_annotate

auto_annotate(data="path/to/images", det_model="yolo26x.pt", sam_model="sam_b.pt")
```

This function takes the path to your images and optional arguments for pretrained detection and SAM segmentation models, along with device and output directory specifications. For a complete guide, see [Auto-Annotation](#auto-annotation-a-quick-path-to-segmentation-datasets).

### What datasets are used to train the Segment Anything Model (SAM)?

SAM is trained on the SA-1B dataset, which comprises over 1 billion masks across 11 million images. SA-1B was the largest segmentation dataset at its release, providing high-quality and diverse [training data](https://www.ultralytics.com/glossary/training-data) that underpins SAM's zero-shot performance across segmentation tasks. For more details, visit the [Dataset section](#key-features-of-the-segment-anything-model-sam).
