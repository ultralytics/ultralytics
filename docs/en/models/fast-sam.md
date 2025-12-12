---
comments: true
description: Discover FastSAM, a real-time CNN-based solution for segmenting any object in an image. Efficient, competitive, and ideal for various vision tasks.
keywords: FastSAM, Fast Segment Anything Model, Ultralytics, real-time segmentation, CNN, YOLOv8-seg, object segmentation, image processing, computer vision
---

# Fast Segment Anything Model (FastSAM)

The Fast Segment Anything Model (FastSAM) is a novel, real-time CNN-based solution for the Segment Anything task. This task is designed to segment any object within an image based on various possible user interaction prompts. FastSAM significantly reduces computational demands while maintaining competitive performance, making it a practical choice for a variety of vision tasks.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/F7db-EHhxss"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Object Tracking using FastSAM with Ultralytics
</p>

## Model Architecture

![Fast Segment Anything Model (FastSAM) architecture overview](https://github.com/ultralytics/docs/releases/download/0/fastsam-architecture-overview.avif)

## Overview

FastSAM is designed to address the limitations of the [Segment Anything Model (SAM)](sam.md), a heavy [Transformer](https://www.ultralytics.com/glossary/transformer) model with substantial computational resource requirements. The FastSAM decouples the segment anything task into two sequential stages: all-[instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) and prompt-guided selection. The first stage uses [YOLOv8-seg](../tasks/segment.md) to produce the segmentation masks of all instances in the image. In the second stage, it outputs the region-of-interest corresponding to the prompt.

## Key Features

1. **Real-time Solution:** By leveraging the computational efficiency of CNNs, FastSAM provides a real-time solution for the segment anything task, making it valuable for industrial applications that require quick results.

2. **Efficiency and Performance:** FastSAM offers a significant reduction in computational and resource demands without compromising on performance quality. It achieves comparable performance to SAM but with drastically reduced computational resources, enabling real-time application.

3. **Prompt-guided Segmentation:** FastSAM can segment any object within an image guided by various possible user interaction prompts, providing flexibility and adaptability in different scenarios.

4. **Based on YOLOv8-seg:** FastSAM is based on [YOLOv8-seg](../tasks/segment.md), an object detector equipped with an instance segmentation branch. This allows it to effectively produce the segmentation masks of all instances in an image.

5. **Competitive Results on Benchmarks:** On the object proposal task on MS COCO, FastSAM achieves high scores at a significantly faster speed than [SAM](sam.md) on a single NVIDIA RTX 3090, demonstrating its efficiency and capability.

6. **Practical Applications:** The proposed approach provides a new, practical solution for a large number of vision tasks at a really high speed, tens or hundreds of times faster than current methods.

7. **Model Compression Feasibility:** FastSAM demonstrates the feasibility of a path that can significantly reduce the computational effort by introducing an artificial prior to the structure, thus opening new possibilities for large model architecture for general vision tasks.

## Available Models, Supported Tasks, and Operating Modes

This table presents the available models with their specific pretrained weights, the tasks they support, and their compatibility with different operating modes like [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), and [Export](../modes/export.md), indicated by ✅ emojis for supported modes and ❌ emojis for unsupported modes.

| Model Type | Pretrained Weights                                                                          | Tasks Supported                              | Inference | Validation | Training | Export |
| ---------- | ------------------------------------------------------------------------------------------- | -------------------------------------------- | --------- | ---------- | -------- | ------ |
| FastSAM-s  | [FastSAM-s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-s.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ✅     |
| FastSAM-x  | [FastSAM-x.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-x.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ✅     |

## FastSAM Comparison vs YOLO

Here we compare Meta's SAM 2 models, including the smallest SAM2-t variant, with Ultralytics smallest segmentation model, [YOLO11n-seg](../tasks/segment.md):

| Model                                                                                          | Size<br><sup>(MB)</sup> | Parameters<br><sup>(M)</sup> | Speed (CPU)<br><sup>(ms/im)</sup> |
| ---------------------------------------------------------------------------------------------- | ----------------------- | ---------------------------- | --------------------------------- |
| [Meta SAM-b](sam.md)                                                                           | 375                     | 93.7                         | 49401                             |
| [Meta SAM2-b](sam-2.md)                                                                        | 162                     | 80.8                         | 31901                             |
| [Meta SAM2-t](sam-2.md)                                                                        | 78.1                    | 38.9                         | 25997                             |
| [MobileSAM](mobile-sam.md)                                                                     | 40.7                    | 10.1                         | 25381                             |
| [FastSAM-s](fast-sam.md) with YOLOv8 [backbone](https://www.ultralytics.com/glossary/backbone) | 23.7                    | 11.8                         | 55.9                              |
| Ultralytics [YOLOv8n-seg](yolov8.md)                                                           | **6.7** (11.7x smaller) | **3.4** (11.4x less)         | **24.5** (1061x faster)           |
| Ultralytics [YOLO11n-seg](yolo11.md)                                                           | **5.9** (13.2x smaller) | **2.9** (13.4x less)         | **30.1** (864x faster)            |

This comparison demonstrates the substantial differences in model sizes and speeds between SAM variants and YOLO segmentation models. While SAM provides unique automatic segmentation capabilities, YOLO models, particularly YOLOv8n-seg and YOLO11n-seg, are significantly smaller, faster, and more computationally efficient.

Tests run on a 2025 Apple M4 Pro with 24GB of RAM using `torch==2.6.0` and `ultralytics==8.3.90`. To reproduce this test:

!!! example

    === "Python"

        ```python
        from ultralytics import ASSETS, SAM, YOLO, FastSAM

        # Profile SAM2-t, SAM2-b, SAM-b, MobileSAM
        for file in ["sam_b.pt", "sam2_b.pt", "sam2_t.pt", "mobile_sam.pt"]:
            model = SAM(file)
            model.info()
            model(ASSETS)

        # Profile FastSAM-s
        model = FastSAM("FastSAM-s.pt")
        model.info()
        model(ASSETS)

        # Profile YOLO models
        for file_name in ["yolov8n-seg.pt", "yolo11n-seg.pt"]:
            model = YOLO(file_name)
            model.info()
            model(ASSETS)
        ```

## Usage Examples

The FastSAM models are easy to integrate into your Python applications. Ultralytics provides user-friendly Python API and CLI commands to streamline development.

### Predict Usage

To perform [object detection](https://www.ultralytics.com/glossary/object-detection) on an image, use the `predict` method as shown below:

!!! example

    === "Python"

        ```python
        from ultralytics import FastSAM

        # Define an inference source
        source = "path/to/bus.jpg"

        # Create a FastSAM model
        model = FastSAM("FastSAM-s.pt")  # or FastSAM-x.pt

        # Run inference on an image
        everything_results = model(source, device="cpu", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

        # Run inference with bboxes prompt
        results = model(source, bboxes=[439, 437, 524, 709])

        # Run inference with points prompt
        results = model(source, points=[[200, 200]], labels=[1])

        # Run inference with texts prompt
        results = model(source, texts="a photo of a dog")

        # Run inference with bboxes and points and texts prompt at the same time
        results = model(source, bboxes=[439, 437, 524, 709], points=[[200, 200]], labels=[1], texts="a photo of a dog")
        ```

    === "CLI"

        ```bash
        # Load a FastSAM model and segment everything with it
        yolo segment predict model=FastSAM-s.pt source=path/to/bus.jpg imgsz=640
        ```

This snippet demonstrates the simplicity of loading a pretrained model and running a prediction on an image.

!!! example "FastSAMPredictor example"

    This way you can run inference on image and get all the segment `results` once and run prompts inference multiple times without running inference multiple times.

    === "Prompt inference"

        ```python
        from ultralytics.models.fastsam import FastSAMPredictor

        # Create FastSAMPredictor
        overrides = dict(conf=0.25, task="segment", mode="predict", model="FastSAM-s.pt", save=False, imgsz=1024)
        predictor = FastSAMPredictor(overrides=overrides)

        # Segment everything
        everything_results = predictor("ultralytics/assets/bus.jpg")

        # Prompt inference
        bbox_results = predictor.prompt(everything_results, bboxes=[[200, 200, 300, 300]])
        point_results = predictor.prompt(everything_results, points=[200, 200])
        text_results = predictor.prompt(everything_results, texts="a photo of a dog")
        ```

!!! note

    All the returned `results` in above examples are [Results](../modes/predict.md#working-with-results) object which allows access predicted masks and source image easily.

### Val Usage

Validation of the model on a dataset can be done as follows:

!!! example

    === "Python"

        ```python
        from ultralytics import FastSAM

        # Create a FastSAM model
        model = FastSAM("FastSAM-s.pt")  # or FastSAM-x.pt

        # Validate the model
        results = model.val(data="coco8-seg.yaml")
        ```

    === "CLI"

        ```bash
        # Load a FastSAM model and validate it on the COCO8 example dataset at image size 640
        yolo segment val model=FastSAM-s.pt data=coco8.yaml imgsz=640
        ```

Please note that FastSAM only supports detection and segmentation of a single class of object. This means it will recognize and segment all objects as the same class. Therefore, when preparing the dataset, you need to convert all object category IDs to 0.

### Track Usage

To perform object tracking on an image, use the `track` method as shown below:

!!! example

    === "Python"

        ```python
        from ultralytics import FastSAM

        # Create a FastSAM model
        model = FastSAM("FastSAM-s.pt")  # or FastSAM-x.pt

        # Track with a FastSAM model on a video
        results = model.track(source="path/to/video.mp4", imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo segment track model=FastSAM-s.pt source="path/to/video.mp4" imgsz=640
        ```

## FastSAM official Usage

FastSAM is also available directly from the [https://github.com/CASIA-IVA-Lab/FastSAM](https://github.com/CASIA-LMC-Lab/FastSAM) repository. Here is a brief overview of the typical steps you might take to use FastSAM:

### Installation

1. Clone the FastSAM repository:

    ```bash
    git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
    ```

2. Create and activate a Conda environment with Python 3.9:

    ```bash
    conda create -n FastSAM python=3.9
    conda activate FastSAM
    ```

3. Navigate to the cloned repository and install the required packages:

    ```bash
    cd FastSAM
    pip install -r requirements.txt
    ```

4. Install the CLIP model:
    ```bash
    pip install git+https://github.com/ultralytics/CLIP.git
    ```

### Example Usage

1. Download a [model checkpoint](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing).

2. Use FastSAM for inference. Example commands:
    - Segment everything in an image:

        ```bash
        python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg
        ```

    - Segment specific objects using text prompt:

        ```bash
        python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --text_prompt "the yellow dog"
        ```

    - Segment objects within a [bounding box](https://www.ultralytics.com/glossary/bounding-box) (provide box coordinates in xywh format):

        ```bash
        python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --box_prompt "[570,200,230,400]"
        ```

    - Segment objects near specific points:
        ```bash
        python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --point_prompt "[[520,360],[620,300]]" --point_label "[1,0]"
        ```

Additionally, you can try FastSAM through the CASIA-IVA-Lab [Colab demo](https://colab.research.google.com/drive/1oX14f6IneGGw612WgVlAiy91UHwFAvr9?usp=sharing).

## Citations and Acknowledgments

We would like to acknowledge the FastSAM authors for their significant contributions in the field of real-time instance segmentation:

!!! quote ""

    === "BibTeX"

      ```bibtex
      @misc{zhao2023fast,
            title={Fast Segment Anything},
            author={Xu Zhao and Wenchao Ding and Yongqi An and Yinglong Du and Tao Yu and Min Li and Ming Tang and Jinqiao Wang},
            year={2023},
            eprint={2306.12156},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
      }
      ```

The original FastSAM paper can be found on [arXiv](https://arxiv.org/abs/2306.12156). The authors have made their work publicly available, and the codebase can be accessed on [GitHub](https://github.com/CASIA-LMC-Lab/FastSAM). We appreciate their efforts in advancing the field and making their work accessible to the broader community.

## FAQ

### What is FastSAM and how does it differ from SAM?

FastSAM, short for Fast Segment Anything Model, is a real-time [convolutional neural network](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) (CNN)-based solution designed to reduce computational demands while maintaining high performance in object segmentation tasks. Unlike the Segment Anything Model (SAM), which uses a heavier Transformer-based architecture, FastSAM leverages [Ultralytics YOLOv8-seg](../tasks/segment.md) for efficient instance segmentation in two stages: all-instance segmentation followed by prompt-guided selection.

### How does FastSAM achieve real-time segmentation performance?

FastSAM achieves real-time segmentation by decoupling the segmentation task into all-instance segmentation with YOLOv8-seg and prompt-guided selection stages. By utilizing the computational efficiency of CNNs, FastSAM offers significant reductions in computational and resource demands while maintaining competitive performance. This dual-stage approach enables FastSAM to deliver fast and efficient segmentation suitable for applications requiring quick results.

### What are the practical applications of FastSAM?

FastSAM is practical for a variety of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks that require real-time segmentation performance. Applications include:

- Industrial automation for quality control and assurance
- Real-time video analysis for security and surveillance
- [Autonomous vehicles](https://www.ultralytics.com/glossary/autonomous-vehicles) for object detection and segmentation
- Medical imaging for precise and quick segmentation tasks

Its ability to handle various user interaction prompts makes FastSAM adaptable and flexible for diverse scenarios.

### How do I use the FastSAM model for inference in Python?

To use FastSAM for inference in Python, you can follow the example below:

```python
from ultralytics import FastSAM

# Define an inference source
source = "path/to/bus.jpg"

# Create a FastSAM model
model = FastSAM("FastSAM-s.pt")  # or FastSAM-x.pt

# Run inference on an image
everything_results = model(source, device="cpu", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

# Run inference with bboxes prompt
results = model(source, bboxes=[439, 437, 524, 709])

# Run inference with points prompt
results = model(source, points=[[200, 200]], labels=[1])

# Run inference with texts prompt
results = model(source, texts="a photo of a dog")

# Run inference with bboxes and points and texts prompt at the same time
results = model(source, bboxes=[439, 437, 524, 709], points=[[200, 200]], labels=[1], texts="a photo of a dog")
```

For more details on inference methods, check the [Predict Usage](#predict-usage) section of the documentation.

### What types of prompts does FastSAM support for segmentation tasks?

FastSAM supports multiple prompt types for guiding the segmentation tasks:

- **Everything Prompt**: Generates segmentation for all visible objects.
- **Bounding Box (BBox) Prompt**: Segments objects within a specified bounding box.
- **Text Prompt**: Uses a descriptive text to segment objects matching the description.
- **Point Prompt**: Segments objects near specific user-defined points.

This flexibility allows FastSAM to adapt to a wide range of user interaction scenarios, enhancing its utility across different applications. For more information on using these prompts, refer to the [Key Features](#key-features) section.
