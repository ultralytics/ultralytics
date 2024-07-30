---
comments: true
description: Discover SAM2, the next generation of Meta's Segment Anything Model, supporting real-time promptable segmentation in both images and videos with state-of-the-art performance. Learn about its key features, datasets, and how to use it.
keywords: SAM2, Segment Anything, video segmentation, image segmentation, promptable segmentation, zero-shot performance, SA-V dataset, Ultralytics, real-time segmentation, AI, machine learning
---

# SAM2: Segment Anything Model 2

Experience the next evolution in object segmentation with SAM2. This advanced model extends the capabilities of the original Segment Anything Model (SAM) to support real-time, promptable segmentation across both images and videos. SAM2's innovative design and robust dataset make it a powerful tool for a wide range of applications.

## Introduction to SAM2

SAM2 builds upon the foundational concepts of SAM, introducing a unified model architecture that excels in handling dynamic visual data. It is designed to perform zero-shot generalization, allowing it to segment unseen objects in new visual domains without additional training. This capability, coupled with real-time processing speeds, sets SAM2 apart as a leading model for both image and video segmentation.

Trained on the comprehensive [SA-V dataset](https://ai.meta.com/datasets/segment-anything/), which includes over 51,000 videos and 600,000 mask annotations, SAM2 is equipped to deliver state-of-the-art performance in diverse scenarios.

![Dataset sample image](https://user-images.githubusercontent.com/26833433/238056229-0e8ffbeb-f81a-477e-a490-aff3d82fd8ce.jpg)
**SA-V Example images.** The SA-V dataset includes diverse, real-world videos with detailed mask annotations, providing a rich training ground for SAM2.

## Key Features of SAM2

- **Unified Model Architecture:** SAM2 integrates image and video segmentation capabilities into a single model, facilitating seamless transitions between media types.
- **Real-Time Performance:** Achieves approximately 44 frames per second, making it suitable for real-time applications.
- **Zero-Shot Generalization:** Capable of segmenting objects in unfamiliar domains without needing domain-specific training.
- **Interactive Refinement:** Supports iterative refinement through user prompts, allowing precise control over segmentation output.
- **Advanced Handling of Visual Challenges:** Equipped with mechanisms to manage occlusions, reappearances, and ambiguous scenarios through multi-mask predictions.

For a deeper understanding of SAM2's architecture and capabilities, explore the [SAM2 research paper](https://arxiv.org/abs/2401.12741).

## Available Models, Supported Tasks, and Operating Modes

The following table details the available SAM2 models, their pre-trained weights, supported tasks, and compatibility with different operating modes like [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), and [Export](../modes/export.md).

| Model Type | Pre-trained Weights                                                                 | Tasks Supported                              | Inference | Validation | Training | Export |
|------------|-------------------------------------------------------------------------------------|----------------------------------------------|-----------|------------|----------|--------|
| SAM2 base  | [sam2_b.pt](https://github.com/ultralytics/assets/releases/download/v9.0/sam2_b.pt) | [Instance Segmentation](../tasks/segment.md) | ❌         | ❌          | ❌        | ❌      |
| SAM2 large | [sam2_l.pt](https://github.com/ultralytics/assets/releases/download/v9.0/sam2_l.pt) | [Instance Segmentation](../tasks/segment.md) | ❌         | ❌          | ❌        | ❌      |

## How to Use SAM2: Versatility in Image and Video Segmentation

SAM2 can be utilized across a broad spectrum of tasks, including real-time video editing, medical imaging, and autonomous systems. Its ability to segment both static and dynamic visual data makes it a versatile tool for researchers and developers.

### SAM2 Prediction Examples

#### Segment with Prompts

!!! Example "Segment with Prompts"

    Use prompts to segment specific objects in images or videos.

    === "Python"

        ```python
        from ultralytics import SAM2

        # Load a model
        model = SAM2("sam2_b.pt")

        # Display model information (optional)
        model.info()

        # Segment with bounding box prompt
        results = model("path/to/image.jpg", bboxes=[100, 100, 200, 200])

        # Segment with point prompt
        results = model("path/to/image.jpg", points=[150, 150], labels=[1])
        ```

#### Segment Everything

!!! Example "Segment Everything"

    Segment the entire image or video content without specific prompts.

    === "Python"

        ```python
        from ultralytics import SAM2

        # Load a model
        model = SAM2("sam2_b.pt")

        # Display model information (optional)
        model.info()

        # Run inference
        model("path/to/video.mp4")
        ```

    === "CLI"

        ```bash
        # Run inference with a SAM2 model
        yolo predict model=sam2_b.pt source=path/to/video.mp4
        ```

- This example demonstrates how SAM2 can be used to segment the entire content of an image or video if no prompts (bboxes/points/masks) are provided.

## SAM comparison vs YOLOv8

Here we compare Meta's smallest SAM model, SAM-b, with Ultralytics smallest segmentation model, [YOLOv8n-seg](../tasks/segment.md):

| Model                                          | Size                       | Parameters             | Speed (CPU)                |
|------------------------------------------------|----------------------------|------------------------|----------------------------|
| Meta's SAM-b                                   | 358 MB                     | 94.7 M                 | 51096 ms/im                |
| [MobileSAM](mobile-sam.md)                     | 40.7 MB                    | 10.1 M                 | 46122 ms/im                |
| [FastSAM-s](fast-sam.md) with YOLOv8 backbone  | 23.7 MB                    | 11.8 M                 | 115 ms/im                  |
| Ultralytics [YOLOv8n-seg](../tasks/segment.md) | **6.7 MB** (53.4x smaller) | **3.4 M** (27.9x less) | **59 ms/im** (866x faster) |

This comparison shows the order-of-magnitude differences in the model sizes and speeds between models. Whereas SAM presents unique capabilities for automatic segmenting, it is not a direct competitor to YOLOv8 segment models, which are smaller, faster and more efficient.

Tests run on a 2023 Apple M2 Macbook with 16GB of RAM. To reproduce this test:

!!! Example

    === "Python"

        ```python
        from ultralytics import SAM, YOLO, FastSAM

        # Profile SAM-b
        model = SAM("sam_b.pt")
        model.info()
        model("ultralytics/assets")

        # Profile MobileSAM
        model = SAM("mobile_sam.pt")
        model.info()
        model("ultralytics/assets")

        # Profile FastSAM-s
        model = FastSAM("FastSAM-s.pt")
        model.info()
        model("ultralytics/assets")

        # Profile YOLOv8n-seg
        model = YOLO("yolov8n-seg.pt")
        model.info()
        model("ultralytics/assets")
        ```

## Auto-Annotation: Efficient Dataset Creation

Auto-annotation is a powerful feature of SAM2, enabling users to generate segmentation datasets quickly and accurately by leveraging pre-trained models. This capability is particularly useful for creating large, high-quality datasets without extensive manual effort.

### How to Auto-Annotate with SAM2

To auto-annotate your dataset using SAM2, follow this example:

!!! Example "Auto-Annotation Example"

    ```python
    from ultralytics.data.annotator import auto_annotate

    auto_annotate(data="path/to/images", det_model="yolov8x.pt", sam_model="sam2_b.pt")
    ```

| Argument   | Type                | Description                                                                                             | Default      |
|------------|---------------------|---------------------------------------------------------------------------------------------------------|--------------|
| data       | str                 | Path to a folder containing images to be annotated.                                                     |              |
| det_model  | str, optional       | Pre-trained YOLO detection model. Defaults to 'yolov8x.pt'.                                             | 'yolov8x.pt' |
| sam_model  | str, optional       | Pre-trained SAM2 segmentation model. Defaults to 'sam2_b.pt'.                                           | 'sam2_b.pt'  |
| device     | str, optional       | Device to run the models on. Defaults to an empty string (CPU or GPU, if available).                    |              |
| output_dir | str, None, optional | Directory to save the annotated results. Defaults to a 'labels' folder in the same directory as 'data'. | None         |

This function facilitates the rapid creation of high-quality segmentation datasets, ideal for researchers and developers aiming to accelerate their projects.

## Citations and Acknowledgements

If SAM2 is a crucial part of your research or development work, please cite it using the following reference:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{kirillov2024sam2,
          title={SAM2: Segment Anything Model 2},
          author={Alexander Kirillov and others},
          journal={arXiv preprint arXiv:2401.12741},
          year={2024}
        }
        ```

We extend our gratitude to Meta AI for their contributions to the AI community with this groundbreaking model and dataset.
