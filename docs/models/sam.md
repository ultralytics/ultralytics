---
comments: true
description: Explore the cutting-edge Segment Anything Model (SAM) from Ultralytics that allows real-time image segmentation. Learn about its promptable segmentation, zero-shot performance, and how to use it.
keywords: Ultralytics, image segmentation, Segment Anything Model, SAM, SA-1B dataset, real-time performance, zero-shot transfer, object detection, image analysis, machine learning
---

# Segment Anything Model (SAM)

Welcome to the frontier of image segmentation with the Segment Anything Model, or SAM. This revolutionary model has changed the game by introducing promptable image segmentation with real-time performance, setting new standards in the field.

## Introduction to SAM: The Segment Anything Model

The Segment Anything Model, or SAM, is a cutting-edge image segmentation model that allows for promptable segmentation, providing unparalleled versatility in image analysis tasks. SAM forms the heart of the Segment Anything initiative, a groundbreaking project that introduces a novel model, task, and dataset for image segmentation.

SAM's advanced design allows it to adapt to new image distributions and tasks without prior knowledge, a feature known as zero-shot transfer. Trained on the expansive [SA-1B dataset](https://ai.facebook.com/datasets/segment-anything/), which contains more than 1 billion masks spread over 11 million carefully curated images, SAM has displayed impressive zero-shot performance, surpassing previous fully supervised results in many cases.

![Dataset sample image](https://user-images.githubusercontent.com/26833433/238056229-0e8ffbeb-f81a-477e-a490-aff3d82fd8ce.jpg)
Example images with overlaid masks from our newly introduced dataset, SA-1B. SA-1B contains 11M diverse, high-resolution, licensed, and privacy protecting images and 1.1B high-quality segmentation masks. These masks were annotated fully automatically by SAM, and as verified by human ratings and numerous experiments, are of high quality and diversity. Images are grouped by number of masks per image for visualization (there are ∼100 masks per image on average).

## Key Features of the Segment Anything Model (SAM)

- **Promptable Segmentation Task:** SAM was designed with a promptable segmentation task in mind, allowing it to generate valid segmentation masks from any given prompt, such as spatial or text clues identifying an object.
- **Advanced Architecture:** The Segment Anything Model employs a powerful image encoder, a prompt encoder, and a lightweight mask decoder. This unique architecture enables flexible prompting, real-time mask computation, and ambiguity awareness in segmentation tasks.
- **The SA-1B Dataset:** Introduced by the Segment Anything project, the SA-1B dataset features over 1 billion masks on 11 million images. As the largest segmentation dataset to date, it provides SAM with a diverse and large-scale training data source.
- **Zero-Shot Performance:** SAM displays outstanding zero-shot performance across various segmentation tasks, making it a ready-to-use tool for diverse applications with minimal need for prompt engineering.

For an in-depth look at the Segment Anything Model and the SA-1B dataset, please visit the [Segment Anything website](https://segment-anything.com) and check out the research paper [Segment Anything](https://arxiv.org/abs/2304.02643).

## How to Use SAM: Versatility and Power in Image Segmentation

The Segment Anything Model can be employed for a multitude of downstream tasks that go beyond its training data. This includes edge detection, object proposal generation, instance segmentation, and preliminary text-to-mask prediction. With prompt engineering, SAM can swiftly adapt to new tasks and data distributions in a zero-shot manner, establishing it as a versatile and potent tool for all your image segmentation needs.

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
        model("ultralytics/assets/zidane.jpg", bboxes=[439, 437, 524, 709])

        # Run inference with points prompt
        model("ultralytics/assets/zidane.jpg", points=[900, 370], labels=[1])
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

- The logic here is to segment the whole image if you don't pass any prompts(bboxes/points/masks).

!!! example "SAMPredictor example"

    This way you can set image once and run prompts inference multiple times without running image encoder multiple times.

    === "Prompt inference"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # Create SAMPredictor
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(overrides=overrides)

        # Set image
        predictor.set_image("ultralytics/assets/zidane.jpg")  # set with image file
        predictor.set_image(cv2.imread("ultralytics/assets/zidane.jpg"))  # set with np.ndarray
        results = predictor(bboxes=[439, 437, 524, 709])
        results = predictor(points=[900, 370], labels=[1])

        # Reset image
        predictor.reset_image()
        ```

    Segment everything with additional args.

    === "Segment everything"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # Create SAMPredictor
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(overrides=overrides)

        # Segment with additional args
        results = predictor(source="ultralytics/assets/zidane.jpg", crop_n_layers=1, points_stride=64)
        ```

- More additional args for `Segment everything` see [`Predictor/generate` Reference](../reference/models/sam/predict.md).

## Available Models and Supported Tasks

| Model Type | Pre-trained Weights | Tasks Supported       |
| ---------- | ------------------- | --------------------- |
| SAM base   | `sam_b.pt`          | Instance Segmentation |
| SAM large  | `sam_l.pt`          | Instance Segmentation |

## Operating Modes

| Mode       | Supported          |
| ---------- | ------------------ |
| Inference  | :heavy_check_mark: |
| Validation | :x:                |
| Training   | :x:                |

## SAM comparison vs YOLOv8

Here we compare Meta's smallest SAM model, SAM-b, with Ultralytics smallest segmentation model, [YOLOv8n-seg](../tasks/segment.md):

| Model                                          | Size                       | Parameters             | Speed (CPU)                |
| ---------------------------------------------- | -------------------------- | ---------------------- | -------------------------- |
| Meta's SAM-b                                   | 358 MB                     | 94.7 M                 | 51096 ms/im                |
| [MobileSAM](mobile-sam.md)                     | 40.7 MB                    | 10.1 M                 | 46122 ms/im                |
| [FastSAM-s](fast-sam.md) with YOLOv8 backbone  | 23.7 MB                    | 11.8 M                 | 115 ms/im                  |
| Ultralytics [YOLOv8n-seg](../tasks/segment.md) | **6.7 MB** (53.4x smaller) | **3.4 M** (27.9x less) | **59 ms/im** (866x faster) |

This comparison shows the order-of-magnitude differences in the model sizes and speeds between models. Whereas SAM presents unique capabilities for automatic segmenting, it is not a direct competitor to YOLOv8 segment models, which are smaller, faster and more efficient.

Tests run on a 2023 Apple M2 Macbook with 16GB of RAM. To reproduce this test:

!!! example ""

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

## Auto-Annotation: A Quick Path to Segmentation Datasets

Auto-annotation is a key feature of SAM, allowing users to generate a [segmentation dataset](https://docs.ultralytics.com/datasets/segment) using a pre-trained detection model. This feature enables rapid and accurate annotation of a large number of images, bypassing the need for time-consuming manual labeling.

### Generate Your Segmentation Dataset Using a Detection Model

To auto-annotate your dataset with the Ultralytics framework, use the `auto_annotate` function as shown below:

!!! example ""

    === "Python"
        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data="path/to/images", det_model="yolov8x.pt", sam_model="sam_b.pt")
        ```

| Argument   | Type                | Description                                                                                             | Default      |
| ---------- | ------------------- | ------------------------------------------------------------------------------------------------------- | ------------ |
| data       | str                 | Path to a folder containing images to be annotated.                                                     |              |
| det_model  | str, optional       | Pre-trained YOLO detection model. Defaults to 'yolov8x.pt'.                                             | 'yolov8x.pt' |
| sam_model  | str, optional       | Pre-trained SAM segmentation model. Defaults to 'sam_b.pt'.                                             | 'sam_b.pt'   |
| device     | str, optional       | Device to run the models on. Defaults to an empty string (CPU or GPU, if available).                    |              |
| output_dir | str, None, optional | Directory to save the annotated results. Defaults to a 'labels' folder in the same directory as 'data'. | None         |

The `auto_annotate` function takes the path to your images, with optional arguments for specifying the pre-trained detection and SAM segmentation models, the device to run the models on, and the output directory for saving the annotated results.

Auto-annotation with pre-trained models can dramatically cut down the time and effort required for creating high-quality segmentation datasets. This feature is especially beneficial for researchers and developers dealing with large image collections, as it allows them to focus on model development and evaluation rather than manual annotation.

## Citations and Acknowledgements

If you find SAM useful in your research or development work, please consider citing our paper:

!!! note ""

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

We would like to express our gratitude to Meta AI for creating and maintaining this valuable resource for the computer vision community.

_keywords: Segment Anything, Segment Anything Model, SAM, Meta SAM, image segmentation, promptable segmentation, zero-shot performance, SA-1B dataset, advanced architecture, auto-annotation, Ultralytics, pre-trained models, SAM base, SAM large, instance segmentation, computer vision, AI, artificial intelligence, machine learning, data annotation, segmentation masks, detection model, YOLO detection model, bibtex, Meta AI._
