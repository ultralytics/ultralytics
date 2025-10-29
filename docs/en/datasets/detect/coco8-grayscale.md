---
comments: true
description: Explore the Ultralytics COCO8-Grayscale dataset, a versatile and manageable set of 8 images perfect for testing object detection models and training pipelines.
keywords: COCO8-Grayscale, Ultralytics, dataset, object detection, YOLO11, training, validation, machine learning, computer vision
---

# COCO8-Grayscale Dataset

## Introduction

The [Ultralytics](https://www.ultralytics.com/) COCO8-Grayscale dataset is a compact yet powerful [object detection](https://www.ultralytics.com/glossary/object-detection) dataset, consisting of the first 8 images from the COCO train 2017 set and converted to grayscale format—4 for training and 4 for validation. This dataset is specifically designed for rapid testing, debugging, and experimentation with [YOLO](https://docs.ultralytics.com/models/yolo11/) grayscale models and training pipelines. Its small size makes it highly manageable, while its diversity ensures it serves as an effective sanity check before scaling up to larger datasets.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yw2Fo6qjJU4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train Ultralytics YOLO11 on Grayscale Datasets 🚀
</p>

COCO8-Grayscale is fully compatible with [Ultralytics HUB](https://hub.ultralytics.com/) and [YOLO11](../../models/yolo11.md), enabling seamless integration into your computer vision workflows.

## Dataset YAML

The COCO8-Grayscale dataset configuration is defined in a YAML (Yet Another Markup Language) file, which specifies dataset paths, class names, and other essential metadata. You can review the official `coco8-grayscale.yaml` file in the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-grayscale.yaml).

!!! note

    To train your RGB images in grayscale, you could simply add `channels: 1` to your dataset YAML file. This converts all images to grayscale during training, enabling you to utilize grayscale benefits without requiring a separate dataset.

!!! example "ultralytics/cfg/datasets/coco8-grayscale.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8-grayscale.yaml"
    ```

## Usage

To train a YOLO11n model on the COCO8-Grayscale dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the following examples. For a full list of training options, see the [YOLO Training documentation](../../modes/train.md).

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO11n model
        model = YOLO("yolo11n.pt")

        # Train the model on COCO8-Grayscale
        results = model.train(data="coco8-grayscale.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Train YOLO11n on COCO8-Grayscale using the command line
        yolo detect train data=coco8-grayscale.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

Below is an example of a mosaiced training batch from the COCO8-Grayscale dataset:

<img src="https://github.com/ultralytics/docs/releases/download/0/grayscale-mosaic.avif" alt="Dataset sample image" width="800">

- **Mosaiced Image**: This image illustrates a training batch where multiple dataset images are combined using mosaic augmentation. Mosaic augmentation increases the diversity of objects and scenes within each batch, helping the model generalize better to various object sizes, aspect ratios, and backgrounds.

This technique is especially useful for small datasets like COCO8-Grayscale, as it maximizes the value of each image during training.

## Citations and Acknowledgments

If you use the COCO dataset in your research or development, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{lin2015microsoft,
              title={Microsoft COCO: Common Objects in Context},
              author={Tsung-Yi Lin and Michael Maire and Serge Belongie and Lubomir Bourdev and Ross Girshick and James Hays and Pietro Perona and Deva Ramanan and C. Lawrence Zitnick and Piotr Dollár},
              year={2015},
              eprint={1405.0312},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

Special thanks to the [COCO Consortium](https://cocodataset.org/#home) for their ongoing contributions to the [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) community.

## FAQ

### What Is the Ultralytics COCO8-Grayscale Dataset Used For?

The Ultralytics COCO8-Grayscale dataset is designed for rapid testing and debugging of [object detection](https://www.ultralytics.com/glossary/object-detection) models. With only 8 images (4 for training, 4 for validation), it is ideal for verifying your [YOLO](https://docs.ultralytics.com/models/yolo11/) training pipelines and ensuring everything works as expected before scaling to larger datasets. Explore the [COCO8-Grayscale YAML configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-grayscale.yaml) for more details.

### How Do I Train a YOLO11 Model Using the COCO8-Grayscale Dataset?

You can train a YOLO11 model on COCO8-Grayscale using either Python or the CLI:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO11n model
        model = YOLO("yolo11n.pt")

        # Train the model on COCO8-Grayscale
        results = model.train(data="coco8-grayscale.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=coco8-grayscale.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

For additional training options, refer to the [YOLO Training documentation](../../modes/train.md).

### Why Should I Use Ultralytics HUB for Managing My COCO8-Grayscale Training?

[Ultralytics HUB](https://hub.ultralytics.com/) streamlines dataset management, training, and deployment for [YOLO](https://docs.ultralytics.com/models/yolo11/) models—including COCO8-Grayscale. With features like cloud training, real-time monitoring, and intuitive dataset handling, HUB enables you to launch experiments with a single click and eliminates manual setup hassles. Learn more about [Ultralytics HUB](https://hub.ultralytics.com/) and how it can accelerate your computer vision projects.

### What Are the Benefits of Using Mosaic Augmentation in Training With the COCO8-Grayscale Dataset?

Mosaic augmentation, as used in COCO8-Grayscale training, combines multiple images into one during each batch. This increases the diversity of objects and backgrounds, helping your [YOLO](https://docs.ultralytics.com/models/yolo11/) model generalize better to new scenarios. Mosaic augmentation is especially valuable for small datasets, as it maximizes the information available in each training step. For more on this, see the [training guide](#usage).

### How Can I Validate My YOLO11 Model Trained on the COCO8-Grayscale Dataset?

To validate your YOLO11 model after training on COCO8-Grayscale, use the model's validation commands in either Python or CLI. This evaluates your model's performance using standard metrics. For step-by-step instructions, visit the [YOLO Validation documentation](../../modes/val.md).
