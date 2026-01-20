---
comments: true
description: Explore the Ultralytics COCO8 dataset, a versatile and manageable set of 8 images perfect for testing object detection models and training pipelines.
keywords: COCO8, Ultralytics, dataset, object detection, YOLO26, training, validation, machine learning, computer vision
---

# COCO8 Dataset

## Introduction

The [Ultralytics](https://www.ultralytics.com/) COCO8 dataset is a compact yet powerful [object detection](https://www.ultralytics.com/glossary/object-detection) dataset, consisting of the first 8 images from the COCO train 2017 set—4 for training and 4 for validation. This dataset is specifically designed for rapid testing, debugging, and experimentation with [YOLO](https://docs.ultralytics.com/models/yolo26/) models and training pipelines. Its small size makes it highly manageable, while its diversity ensures it serves as an effective sanity check before scaling up to larger datasets.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/uDrn9QZJ2lk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics COCO Dataset Overview
</p>

COCO8 is fully compatible with [Ultralytics Platform](https://platform.ultralytics.com/) and [YOLO26](../../models/yolo26.md), enabling seamless integration into your computer vision workflows.

## Dataset YAML

The COCO8 dataset configuration is defined in a YAML (Yet Another Markup Language) file, which specifies dataset paths, class names, and other essential metadata. You can review the official `coco8.yaml` file in the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml).

!!! example "ultralytics/cfg/datasets/coco8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8.yaml"
    ```

## Usage

To train a YOLO26n model on the COCO8 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the following examples. For a full list of training options, see the [YOLO Training documentation](../../modes/train.md).

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO26n model
        model = YOLO("yolo26n.pt")

        # Train the model on COCO8
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Train YOLO26n on COCO8 using the command line
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

Below is an example of a mosaiced training batch from the COCO8 dataset:

<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-training-batch-1.avif" alt="COCO8 object detection dataset mosaic training batch" width="800">

- **Mosaiced Image**: This image illustrates a training batch where multiple dataset images are combined using mosaic augmentation. Mosaic augmentation increases the diversity of objects and scenes within each batch, helping the model generalize better to various object sizes, aspect ratios, and backgrounds.

This technique is especially useful for small datasets like COCO8, as it maximizes the value of each image during training.

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

### What Is the Ultralytics COCO8 Dataset Used For?

The Ultralytics COCO8 dataset is designed for rapid testing and debugging of [object detection](https://www.ultralytics.com/glossary/object-detection) models. With only 8 images (4 for training, 4 for validation), it is ideal for verifying your [YOLO](https://docs.ultralytics.com/models/yolo26/) training pipelines and ensuring everything works as expected before scaling to larger datasets. Explore the [COCO8 YAML configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml) for more details.

### How Do I Train a YOLO26 Model Using the COCO8 Dataset?

You can train a YOLO26 model on COCO8 using either Python or the CLI:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO26n model
        model = YOLO("yolo26n.pt")

        # Train the model on COCO8
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

For additional training options, refer to the [YOLO Training documentation](../../modes/train.md).

### Why Should I Use Ultralytics Platform for Managing My COCO8 Training?

[Ultralytics Platform](https://platform.ultralytics.com/) streamlines dataset management, training, and deployment for [YOLO](https://docs.ultralytics.com/models/yolo26/) models—including COCO8. With features like cloud training, real-time monitoring, and intuitive dataset handling, HUB enables you to launch experiments with a single click and eliminates manual setup hassles. Learn more about [Ultralytics Platform](https://platform.ultralytics.com/) and how it can accelerate your computer vision projects.

### What Are the Benefits of Using Mosaic Augmentation in Training With the COCO8 Dataset?

Mosaic augmentation, as used in COCO8 training, combines multiple images into one during each batch. This increases the diversity of objects and backgrounds, helping your [YOLO](https://docs.ultralytics.com/models/yolo26/) model generalize better to new scenarios. Mosaic augmentation is especially valuable for small datasets, as it maximizes the information available in each training step. For more on this, see the [training guide](#usage).

### How Can I Validate My YOLO26 Model Trained on the COCO8 Dataset?

To validate your YOLO26 model after training on COCO8, use the model's validation commands in either Python or CLI. This evaluates your model's performance using standard metrics. For step-by-step instructions, visit the [YOLO Validation documentation](../../modes/val.md).
