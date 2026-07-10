---
title: COCO8 Segmentation Dataset
comments: true
description: Discover the versatile and manageable COCO8-Seg dataset by Ultralytics, an 8-image, ~1 MB segmentation set ideal for testing and debugging models.
keywords: COCO8-Seg, Ultralytics, segmentation dataset, YOLO26, COCO 2017, model training, computer vision, dataset configuration
---

# COCO8-Seg Dataset

## Introduction

[Ultralytics](https://www.ultralytics.com/) COCO8-Seg is a small but versatile [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) dataset composed of the first 8 images of the COCO train 2017 set, 4 for training and 4 for validation. This dataset is ideal for testing and debugging segmentation models, or for experimenting with new detection approaches. With 8 images, it is small enough to be easily manageable, yet diverse enough to test training pipelines for errors and act as a sanity check before training larger datasets.

## Dataset Structure

- **Images**: 8 total (4 train / 4 val).
- **Classes**: 80 COCO categories.
- **Labels**: YOLO-format polygons stored under `labels/{train,val}` matching each image file.
- **Download size**: ~1 MB.

Explore [COCO8-Seg on Ultralytics Platform](https://platform.ultralytics.com/ultralytics/datasets/coco8-seg) to browse every image with its polygon masks, view the class distribution and annotation heatmaps in the **Charts** tab, and clone it to train your own model in the cloud.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the COCO8-Seg dataset, the `coco8-seg.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-seg.yaml).

!!! example "ultralytics/cfg/datasets/coco8-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8-seg.yaml"
    ```

## Usage

To train a YOLO26n-seg model on the COCO8-Seg dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-seg.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo segment train data=coco8-seg.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

Here are some examples of images from the COCO8-Seg dataset, along with their corresponding annotations:

<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-training-batch-2.avif" alt="COCO8-seg instance segmentation dataset mosaic" width="800">

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

## Citations and Acknowledgments

If you use the COCO dataset in your research or development work, please cite the following paper:

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

We would like to acknowledge the COCO Consortium for creating and maintaining this valuable resource for the [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) community. For more information about the COCO dataset and its creators, visit the [COCO dataset website](https://cocodataset.org/#home).

## FAQ

### What is the COCO8-Seg dataset, and how is it used in Ultralytics YOLO26?

The **COCO8-Seg dataset** is a compact instance segmentation dataset by Ultralytics, consisting of the first 8 images from the COCO train 2017 set (4 for training, 4 for validation). This dataset is tailored for testing and debugging segmentation models or experimenting with new detection methods. It is particularly useful with Ultralytics [YOLO26](https://github.com/ultralytics/ultralytics) for rapid iteration and pipeline error-checking before scaling to larger datasets. For detailed usage, refer to the model [Training](../../modes/train.md) page.

### How can I train a YOLO26n-seg model using the COCO8-Seg dataset?

To train a **YOLO26n-seg** model on the COCO8-Seg dataset for 100 epochs with an image size of 640, you can use Python or CLI commands. Here's a quick example:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-seg.pt")  # Load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo segment train data=coco8-seg.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
        ```

For a thorough explanation of available arguments and configuration options, you can check the [Training](../../modes/train.md) documentation.

### Why is the COCO8-Seg dataset important for model development and debugging?

Because the ~1 MB download and 8-image train/val loop complete in seconds, COCO8-Seg lets you run a full training-and-validation pass to catch pipeline errors — a broken data loader, a misconfigured loss, a bad augmentation — before committing to a multi-hour run on a larger dataset. Learn more about supported dataset formats in the [Ultralytics segmentation dataset guide](index.md).

### Where can I find the YAML configuration file for the COCO8-Seg dataset?

The YAML configuration file for the **COCO8-Seg dataset** is available in the Ultralytics repository. You can access the file directly at <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-seg.yaml>. The YAML file includes essential information about dataset paths, classes, and configuration settings required for model training and validation.

### How does COCO8-Seg compare to COCO128-Seg and the full COCO-Seg dataset?

COCO8-Seg (8 images) sits below [COCO128-Seg](coco128-seg.md) (128 images) and the full [COCO-Seg](coco.md) dataset (118,287 training images) in terms of size:

- **COCO8-Seg**: 8 images (4 train, 4 val) — the fastest sanity check, ideal for CI and quick debugging.
- **COCO128-Seg**: 128 images — balanced between size and diversity, with train and val sharing the same directory.
- **Full COCO-Seg**: 118,287 training images — comprehensive but resource-intensive, requiring ~27 GB on first download.

Use COCO8-Seg for the fastest possible pipeline check, then scale to COCO128-Seg or the full COCO-Seg dataset as confidence grows.
