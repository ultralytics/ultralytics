---
comments: true
description: Explore the compact, versatile COCO8-Pose dataset for testing and debugging object detection models. Ideal for quick experiments with YOLO26.
keywords: COCO8-Pose, Ultralytics, pose detection dataset, object detection, YOLO26, machine learning, computer vision, training data
---

# COCO8-Pose Dataset

## Introduction

[Ultralytics](https://www.ultralytics.com/) COCO8-Pose is a small but versatile pose detection dataset composed of the first 8 images of the COCO train 2017 set, 4 for training and 4 for validation. This dataset is ideal for testing and debugging [object detection](https://www.ultralytics.com/glossary/object-detection) models, or for experimenting with new detection approaches. With 8 images, it is small enough to be easily manageable, yet diverse enough to test training pipelines for errors and act as a sanity check before training larger datasets.

## Dataset Structure

- **Total images**: 8 (4 train / 4 val).
- **Classes**: 1 (person) with 17 keypoints per annotation.
- **Recommended directory layout**: `datasets/coco8-pose/images/{train,val}` and `datasets/coco8-pose/labels/{train,val}` with YOLO-format keypoints stored as `.txt` files.

This dataset is intended for use with [Ultralytics Platform](https://platform.ultralytics.com/) and [YOLO26](https://github.com/ultralytics/ultralytics).

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the COCO8-Pose dataset, the `coco8-pose.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml).

!!! example "ultralytics/cfg/datasets/coco8-pose.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8-pose.yaml"
    ```

## Usage

To train a YOLO26n-pose model on the COCO8-Pose dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo pose train data=coco8-pose.yaml model=yolo26n-pose.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

Here are some examples of images from the COCO8-Pose dataset, along with their corresponding annotations:

<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-training-batch-5.avif" alt="COCO8-pose keypoint estimation dataset mosaic" width="800">

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the COCO8-Pose dataset and the benefits of using mosaicing during the training process.

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

### What is the COCO8-Pose dataset, and how is it used with Ultralytics YOLO26?

The COCO8-Pose dataset is a small, versatile pose detection dataset that includes the first 8 images from the COCO train 2017 set, with 4 images for training and 4 for validation. It's designed for testing and debugging object detection models and experimenting with new detection approaches. This dataset is ideal for quick experiments with [Ultralytics YOLO26](../../models/yolo26.md). For more details on dataset configuration, check out the [dataset YAML file](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml).

### How do I train a YOLO26 model using the COCO8-Pose dataset in Ultralytics?

To train a YOLO26n-pose model on the COCO8-Pose dataset for 100 epochs with an image size of 640, follow these examples:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-pose.pt")

        # Train the model
        results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo pose train data=coco8-pose.yaml model=yolo26n-pose.pt epochs=100 imgsz=640
        ```

For a comprehensive list of training arguments, refer to the model [Training](../../modes/train.md) page.

### What are the benefits of using the COCO8-Pose dataset?

The COCO8-Pose dataset offers several benefits:

- **Compact Size**: With only 8 images, it is easy to manage and perfect for quick experiments.
- **Diverse Data**: Despite its small size, it includes a variety of scenes, useful for thorough pipeline testing.
- **Error Debugging**: Ideal for identifying training errors and performing sanity checks before scaling up to larger datasets.

For more about its features and usage, see the [Dataset Introduction](#introduction) section.

### How does mosaicing benefit the YOLO26 training process using the COCO8-Pose dataset?

Mosaicing, demonstrated in the sample images of the COCO8-Pose dataset, combines multiple images into one, increasing the variety of objects and scenes within each training batch. This technique helps improve the model's ability to generalize across various object sizes, aspect ratios, and contexts, ultimately enhancing model performance. See the [Sample Images and Annotations](#sample-images-and-annotations) section for example images.

### Where can I find the COCO8-Pose dataset YAML file and how do I use it?

The COCO8-Pose dataset YAML file can be found at <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml>. This file defines the dataset configuration, including paths, classes, and other relevant information. Use this file with the YOLO26 training scripts as mentioned in the [Train Example](#how-do-i-train-a-yolo26-model-using-the-coco8-pose-dataset-in-ultralytics) section.

For more FAQs and detailed documentation, visit the [Ultralytics Documentation](https://docs.ultralytics.com/).
