---
comments: true
description: Explore the Ultralytics COCO128 dataset, a versatile and manageable set of 128 images perfect for testing object detection models and training pipelines.
keywords: COCO128, Ultralytics, dataset, object detection, YOLO26, training, validation, machine learning, computer vision
---

# COCO128 Dataset

## Introduction

[Ultralytics](https://www.ultralytics.com/) COCO128 is a small, but versatile [object detection](https://www.ultralytics.com/glossary/object-detection) dataset composed of the first 128 images of the COCO train 2017 set. This dataset is ideal for testing and debugging object detection models, or for experimenting with new detection approaches. With 128 images, it is small enough to be easily manageable, yet diverse enough to test training pipelines for errors and act as a sanity check before training larger datasets.

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

This dataset is intended for use with [Ultralytics Platform](https://platform.ultralytics.com/) and [YOLO26](https://github.com/ultralytics/ultralytics).

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the COCO128 dataset, the `coco128.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml).

!!! example "ultralytics/cfg/datasets/coco128.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco128.yaml"
    ```

## Usage

To train a YOLO26n model on the COCO128 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco128.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=coco128.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

Here are some examples of images from the COCO128 dataset, along with their corresponding annotations:

<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-training-batch-1.avif" alt="COCO128 object detection dataset mosaic training batch" width="800">

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the COCO128 dataset and the benefits of using mosaicing during the training process.

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

### What is the Ultralytics COCO128 dataset used for?

The Ultralytics COCO128 dataset is a compact subset containing the first 128 images from the COCO train 2017 dataset. It's primarily used for testing and debugging [object detection](https://www.ultralytics.com/glossary/object-detection) models, experimenting with new detection approaches, and validating training pipelines before scaling to larger datasets. Its manageable size makes it perfect for quick iterations while still providing enough diversity to be a meaningful test case.

### How do I train a YOLO26 model using the COCO128 dataset?

To train a YOLO26 model on the COCO128 dataset, you can use either Python or CLI commands. Here's how:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n.pt")

        # Train the model
        results = model.train(data="coco128.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        yolo detect train data=coco128.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

For more training options and parameters, refer to the [Training](../../modes/train.md) documentation.

### What are the benefits of using mosaic augmentation with COCO128?

Mosaic augmentation, as shown in the sample images, combines multiple training images into a single composite image. This technique offers several benefits when training with COCO128:

- Increases the variety of objects and contexts within each training batch
- Improves model generalization across different object sizes and aspect ratios
- Enhances detection performance for objects at various scales
- Maximizes the utility of a small dataset by creating more diverse training samples

This technique is particularly valuable for smaller datasets like COCO128, helping models learn more robust features from limited data.

### How does COCO128 compare to other COCO dataset variants?

COCO128 (128 images) sits between [COCO8](../detect/coco8.md) (8 images) and the full [COCO](../detect/coco.md) dataset (118K+ images) in terms of size:

- **COCO8**: Contains just 8 images (4 train, 4 val) - ideal for quick tests and debugging
- **COCO128**: Contains 128 images - balanced between size and diversity
- **Full COCO**: Contains 118K+ training images - comprehensive but resource-intensive

COCO128 provides a good middle ground, offering more diversity than COCO8 while remaining much more manageable than the full COCO dataset for experimentation and initial model development.

### Can I use COCO128 for tasks other than object detection?

While COCO128 is primarily designed for object detection, the dataset's annotations can be adapted for other computer vision tasks:

- **Instance segmentation**: Using the segmentation masks provided in the annotations
- **Keypoint detection**: For images containing people with keypoint annotations
- **Transfer learning**: As a starting point for fine-tuning models for custom tasks

For specialized tasks like [segmentation](../../tasks/segment.md), consider using purpose-built variants like [COCO8-seg](../segment/coco8-seg.md) which include the appropriate annotations.
