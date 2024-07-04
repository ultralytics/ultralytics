---
comments: true
description: Explore the Ultralytics COCO8 dataset, a versatile and manageable set of 8 images perfect for testing object detection models and training pipelines.
keywords: COCO8, Ultralytics, dataset, object detection, YOLOv8, training, validation, machine learning, computer vision
---

# COCO8 Dataset

## Introduction

[Ultralytics](https://ultralytics.com) COCO8 is a small, but versatile object detection dataset composed of the first 8 images of the COCO train 2017 set, 4 for training and 4 for validation. This dataset is ideal for testing and debugging object detection models, or for experimenting with new detection approaches. With 8 images, it is small enough to be easily manageable, yet diverse enough to test training pipelines for errors and act as a sanity check before training larger datasets.

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

This dataset is intended for use with Ultralytics [HUB](https://hub.ultralytics.com) and [YOLOv8](https://github.com/ultralytics/ultralytics).

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the COCO8 dataset, the `coco8.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml).

!!! Example "ultralytics/cfg/datasets/coco8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8.yaml"
    ```

## Usage

To train a YOLOv8n model on the COCO8 dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! Example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=coco8.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

Here are some examples of images from the COCO8 dataset, along with their corresponding annotations:

<img src="https://user-images.githubusercontent.com/26833433/236818348-e6260a3d-0454-436b-83a9-de366ba07235.jpg" alt="Dataset sample image" width="800">

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the COCO8 dataset and the benefits of using mosaicing during the training process.

## Citations and Acknowledgments

If you use the COCO dataset in your research or development work, please cite the following paper:

!!! Quote ""

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

We would like to acknowledge the COCO Consortium for creating and maintaining this valuable resource for the computer vision community. For more information about the COCO dataset and its creators, visit the [COCO dataset website](https://cocodataset.org/#home).

## FAQ

### What is the Ultralytics COCO8 dataset used for?

The Ultralytics COCO8 dataset is a compact yet versatile object detection dataset consisting of the first 8 images from the COCO train 2017 set, with 4 images for training and 4 for validation. It is designed for testing and debugging object detection models and experimentation with new detection approaches. Despite its small size, COCO8 offers enough diversity to act as a sanity check for your training pipelines before deploying larger datasets. For more details, view the [COCO8 dataset](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml).

### How do I train a YOLOv8 model using the COCO8 dataset?

To train a YOLOv8 model using the COCO8 dataset, you can employ either Python or CLI commands. Here's how you can start:

!!! Example "Train Example"

    === "Python"
    
        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=coco8.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

### Why should I use Ultralytics HUB for managing my COCO8 training?

Ultralytics HUB is an all-in-one web tool designed to simplify the training and deployment of YOLO models, including the Ultralytics YOLOv8 models on the COCO8 dataset. It offers cloud training, real-time tracking, and seamless dataset management. HUB allows you to start training with a single click and avoids the complexities of manual setups. Discover more about [Ultralytics HUB](https://hub.ultralytics.com) and its benefits.

### What are the benefits of using mosaic augmentation in training with the COCO8 dataset?

Mosaic augmentation, demonstrated in the COCO8 dataset, combines multiple images into a single image during training. This technique increases the variety of objects and scenes in each training batch, improving the model's ability to generalize across different object sizes, aspect ratios, and contexts. This results in a more robust object detection model. For more details, refer to the [training guide](#usage).

### How can I validate my YOLOv8 model trained on the COCO8 dataset?

Validation of your YOLOv8 model trained on the COCO8 dataset can be performed using the model's validation commands. You can invoke the validation mode via CLI or Python script to evaluate the model's performance using precise metrics. For detailed instructions, visit the [Validation](../../modes/val.md) page.
