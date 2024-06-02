---
comments: true
description: Discover the benefits of using the practical and diverse human8 dataset for object detection model testing. Learn to configure and use it via Ultralytics HUB and YOLOv8.
keywords: Ultralytics, human8 dataset, object detection, model testing, dataset configuration, detection approaches, sanity check, training pipelines, YOLOv8
---

# COCO8-Human Dataset

## Introduction

[Ultralytics](https://ultralytics.com) COCO8-Human is a small, but versatile object detection dataset composed of 8 images from the COCO train 2017 set, annotated with person weight (kg), height (cm), biological gender (0: female, 1: male), age (years), and ethnicity (0: asian, 1: white, 2: middle eastern, 3: indian, 4: latino, 5: black). The dataset has been artificially generated using a vision-language pipeline and annotated with the GPT-4 Turbo model (version 2024-04-09) deployed on Microsoft Azure. This dataset is ideal for testing and debugging object detection models, or for experimenting with new detection approaches. With 8 images, it is small enough to be easily manageable, yet diverse enough to test training pipelines for errors and act as a sanity check before training larger datasets.

 This dataset is intended for use with the [Ultralytics python library](https://github.com/ultralytics/ultralytics) and [Ultralytics HUB](https://hub.ultralytics.com).

## Dataset YAML

A YAML file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the COCO8-Human dataset, the `human8.yaml` file is maintained at [`ultralytics/cfg/datasets/human8.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/human8.yaml).

!!! Example "ultralytics/cfg/datasets/coco8-human.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8-human.yaml"
    ```

## Usage

To train a YOLOv8n-human model on the COCO8-human dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [train mode settings](../../modes/train.md#train-settings).

!!! Example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-human.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco8-human.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo human train data=coco8-human.yaml model=yolov8n-human.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

Here are some examples of images from the COCO8-human dataset, along with their corresponding detection annotations. Each human is also annotated with: weight (kg), height (cm), biological gender (0: female, 1: male), age (years), and ethnicity (0: asian, 1: white, 2: middle eastern, 3: indian, 4: latino, 5: black):

<img src="https://github.com/ultralytics/ultralytics/assets/3855193/dc3cbd2e-28e8-4459-98b2-659e52792e51" alt="Dataset sample image" width="800">

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the COCO8-human dataset and the benefits of using mosaicing during the training process.
