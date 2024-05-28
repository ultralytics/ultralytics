---
comments: true
description: Discover the benefits of using the practical and diverse human8 dataset for object detection model testing. Learn to configure and use it via Ultralytics HUB and YOLOv8.
keywords: Ultralytics, human8 dataset, object detection, model testing, dataset configuration, detection approaches, sanity check, training pipelines, YOLOv8
---

# COCO8-Human Dataset

## Introduction

[Ultralytics](https://ultralytics.com) COCO8-Human is a small, but versatile object detection dataset composed of 8 images from the COCO train 2017 set, annotated with person weight (kg), height (cm), gender (0: female, 1: male), age (years), and race (0: asian, 1: white, 2: middle eastern, 3: indian, 4: latino, 5: black). This dataset is ideal for testing and debugging object detection models, or for experimenting with new detection approaches. With 8 images, it is small enough to be easily manageable, yet diverse enough to test training pipelines for errors and act as a sanity check before training larger datasets.

This dataset is intended for use with Ultralytics [HUB](https://hub.ultralytics.com) and [YOLOv8](https://github.com/ultralytics/ultralytics).

## Dataset YAML

A YAML file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the COCO8-Human dataset, the `human8.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/human8.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/human8.yaml).

!!! Example "ultralytics/cfg/datasets/human8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/human8.yaml"
    ```

## Usage

To train a YOLOv8n model on the COCO8-Human dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! Example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="human8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=human8.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

