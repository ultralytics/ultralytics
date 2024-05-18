---
comments: true
description: Discover the versatile Tiger-Pose dataset, perfect for testing and debugging pose detection models. Learn how to get started with YOLOv8-pose model training.
keywords: Ultralytics, YOLOv8, pose detection, COCO8-Pose dataset, dataset, model training, YAML
---

# Tiger-Pose Dataset

## Introduction

[Ultralytics](https://ultralytics.com) introduces the Tiger-Pose dataset, a versatile collection designed for pose estimation tasks. This dataset comprises 263 images sourced from a [YouTube Video](https://www.youtube.com/watch?v=MIBAT6BGE6U&pp=ygUbVGlnZXIgd2Fsa2luZyByZWZlcmVuY2UubXA0), with 210 images allocated for training and 53 for validation. It serves as an excellent resource for testing and troubleshooting pose estimation algorithm.

Despite its manageable size of 210 images, tiger-pose dataset offers diversity, making it suitable for assessing training pipelines, identifying potential errors, and serving as a valuable preliminary step before working with larger datasets for pose estimation.

This dataset is intended for use with [Ultralytics HUB](https://hub.ultralytics.com) and [YOLOv8](https://github.com/ultralytics/ultralytics).

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Gc6K5eKrTNQ"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Train YOLOv8 Pose Model on Tiger-Pose Dataset Using Ultralytics HUB
</p>

## Dataset YAML

A YAML (Yet Another Markup Language) file serves as the means to specify the configuration details of a dataset. It encompasses crucial data such as file paths, class definitions, and other pertinent information. Specifically, for the `tiger-pose.yaml` file, you can check [Ultralytics Tiger-Pose Dataset Configuration File](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/tiger-pose.yaml).

!!! Example "ultralytics/cfg/datasets/tiger-pose.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/tiger-pose.yaml"
    ```

## Usage

To train a YOLOv8n-pose model on the Tiger-Pose dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! Example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="tiger-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo task=pose mode=train data=tiger-pose.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

Here are some examples of images from the Tiger-Pose dataset, along with their corresponding annotations:

<img src="https://user-images.githubusercontent.com/62513924/272491921-c963d2bf-505f-4a15-abd7-259de302cffa.jpg" alt="Dataset sample image" width="100%">

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the Tiger-Pose dataset and the benefits of using mosaicing during the training process.

## Inference Example

!!! Example "Inference Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('path/to/best.pt')  # load a tiger-pose trained model

        # Run inference
        results = model.predict(source="https://www.youtube.com/watch?v=MIBAT6BGE6U&pp=ygUYdGlnZXIgd2Fsa2luZyByZWZlcmVuY2Ug" show=True)
        ```

    === "CLI"

        ```bash
        # Run inference using a tiger-pose trained model
        yolo task=pose mode=predict source="https://www.youtube.com/watch?v=MIBAT6BGE6U&pp=ygUYdGlnZXIgd2Fsa2luZyByZWZlcmVuY2Ug" show=True model="path/to/best.pt"
        ```

## Citations and Acknowledgments

The dataset has been released available under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
