---
comments: true
description: Explore Ultralytics Tiger-Pose dataset with 263 diverse images. Ideal for testing, training, and refining pose estimation algorithms.
keywords: Ultralytics, Tiger-Pose, dataset, pose estimation, YOLO11, training data, machine learning, neural networks
---

# Tiger-Pose Dataset

## Introduction

[Ultralytics](https://www.ultralytics.com/) introduces the Tiger-Pose dataset, a versatile collection designed for pose estimation tasks. This dataset comprises 263 images sourced from a [YouTube Video](https://www.youtube.com/watch?v=MIBAT6BGE6U&pp=ygUbVGlnZXIgd2Fsa2luZyByZWZlcmVuY2UubXA0), with 210 images allocated for training and 53 for validation. It serves as an excellent resource for testing and troubleshooting pose estimation algorithms.

Despite its manageable size of 210 images, the Tiger-Pose dataset offers diversity, making it suitable for assessing training pipelines, identifying potential errors, and serving as a valuable preliminary step before working with larger datasets for [pose estimation](https://docs.ultralytics.com/tasks/pose/).

This dataset is intended for use with [Ultralytics HUB](https://hub.ultralytics.com/) and [YOLO11](https://github.com/ultralytics/ultralytics).

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Gc6K5eKrTNQ"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Train YOLO11 Pose Model on Tiger-Pose Dataset Using Ultralytics HUB
</p>

## Dataset YAML

A YAML (Yet Another Markup Language) file serves as the means to specify the configuration details of a dataset. It encompasses crucial data such as file paths, class definitions, and other pertinent information. Specifically, for the `tiger-pose.yaml` file, you can check [Ultralytics Tiger-Pose Dataset Configuration File](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/tiger-pose.yaml).

!!! example "ultralytics/cfg/datasets/tiger-pose.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/tiger-pose.yaml"
    ```

## Usage

To train a YOLO11n-pose model on the Tiger-Pose dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="tiger-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo task=pose mode=train data=tiger-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

Here are some examples of images from the Tiger-Pose dataset, along with their corresponding annotations:

<img src="https://github.com/ultralytics/docs/releases/download/0/mosaiced-training-batch-4.avif" alt="Dataset sample image" width="100%">

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the Tiger-Pose dataset and the benefits of using mosaicing during the training process.

## Inference Example

!!! example "Inference Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/best.pt")  # load a tiger-pose trained model

        # Run inference
        results = model.predict(source="https://youtu.be/MIBAT6BGE6U", show=True)
        ```

    === "CLI"

        ```bash
        # Run inference using a tiger-pose trained model
        yolo task=pose mode=predict source="https://youtu.be/MIBAT6BGE6U" show=True model="path/to/best.pt"
        ```

## Citations and Acknowledgments

The dataset has been released available under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

## FAQ

### What is the Ultralytics Tiger-Pose dataset used for?

The Ultralytics Tiger-Pose dataset is designed for pose estimation tasks, consisting of 263 images sourced from a [YouTube video](https://www.youtube.com/watch?v=MIBAT6BGE6U&pp=ygUbVGlnZXIgd2Fsa2luZyByZWZlcmVuY2UubXA0). The dataset is divided into 210 training images and 53 validation images. It is particularly useful for testing, training, and refining pose estimation algorithms using [Ultralytics HUB](https://hub.ultralytics.com/) and [YOLO11](https://github.com/ultralytics/ultralytics).

### How do I train a YOLO11 model on the Tiger-Pose dataset?

To train a YOLO11n-pose model on the Tiger-Pose dataset for 100 epochs with an image size of 640, use the following code snippets. For more details, visit the [Training](../../modes/train.md) page:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="tiger-pose.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo task=pose mode=train data=tiger-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
        ```

### What configurations does the `tiger-pose.yaml` file include?

The `tiger-pose.yaml` file is used to specify the configuration details of the Tiger-Pose dataset. It includes crucial data such as file paths and class definitions. To see the exact configuration, you can check out the [Ultralytics Tiger-Pose Dataset Configuration File](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/tiger-pose.yaml).

### How can I run inference using a YOLO11 model trained on the Tiger-Pose dataset?

To perform inference using a YOLO11 model trained on the Tiger-Pose dataset, you can use the following code snippets. For a detailed guide, visit the [Prediction](../../modes/predict.md) page:

!!! example "Inference Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/best.pt")  # load a tiger-pose trained model

        # Run inference
        results = model.predict(source="https://youtu.be/MIBAT6BGE6U", show=True)
        ```


    === "CLI"

        ```bash
        # Run inference using a tiger-pose trained model
        yolo task=pose mode=predict source="https://youtu.be/MIBAT6BGE6U" show=True model="path/to/best.pt"
        ```

### What are the benefits of using the Tiger-Pose dataset for pose estimation?

The Tiger-Pose dataset, despite its manageable size of 210 images for training, provides a diverse collection of images that are ideal for testing pose estimation pipelines. The dataset helps identify potential errors and acts as a preliminary step before working with larger datasets. Additionally, the dataset supports the training and refinement of pose estimation algorithms using advanced tools like [Ultralytics HUB](https://hub.ultralytics.com/) and [YOLO11](https://github.com/ultralytics/ultralytics), enhancing model performance and [accuracy](https://www.ultralytics.com/glossary/accuracy).
