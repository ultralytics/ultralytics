---
title: Tiger-Pose Estimation Dataset
comments: true
description: "Explore the Ultralytics Tiger-Pose dataset: 263 images (210 train / 53 val) with 12 keypoints per tiger, ideal for testing pose estimation pipelines."
keywords: Ultralytics, Tiger-Pose, dataset, pose estimation, YOLO26, training data, machine learning, neural networks
---

# Tiger-Pose Dataset

## Introduction

[Ultralytics](https://www.ultralytics.com/) introduces the Tiger-Pose dataset, a versatile collection designed for pose estimation tasks. This dataset comprises 263 images sourced from a [YouTube video](https://www.youtube.com/watch?v=MIBAT6BGE6U), with 210 images allocated for training and 53 for validation. It serves as an excellent resource for testing and troubleshooting pose estimation algorithms.

Despite its manageable training split of 210 images, the Tiger-Pose dataset offers diversity, making it suitable for assessing training pipelines, identifying potential errors, and serving as a valuable preliminary step before working with larger datasets for [pose estimation](../../tasks/pose.md).

Once your pipeline trains clean on this small set, swap in your own animal or object keypoints and scale up training on [Ultralytics Platform](https://platform.ultralytics.com/) without leaving the browser.

## Dataset Structure

- **Total images**: 263 (210 train / 53 val).
- **Keypoints**: 12 per tiger (no visibility flag).
- **Download size**: ~49.8 MB.
- **Directory layout**: YOLO-format keypoints stored under `labels/{train,val}` alongside `images/{train,val}` directories.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Gc6K5eKrTNQ"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Train an Ultralytics YOLO Pose Model on the Tiger-Pose Dataset
</p>

## Dataset YAML

A YAML file serves as the means to specify the configuration details of a dataset. It encompasses crucial data such as file paths, class definitions, and other pertinent information. Specifically, for the `tiger-pose.yaml` file, you can check [Ultralytics Tiger-Pose Dataset Configuration File](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/tiger-pose.yaml).

!!! example "ultralytics/cfg/datasets/tiger-pose.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/tiger-pose.yaml"
    ```

## Usage

To train a YOLO26n-pose model on the Tiger-Pose dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="tiger-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo pose train data=tiger-pose.yaml model=yolo26n-pose.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

Here are some examples of images from the Tiger-Pose dataset, along with their corresponding annotations:

<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-training-batch-4.avif" alt="Tiger pose estimation dataset mosaic training batch" width="100%">

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the Tiger-Pose dataset and the benefits of using mosaicing during the training process.

## Inference Example

After training, load your best checkpoint and run inference on new images or video — see the [Prediction](../../modes/predict.md) page for the full list of arguments.

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
        yolo pose predict source="https://youtu.be/MIBAT6BGE6U" show=True model="path/to/best.pt"
        ```

## Citations and Acknowledgments

The dataset has been released under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

## FAQ

### What is the Ultralytics Tiger-Pose dataset used for?

The Ultralytics Tiger-Pose dataset is designed for pose estimation tasks, consisting of 263 images sourced from a [YouTube video](https://www.youtube.com/watch?v=MIBAT6BGE6U). The dataset is divided into 210 training images and 53 validation images, making it well-suited for testing, training, and refining pose estimation algorithms.

### How do I train a YOLO26 model on the Tiger-Pose dataset?

Load `yolo26n-pose.pt` and call `model.train(data="tiger-pose.yaml", epochs=100, imgsz=640)` — see the [Train Example](#usage) above for the full Python and CLI snippets, and the [Training](../../modes/train.md) page for a comprehensive list of arguments.

### What configurations does the `tiger-pose.yaml` file include?

The `tiger-pose.yaml` file defines the dataset path, train/val image directories, a single class (`tiger`), and `kpt_shape: [12, 2]` — 12 keypoints per instance with no visibility flag. See the [Ultralytics Tiger-Pose Dataset Configuration File](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/tiger-pose.yaml) for the exact configuration.

### How can I run inference using a YOLO26 model trained on the Tiger-Pose dataset?

Load your trained checkpoint (e.g., `path/to/best.pt`) and call `model.predict(source=..., show=True)` — see the [Inference Example](#inference-example) above for the full Python and CLI snippets, and the [Prediction](../../modes/predict.md) page for a comprehensive list of arguments.

### What are the benefits of using the Tiger-Pose dataset for pose estimation?

With 263 total images (210 train / 53 val), 1 class, 12 keypoints per instance, and a ~49.8 MB download, Tiger-Pose is small enough to manage quickly yet diverse enough to sanity-check a pose training pipeline and identify errors before working with larger datasets.
