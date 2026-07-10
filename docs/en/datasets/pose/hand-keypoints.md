---
title: Hand Keypoints Pose Estimation Dataset
comments: true
description: Explore the Ultralytics Hand Keypoints dataset: 26,768 hand images with 21 keypoints each, for gesture recognition and pose estimation with YOLO26.
keywords: Hand KeyPoints, pose estimation, dataset, keypoints, MediaPipe, YOLO, deep learning, computer vision
---

# Hand Keypoints Dataset

## Introduction

The [Ultralytics](https://www.ultralytics.com/) Hand Keypoints dataset contains 26,768 images of hands annotated with 21 keypoints each, generated using the [Google MediaPipe](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) library for high [accuracy](https://www.ultralytics.com/glossary/accuracy) and consistency. It's compatible with [Ultralytics YOLO26](../../models/yolo26.md) formats for training pose estimation models.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/fd6u1TW_AGY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Hand Keypoints Estimation with Ultralytics YOLO | Human Hand Pose Estimation Tutorial
</p>

## Keypoints

![Hand keypoints landmark diagram with 21 points](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/hand_landmarks.jpg)

Each hand is annotated with 21 keypoints as follows:

1. Wrist
2. Thumb (4 points)
3. Index finger (4 points)
4. Middle finger (4 points)
5. Ring finger (4 points)
6. Little finger (4 points)

## Dataset Structure

- **Total images**: 26,768 (18,776 train / 7,992 val).
- **Classes**: 1 (hand).
- **Keypoints**: 21 per hand with `(x, y, visibility)` triplets.
- **Download size**: ~369 MB.

For a custom gesture vocabulary beyond generic hand landmarks, [Ultralytics Platform](https://platform.ultralytics.com/) handles labeling and training your own dataset from the browser.

## Applications

Hand keypoints support several real-world applications:

- **[Gesture recognition](https://www.ultralytics.com/blog/enhancing-hand-keypoints-estimation-with-ultralytics-yolo11)**: human-computer interaction and touchless control interfaces.
- **[AR/VR controls](../../tasks/pose.md)**: precise interaction with virtual objects.
- **Robotic manipulation**: fine-grained control of robotic hands.
- **Healthcare**: hand movement analysis for medical diagnostics.
- **Animation**: motion capture for realistic hand movement.
- **Biometric authentication**: security systems based on hand geometry.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the Hand Keypoints dataset, the `hand-keypoints.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/hand-keypoints.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/hand-keypoints.yaml).

!!! example "ultralytics/cfg/datasets/hand-keypoints.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/hand-keypoints.yaml"
    ```

## Usage

To train a YOLO26n-pose model on the Hand Keypoints dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="hand-keypoints.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo pose train data=hand-keypoints.yaml model=yolo26n-pose.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

The Hand Keypoints dataset contains a diverse set of images with human hands annotated with keypoints. Here are some examples of images from the dataset, along with their corresponding annotations:

![Hand keypoints pose estimation dataset sample](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/human-hand-pose.avif)

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the Hand Keypoints dataset and the benefits of using mosaicing during the training process.

## Citations and Acknowledgments

If you use the Hand Keypoints dataset in your research or development work, please acknowledge the following sources:

!!! quote ""

    === "Credits"

    We would like to thank the following sources for providing the images used in this dataset:

    - [11k Hands](https://sites.google.com/view/11khands)
    - [2000 Hand Gestures](https://www.kaggle.com/datasets/ritikagiridhar/2000-hand-gestures)
    - [Gesture Recognition](https://www.kaggle.com/datasets/imsparsh/gesture-recognition)

    The images were collected and used under the respective licenses provided by each platform and are distributed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

We would also like to acknowledge the creator of this dataset, [Rion Dsilva](https://www.linkedin.com/in/rion-dsilva-043464229/), for his great contribution to Vision AI research.

## FAQ

### How do I train a YOLO26 model on the Hand Keypoints dataset?

Load `yolo26n-pose.pt` and call `model.train(data="hand-keypoints.yaml", epochs=100, imgsz=640)` — see the [Train Example](#usage) above for the full Python and CLI snippets, and the model [Training](../../modes/train.md) page for a comprehensive list of arguments.

### What are the benefits of using the Hand Keypoints dataset?

With 26,768 annotated images and 21 keypoints per hand generated via Google MediaPipe, the Hand Keypoints dataset gives pose estimation models the scale and annotation accuracy needed for [advanced pose estimation](../../tasks/pose.md) tasks. See the [Keypoints](#keypoints) section for the full landmark breakdown.

### What applications can benefit from using the Hand Keypoints dataset?

Hand Keypoints supports gesture recognition, AR/VR controls, robotic manipulation, healthcare movement analysis, animation, and biometric authentication — see the [Applications](#applications) section for details on each.

### How is the Hand Keypoints dataset structured?

The Hand Keypoints dataset is divided into two subsets:

1. **Train**: Contains 18,776 images for training pose estimation models.
2. **Val**: Contains 7,992 images for validation purposes during model training.

This structure ensures a comprehensive training and validation process. For more details, see the [Dataset Structure](#dataset-structure) section.

### How do I use the dataset YAML file for training?

The dataset configuration is defined in a YAML file, which includes paths, classes, and other relevant information. The `hand-keypoints.yaml` file can be found at [hand-keypoints.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/hand-keypoints.yaml).

To use this YAML file for training, specify it in your training script or CLI command as shown in the training example above. For more details, refer to the [Dataset YAML](#dataset-yaml) section.
