---
comments: true
description: Explore the hand keypoints estimation dataset for advanced pose estimation. Learn about datasets, pretrained models, metrics, and applications for training with YOLO.
keywords: Hand KeyPoints, pose estimation, dataset, keypoints, MediaPipe, YOLO, deep learning, computer vision
---

# Hand Keypoints Dataset

## Introduction

The hand-keypoints dataset contains 26,768 images of hands annotated with keypoints, making it suitable for training models like Ultralytics YOLO for pose estimation tasks. The annotations were generated using the Google MediaPipe library, ensuring high [accuracy](https://www.ultralytics.com/glossary/accuracy) and consistency, and the dataset is compatible [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) formats.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/fd6u1TW_AGY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Hand Keypoints Estimation with Ultralytics YOLO11 | Human Hand Pose Estimation Tutorial
</p>

## Hand Landmarks

![Hand Landmarks](https://github.com/ultralytics/docs/releases/download/0/hand_landmarks.jpg)

## KeyPoints

The dataset includes keypoints for hand detection. The keypoints are annotated as follows:

1. Wrist
2. Thumb (4 points)
3. Index finger (4 points)
4. Middle finger (4 points)
5. Ring finger (4 points)
6. Little finger (4 points)

Each hand has a total of 21 keypoints.

## Key Features

- **Large Dataset**: 26,768 images with hand keypoint annotations.
- **YOLO11 Compatibility**: Ready for use with YOLO11 models.
- **21 Keypoints**: Detailed hand pose representation.

## Dataset Structure

The hand keypoint dataset is split into two subsets:

1. **Train**: This subset contains 18,776 images from the hand keypoints dataset, annotated for training pose estimation models.
2. **Val**: This subset contains 7992 images that can be used for validation purposes during model training.

## Applications

Hand keypoints can be used for gesture recognition, AR/VR controls, robotic manipulation, and hand movement analysis in healthcare. They can be also applied in animation for motion capture and biometric authentication systems for security.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the Hand Keypoints dataset, the `hand-keypoints.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/hand-keypoints.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/hand-keypoints.yaml).

!!! example "ultralytics/cfg/datasets/hand-keypoints.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/hand-keypoints.yaml"
    ```

## Usage

To train a YOLO11n-pose model on the Hand Keypoints dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="hand-keypoints.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo pose train data=hand-keypoints.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

The Hand keypoints dataset contains a diverse set of images with human hands annotated with keypoints. Here are some examples of images from the dataset, along with their corresponding annotations:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/human-hand-pose.jpg)

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the Hand Keypoints dataset and the benefits of using mosaicing during the training process.

## Citations and Acknowledgments

If you use the hand-keypoints dataset in your research or development work, please acknowledge the following sources:

!!! quote ""

    === "Credits"

    We would like to thank the following sources for providing the images used in this dataset:

    - [11k Hands](https://sites.google.com/view/11khands)
    - [2000 Hand Gestures](https://www.kaggle.com/datasets/ritikagiridhar/2000-hand-gestures)
    - [Gesture Recognition](https://www.kaggle.com/datasets/imsparsh/gesture-recognition)

    The images were collected and used under the respective licenses provided by each platform and are distributed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

We would also like to acknowledge the creator of this dataset, [Rion Dsilva](https://www.linkedin.com/in/rion-dsilva-043464229/), for his great contribution to Vision AI research.

## FAQ

### How do I train a YOLO11 model on the Hand Keypoints dataset?

To train a YOLO11 model on the Hand Keypoints dataset, you can use either Python or the command line interface (CLI). Here's an example for training a YOLO11n-pose model for 100 epochs with an image size of 640:

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="hand-keypoints.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo pose train data=hand-keypoints.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
        ```

For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

### What are the key features of the Hand Keypoints dataset?

The Hand Keypoints dataset is designed for advanced pose estimation tasks and includes several key features:

- **Large Dataset**: Contains 26,768 images with hand keypoint annotations.
- **YOLO11 Compatibility**: Ready for use with YOLO11 models.
- **21 Keypoints**: Detailed hand pose representation, including wrist and finger joints.

For more details, you can explore the [Hand Keypoints Dataset](#introduction) section.

### What applications can benefit from using the Hand Keypoints dataset?

The Hand Keypoints dataset can be applied in various fields, including:

- **Gesture Recognition**: Enhancing human-computer interaction.
- **AR/VR Controls**: Improving user experience in augmented and virtual reality.
- **Robotic Manipulation**: Enabling precise control of robotic hands.
- **Healthcare**: Analyzing hand movements for medical diagnostics.
- **Animation**: Capturing motion for realistic animations.
- **Biometric Authentication**: Enhancing security systems.

For more information, refer to the [Applications](#applications) section.

### How is the Hand Keypoints dataset structured?

The Hand Keypoints dataset is divided into two subsets:

1. **Train**: Contains 18,776 images for training pose estimation models.
2. **Val**: Contains 7,992 images for validation purposes during model training.

This structure ensures a comprehensive training and validation process. For more details, see the [Dataset Structure](#dataset-structure) section.

### How do I use the dataset YAML file for training?

The dataset configuration is defined in a YAML file, which includes paths, classes, and other relevant information. The `hand-keypoints.yaml` file can be found at [hand-keypoints.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/hand-keypoints.yaml).

To use this YAML file for training, specify it in your training script or CLI command as shown in the training example above. For more details, refer to the [Dataset YAML](#dataset-yaml) section.
