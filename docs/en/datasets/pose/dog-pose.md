---
comments: true
description: Discover the Dog-Pose dataset for pose detection. Featuring 6,773 training and 1,703 test images, it's a robust dataset for training YOLO11 models.
keywords: Dog-Pose, Ultralytics, pose detection dataset, YOLO11, machine learning, computer vision, training data
---

# Dog-Pose Dataset

## Introduction

The [Ultralytics](https://www.ultralytics.com/) Dog-pose dataset is a high-quality and extensive dataset specifically curated for dog keypoint estimation. With 6,773 training images and 1,703 test images, this dataset provides a solid foundation for training robust pose estimation models. Each annotated image includes 24 keypoints with 3 dimensions per keypoint (x, y, visibility), making it a valuable resource for advanced research and development in computer vision.

<img src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-dogs.avif" alt="Ultralytics Dog-pose display image" width="800">

This dataset is intended for use with Ultralytics [HUB](https://hub.ultralytics.com/) and [YOLO11](https://github.com/ultralytics/ultralytics).

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It includes paths, keypoint details, and other relevant information. In the case of the Dog-pose dataset, The `dog-pose.yaml` is available at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dog-pose.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dog-pose.yaml).

!!! example "ultralytics/cfg/datasets/dog-pose.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/dog-pose.yaml"
    ```

## Usage

To train a YOLO11n-pose model on the Dog-pose dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="dog-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo pose train data=dog-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

Here are some examples of images from the Dog-pose dataset, along with their corresponding annotations:

<img src="https://github.com/ultralytics/docs/releases/download/0/mosaiced-training-batch-2-dog-pose.avif" alt="Dataset sample image" width="800">

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the Dog-pose dataset and the benefits of using mosaicing during the training process.

## Citations and Acknowledgments

If you use the Dog-pose dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{khosla2011fgvc,
          title={Novel dataset for Fine-Grained Image Categorization},
          author={Aditya Khosla and Nityananda Jayadevaprakash and Bangpeng Yao and Li Fei-Fei},
          booktitle={First Workshop on Fine-Grained Visual Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
          year={2011}
        }
        @inproceedings{deng2009imagenet,
          title={ImageNet: A Large-Scale Hierarchical Image Database},
          author={Jia Deng and Wei Dong and Richard Socher and Li-Jia Li and Kai Li and Li Fei-Fei},
          booktitle={IEEE Computer Vision and Pattern Recognition (CVPR)},
          year={2009}
        }
        ```

We would like to acknowledge the Stanford team for creating and maintaining this valuable resource for the [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) community. For more information about the Dog-pose dataset and its creators, visit the [Stanford Dogs Dataset website](http://vision.stanford.edu/aditya86/ImageNetDogs/).

## FAQ

### What is the Dog-pose dataset, and how is it used with Ultralytics YOLO11?

The Dog-Pose dataset features 6,000 images annotated with 17 keypoints for dog pose estimation. Ideal for training and validating models with [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), it supports applications like animal behavior analysis and veterinary studies.

### How do I train a YOLO11 model using the Dog-pose dataset in Ultralytics?

To train a YOLO11n-pose model on the Dog-pose dataset for 100 epochs with an image size of 640, follow these examples:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-pose.pt")

        # Train the model
        results = model.train(data="dog-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo pose train data=dog-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
        ```

For a comprehensive list of training arguments, refer to the model [Training](../../modes/train.md) page.

### What are the benefits of using the Dog-pose dataset?

The Dog-pose dataset offers several benefits:

**Large and Diverse Dataset**: With 6,000 images, it provides a substantial amount of data covering a wide range of dog poses, breeds, and contexts, enabling robust model training and evaluation.

**Pose-specific Annotations**: Offers detailed annotations for pose estimation, ensuring high-quality data for training pose detection models.

**Real-World Scenarios**: Includes images from varied environments, enhancing the model's ability to generalize to real-world applications.

**Model Performance Improvement**: The diversity and scale of the dataset help improve model accuracy and robustness, particularly for tasks involving fine-grained pose estimation.

For more about its features and usage, see the [Dataset Introduction](#introduction) section.

### How does mosaicing benefit the YOLO11 training process using the Dog-pose dataset?

Mosaicing, as illustrated in the sample images from the Dog-pose dataset, merges multiple images into a single composite, enriching the diversity of objects and scenes in each training batch. This approach enhances the model's capacity to generalize across different object sizes, aspect ratios, and contexts, leading to improved performance. For example images, refer to the [Sample Images and Annotations](#sample-images-and-annotations) section.

### Where can I find the Dog-pose dataset YAML file and how do I use it?

The Dog-pose dataset YAML file can be found [here](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dog-pose.yaml). This file defines the dataset configuration, including paths, classes, and other relevant information. Use this file with the YOLO11 training scripts as mentioned in the [Train Example](#how-do-i-train-a-yolo11-model-using-the-dog-pose-dataset-in-ultralytics) section.

For more FAQs and detailed documentation, visit the [Ultralytics Documentation](https://docs.ultralytics.com/).
