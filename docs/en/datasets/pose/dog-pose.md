---
title: Dog-Pose Estimation Dataset
comments: true
creator:
    name: Stanford Vision Lab
    url: http://vision.stanford.edu/aditya86/ImageNetDogs/
license:
    name: Research-Only
    url: https://www.image-net.org/download.php
description: "Explore the Ultralytics Dog-Pose dataset: 6,773 training and 1,703 validation images with 24 keypoints per dog, for canine pose estimation with YOLO26."
keywords: Dog-Pose, Ultralytics, pose estimation dataset, YOLO26, machine learning, computer vision, training data
---

# Dog-Pose Dataset

## Introduction

The [Ultralytics](https://www.ultralytics.com/) Dog-Pose dataset is a high-quality and extensive dataset specifically curated for dog keypoint estimation, providing 6,773 training and 1,703 validation images.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ZhjO32tZUek"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train an Ultralytics YOLO Model on the Stanford Dog Pose Estimation Dataset | Step-by-Step Tutorial
</p>

Each annotated image includes 24 keypoints with 3 dimensions per keypoint (x, y, visibility), making it a valuable resource for advanced research and development in computer vision.

<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-dogs.avif" alt="Ultralytics Dog-Pose display image" width="800">

For a specific breed or a different animal altogether, [Ultralytics Platform](https://platform.ultralytics.com/) handles uploading, labeling, and training a custom keypoint model on your own data without managing infrastructure.

## Dataset Structure

- **Total images**: 8,476 (6,773 train / 1,703 val) with matching YOLO-format label files.
- **Keypoints**: 24 per dog with `(x, y, visibility)` triplets.
- **Download size**: ~337 MB.
- **Layout**:

    ```text
    datasets/dog-pose/
    ├── images/{train,val}
    └── labels/{train,val}
    ```

## Dataset YAML

A YAML file is used to define the dataset configuration. It includes paths, keypoint details, and other relevant information. In the case of the Dog-Pose dataset, the `dog-pose.yaml` file is available at <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dog-pose.yaml>.

!!! example "ultralytics/cfg/datasets/dog-pose.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/dog-pose.yaml"
    ```

## Usage

To train a YOLO26n-pose model on the Dog-Pose dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="dog-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo pose train data=dog-pose.yaml model=yolo26n-pose.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

Here are some examples of images from the Dog-Pose dataset, along with their corresponding annotations:

<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-training-batch-2-dog-pose.avif" alt="Dog pose estimation dataset mosaic training batch" width="800">

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the Dog-Pose dataset and the benefits of using mosaicing during the training process.

## Citations and Acknowledgments

If you use the Dog-Pose dataset in your research or development work, please cite the following paper:

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

We would like to acknowledge the Stanford team for creating and maintaining this valuable resource for the [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) community. For more information about the Dog-Pose dataset and its creators, visit the [Stanford Dogs Dataset website](http://vision.stanford.edu/aditya86/ImageNetDogs/).

## FAQ

### What is the Dog-Pose dataset, and how is it used with Ultralytics YOLO26?

The Dog-Pose dataset features 6,773 training and 1,703 validation images annotated with 24 keypoints for dog pose estimation. It's designed for training and validating models with [Ultralytics YOLO26](../../models/yolo26.md), supporting applications like animal behavior analysis, pet monitoring, and veterinary studies. The dataset's comprehensive annotations make it ideal for developing accurate pose estimation models for canines.

### How do I train a YOLO26 model using the Dog-Pose dataset in Ultralytics?

Load `yolo26n-pose.pt` and call `model.train(data="dog-pose.yaml", epochs=100, imgsz=640)` — see the [Train Example](#usage) above for the full Python and CLI snippets, and the model [Training](../../modes/train.md) page for a comprehensive list of arguments.

### What are the benefits of using the Dog-Pose dataset?

With 8,476 total images (6,773 train / 1,703 val) covering a wide range of dog breeds and poses, and 24 keypoints in 3 dimensions (x, y, visibility) per annotation, the Dog-Pose dataset gives models the real-world scenario coverage needed for applications like [pet monitoring](https://www.ultralytics.com/blog/custom-training-ultralytics-yolo11-for-dog-pose-estimation) and behavior analysis. For more about its features and usage, see the [Dataset Introduction](#introduction) section.

### How does mosaicing benefit the YOLO26 training process using the Dog-Pose dataset?

Mosaicing combines multiple Dog-Pose images into a single training image, increasing the variety of dog poses, sizes, and backgrounds the model sees per step, which improves generalization to new contexts and scales while reducing overfitting. For example images, refer to the [Sample Images and Annotations](#sample-images-and-annotations) section.

### Where can I find the Dog-Pose dataset YAML file and how do I use it?

The Dog-Pose dataset YAML file can be found at <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dog-pose.yaml>. This file defines the dataset configuration, including paths, classes, keypoint details, and other relevant information. The YAML specifies 24 keypoints with 3 dimensions per keypoint, making it suitable for detailed pose estimation tasks.

To use this file with YOLO26 training scripts, simply reference it in your training command as shown in the [Usage](#usage) section. The dataset will be automatically downloaded when first used, making setup straightforward.

For more on keypoint models, see the [Pose Estimation](../../tasks/pose.md) task docs.
