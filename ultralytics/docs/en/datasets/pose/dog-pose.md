---
comments: true
description: Discover the Dog-Pose dataset for pose detection. Featuring 6,773 training and 1,703 test images, it's a robust dataset for training YOLO11 models.
keywords: Dog-Pose, Ultralytics, pose detection dataset, YOLO11, machine learning, computer vision, training data
---

# Dog-Pose Dataset

## Introduction

The [Ultralytics](https://www.ultralytics.com/) Dog-Pose dataset is a high-quality and extensive dataset specifically curated for dog keypoint estimation. With 6,773 training images and 1,703 test images, this dataset provides a solid foundation for training robust pose estimation models.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ZhjO32tZUek"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train Ultralytics YOLO11 on the Stanford Dog Pose Estimation Dataset | Step-by-Step Tutorial🚀
</p>

Each annotated image includes 24 keypoints with 3 dimensions per keypoint (x, y, visibility), making it a valuable resource for advanced research and development in computer vision.

<img src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-dogs.avif" alt="Ultralytics Dog-pose display image" width="800">

This dataset is intended for use with Ultralytics [HUB](https://hub.ultralytics.com/) and [YOLO11](https://github.com/ultralytics/ultralytics).

## Dataset Structure

- **Split**: 6,773 train / 1,703 test images with matching YOLO-format label files.
- **Keypoints**: 24 per dog with `(x, y, visibility)` triplets.
- **Layout**:

    ```
    datasets/dog-pose/
    ├── images/{train,test}
    └── labels/{train,test}
    ```

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

The Dog-Pose dataset features 6,773 training and 1,703 test images annotated with 24 keypoints for dog pose estimation. It's designed for training and validating models with [Ultralytics YOLO11](../../models/yolo11.md), supporting applications like animal behavior analysis, pet monitoring, and veterinary studies. The dataset's comprehensive annotations make it ideal for developing accurate pose estimation models for canines.

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

**Large and Diverse Dataset**: With over 8,400 images, it provides substantial data covering a wide range of dog poses, breeds, and contexts, enabling robust model training and evaluation.

**Detailed Keypoint Annotations**: Each image includes 24 keypoints with 3 dimensions per keypoint (x, y, visibility), offering precise annotations for training accurate pose detection models.

**Real-World Scenarios**: Includes images from varied environments, enhancing the model's ability to generalize to real-world applications like [pet monitoring](https://www.ultralytics.com/blog/custom-training-ultralytics-yolo11-for-dog-pose-estimation) and behavior analysis.

**Transfer Learning Advantage**: The dataset works well with [transfer learning](https://www.ultralytics.com/blog/understanding-few-shot-zero-shot-and-transfer-learning) techniques, allowing models pre-trained on human pose datasets to adapt to dog-specific features.

For more about its features and usage, see the [Dataset Introduction](#introduction) section.

### How does mosaicing benefit the YOLO11 training process using the Dog-pose dataset?

Mosaicing, as illustrated in the sample images from the Dog-pose dataset, merges multiple images into a single composite, enriching the diversity of objects and scenes in each training batch. This technique offers several benefits:

- Increases the variety of dog poses, sizes, and backgrounds in each batch
- Improves the model's ability to detect dogs in different contexts and scales
- Enhances generalization by exposing the model to more diverse visual patterns
- Reduces overfitting by creating novel combinations of training examples

This approach leads to more robust models that perform better in real-world scenarios. For example images, refer to the [Sample Images and Annotations](#sample-images-and-annotations) section.

### Where can I find the Dog-pose dataset YAML file and how do I use it?

The Dog-pose dataset YAML file can be found at <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dog-pose.yaml>. This file defines the dataset configuration, including paths, classes, keypoint details, and other relevant information. The YAML specifies 24 keypoints with 3 dimensions per keypoint, making it suitable for detailed pose estimation tasks.

To use this file with YOLO11 training scripts, simply reference it in your training command as shown in the [Usage](#usage) section. The dataset will be automatically downloaded when first used, making setup straightforward.

For more FAQs and detailed documentation, visit the [Ultralytics Documentation](https://docs.ultralytics.com/).
