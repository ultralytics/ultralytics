---
title: CIFAR-10 Image Classification Dataset
comments: true
creator:
    name: Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
    type: Person
    url: https://www.cs.toronto.edu/~kriz/cifar.html
license:
    name: None
description: Train YOLO image classification models on CIFAR-10, a benchmark of 60,000 32x32 color images in 10 balanced classes with a predefined 50k/10k train/test split.
keywords: CIFAR-10, dataset, image classification, object recognition, machine learning, computer vision, YOLO, deep learning, neural networks, AI
---

# CIFAR-10 Dataset

The [CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html) (Canadian Institute For Advanced Research) dataset is a classic [image classification](https://www.ultralytics.com/glossary/image-classification) benchmark of 60,000 32x32 color images evenly split across 10 classes — airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. It ships with a predefined split of 50,000 training and 10,000 test images (6,000 per class), making it a lightweight, well-balanced starting point for training and benchmarking classification models. For a more fine-grained challenge, see the related [CIFAR-100](cifar100.md) dataset.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/fLBbyhPbWzY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train an <a href="https://www.ultralytics.com/glossary/image-classification">Image Classification</a> Model with CIFAR-10 Dataset using Ultralytics YOLO
</p>

## Key Features

- CIFAR-10 contains 60,000 color images of 32x32 pixels, evenly divided into 10 classes.
- Each class holds exactly 6,000 images — 5,000 for training and 1,000 for testing — so the dataset is perfectly balanced.
- The 10 classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
- The dataset ships with a predefined train/test split, so no manual or automatic splitting is required.
- CIFAR-10 is a standard benchmark for [image classification](https://www.ultralytics.com/glossary/image-classification) and object recognition research.

## Dataset Structure

CIFAR-10 ships with an official, predefined split, so no automatic or manual partitioning is needed:

- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Total images**: 60,000 (32x32 color)
- **Training set**: 50,000 images (5,000 per class)
- **Test set**: 10,000 images (1,000 per class)

!!! note "Validation split"

    CIFAR-10 has no separate validation folder, so Ultralytics uses the 10,000-image test set as the validation split during training by default.

## Applications

CIFAR-10 is widely used to train and evaluate [image classification](https://www.ultralytics.com/glossary/image-classification) models, from classic [Convolutional Neural Networks](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) (CNNs) and [Support Vector Machines](https://www.ultralytics.com/glossary/support-vector-machine-svm) (SVMs) to modern deep architectures. Its small image size and balanced classes make it ideal for rapid experimentation, benchmarking new algorithms, and teaching [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) fundamentals.

## Usage

Train a YOLO model on CIFAR-10 for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at an image size of 32. For the full list of available arguments, see the [Training](../../modes/train.md) page and the [image classification](../../tasks/classify.md) task guide.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="cifar10", epochs=100, imgsz=32)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=cifar10 model=yolo26n-cls.pt epochs=100 imgsz=32
        ```

## Sample Images and Annotations

The CIFAR-10 dataset contains color images of various objects, providing a well-structured dataset for [image classification](https://www.ultralytics.com/glossary/image-classification) tasks. Here are some examples of images from the dataset:

![CIFAR-10 image classification dataset samples](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/cifar10-sample-image.avif)

The samples show the diversity of the objects in the CIFAR-10 dataset, underlining the value of a varied dataset for training robust image classification models.

## Citations and Acknowledgments

If you use the CIFAR-10 dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @TECHREPORT{Krizhevsky09learningmultiple,
                    author={Alex Krizhevsky},
                    title={Learning multiple layers of features from tiny images},
                    institution={},
                    year={2009}
        }
        ```

We would like to acknowledge Alex Krizhevsky for creating and maintaining the CIFAR-10 dataset as a valuable resource for the [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) research community. For more information about the CIFAR-10 dataset and its creator, visit the [CIFAR-10 dataset website](https://cave.cs.toronto.edu/kriz/cifar.html).

## FAQ

### What is the CIFAR-10 dataset used for in machine learning?

The [CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html) dataset is widely used to train and benchmark [image classification](https://www.ultralytics.com/glossary/image-classification) and object recognition models. It contains 60,000 32x32 color images evenly spread across 10 classes, and its small size and balanced classes make it a fast, reliable benchmark for algorithms such as Convolutional [Neural Networks](https://www.ultralytics.com/glossary/neural-network-nn) (CNNs) and Support Vector Machines (SVMs).

### How can I train an Ultralytics YOLO model on the CIFAR-10 dataset?

To train an Ultralytics YOLO model on CIFAR-10, use the code snippets below. The dataset downloads automatically on first use. For a full list of arguments, see the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="cifar10", epochs=100, imgsz=32)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=cifar10 model=yolo26n-cls.pt epochs=100 imgsz=32
        ```

### How many classes does the CIFAR-10 dataset have?

CIFAR-10 has 10 classes — airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck — with exactly 6,000 images each, for 60,000 images in total. The classes are mutually exclusive and perfectly balanced, with no overlap between categories.

### How is the CIFAR-10 dataset split into training and test sets?

CIFAR-10 ships with a predefined split of 50,000 training images and 10,000 test images, with exactly 5,000 training and 1,000 test images per class. Unlike folder-based classification datasets that Ultralytics splits automatically, CIFAR-10's official partition is used as-is, and the test set serves as the validation split during training by default.

### Can I use Ultralytics Platform for training models on the CIFAR-10 dataset?

Yes. [Ultralytics Platform](https://platform.ultralytics.com) lets you manage datasets, train [image classification](../../tasks/classify.md) models, and deploy them without extensive coding. It is a convenient way to run CIFAR-10 experiments in the cloud, and you can explore more options in our [classification datasets overview](index.md).
