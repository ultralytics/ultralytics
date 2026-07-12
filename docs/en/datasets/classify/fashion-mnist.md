---
title: Fashion-MNIST Image Classification Dataset
comments: true
description: Train YOLO image classification models on Fashion-MNIST, a benchmark of 70,000 28x28 grayscale Zalando clothing images in 10 balanced classes, split 60k/10k.
keywords: Fashion-MNIST, image classification, Zalando dataset, machine learning, deep learning, CNN, YOLO, computer vision, dataset overview
---

# Fashion-MNIST Dataset

The [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset is an [image classification](https://www.ultralytics.com/glossary/image-classification) benchmark of 70,000 28x28 grayscale images of Zalando's clothing articles, evenly split across 10 classes — T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot. It ships with a predefined split of 60,000 training and 10,000 test images (7,000 per class) and serves as a drop-in replacement for the original [MNIST](mnist.md) dataset for benchmarking [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) algorithms. For the color-image equivalent, see the related [CIFAR-10](cifar10.md) dataset.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/eX5ad6udQ9Q"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to do <a href="https://www.ultralytics.com/glossary/image-classification">Image Classification</a> on Fashion-MNIST using Ultralytics YOLO
</p>

## Key Features

- Fashion-MNIST contains 70,000 grayscale images of 28x28 pixels, evenly divided into 10 classes.
- Each class holds exactly 7,000 images — 6,000 for training and 1,000 for testing — so the dataset is perfectly balanced.
- It is a drop-in replacement for MNIST: identical image size, format, and split structure, but with harder clothing categories instead of handwritten digits.
- The dataset ships with a predefined train/test split, so no manual or automatic splitting is required.
- Fashion-MNIST is a standard benchmark for [image classification](https://www.ultralytics.com/glossary/image-classification) and [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) research.

## Dataset Structure

Fashion-MNIST ships with an official, predefined split, so no automatic or manual partitioning is needed:

- **Classes**: 10 (T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot)
- **Total images**: 70,000 (28x28 grayscale)
- **Training set**: 60,000 images (6,000 per class)
- **Test set**: 10,000 images (1,000 per class)

!!! note "Validation split"

    Fashion-MNIST has no separate validation folder, so Ultralytics uses the 10,000-image test set as the validation split during training by default.

## Applications

Fashion-MNIST is widely used to train and evaluate [image classification](https://www.ultralytics.com/glossary/image-classification) models, from classic [Convolutional Neural Networks](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) (CNNs) and [Support Vector Machines](https://www.ultralytics.com/glossary/support-vector-machine-svm) (SVMs) to modern deep architectures. Its small grayscale images and 10 clothing categories make it a fast, reproducible benchmark for algorithm comparison and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) experimentation, while being more challenging than the handwritten digits of MNIST.

## Usage

Train a YOLO model on Fashion-MNIST for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at an image size of 28. For the full list of available arguments, see the [Training](../../modes/train.md) page and the [image classification](../../tasks/classify.md) task guide.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="fashion-mnist", epochs=100, imgsz=28)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=fashion-mnist model=yolo26n-cls.pt epochs=100 imgsz=28
        ```

## Sample Images and Annotations

Sample images from the Fashion-MNIST dataset:

![Fashion-MNIST clothing classification dataset samples](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/fashion-mnist-sample.avif)

The samples show the variety of clothing categories in the Fashion-MNIST dataset, underlining the value of a varied dataset for training robust image classification models.

## Citations and Acknowledgments

If you use the Fashion-MNIST dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{xiao2017fashion,
                 title={Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
                 author={Han Xiao and Kashif Rasul and Roland Vollgraf},
                 year={2017},
                 eprint={1708.07747},
                 archivePrefix={arXiv},
                 primaryClass={cs.LG}
        }
        ```

We would like to acknowledge Zalando Research for creating and maintaining the Fashion-MNIST dataset as a valuable resource for the [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) research community. For more information about the Fashion-MNIST dataset and its creators, visit the [Fashion-MNIST GitHub repository](https://github.com/zalandoresearch/fashion-mnist).

## FAQ

### What is the Fashion-MNIST dataset and how is it different from MNIST?

The [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset is a benchmark of 70,000 28x28 grayscale images of Zalando's clothing articles across 10 classes, created as a drop-in replacement for the original [MNIST](mnist.md) dataset. It shares MNIST's exact image size, format, and 60,000/10,000 train/test split, but replaces handwritten digits with harder fashion categories — such as T-shirt/top, trouser, and ankle boot — making it a more demanding benchmark for [image classification](https://www.ultralytics.com/glossary/image-classification) models.

### How can I train an Ultralytics YOLO model on the Fashion-MNIST dataset?

To train an Ultralytics YOLO model on Fashion-MNIST, use the code snippets below. The dataset downloads automatically on first use. For a full list of arguments, see the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="fashion-mnist", epochs=100, imgsz=28)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=fashion-mnist model=yolo26n-cls.pt epochs=100 imgsz=28
        ```

### How many classes does the Fashion-MNIST dataset have?

Fashion-MNIST has 10 classes — T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot — with exactly 7,000 images each, for 70,000 images in total. Every image is a 28x28 grayscale picture of a single Zalando clothing article, and the classes are perfectly balanced.

### How is the Fashion-MNIST dataset split into training and test sets?

Fashion-MNIST ships with a predefined split of 60,000 training images and 10,000 test images, with exactly 6,000 training and 1,000 test images per class. Unlike folder-based classification datasets that Ultralytics splits automatically, Fashion-MNIST's official partition is used as-is, and the test set serves as the validation split during training by default.

### Can I use Ultralytics Platform for training models on the Fashion-MNIST dataset?

Yes. [Ultralytics Platform](https://platform.ultralytics.com) lets you manage datasets, train [image classification](../../tasks/classify.md) models, and deploy them without extensive coding. It is a convenient way to run Fashion-MNIST experiments in the cloud, and you can explore more options in our [classification datasets overview](index.md).
