---
title: CIFAR-100 Image Classification Dataset
comments: true
creator:
    name: Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
    type: Person
    url: https://www.cs.toronto.edu/~kriz/cifar.html
license:
    name: None
    url: https://www.cs.toronto.edu/~kriz/cifar.html
description: Train YOLO image classification models on CIFAR-100, a benchmark of 60,000 32x32 color images in 100 classes grouped into 20 superclasses, split 50k/10k.
keywords: CIFAR-100, dataset, image classification, object recognition, machine learning, computer vision, YOLO, deep learning, neural networks, AI
---

# CIFAR-100 Dataset

The [CIFAR-100](https://cave.cs.toronto.edu/kriz/cifar.html) (Canadian Institute For Advanced Research) dataset is an [image classification](https://www.ultralytics.com/glossary/image-classification) benchmark of 60,000 32x32 color images spread evenly across 100 fine-grained classes (600 images each), which are in turn grouped into 20 coarse superclasses. Created by Alex Krizhevsky, it ships with a predefined split of 50,000 training and 10,000 test images, making it the harder, more fine-grained sibling of the [CIFAR-10](cifar10.md) dataset.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/6bZeCs0xwO4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train an <a href="https://www.ultralytics.com/glossary/image-classification">Image Classification</a> Model on CIFAR-100 using Ultralytics YOLO
</p>

## Key Features

- CIFAR-100 contains 60,000 color images of 32x32 pixels, evenly divided into 100 classes.
- Each class holds exactly 600 images — 500 for training and 100 for testing — so the dataset is perfectly balanced.
- The 100 fine-grained classes are grouped into 20 coarse superclasses for higher-level classification.
- The dataset ships with a predefined train/test split, so no manual or automatic splitting is required.
- CIFAR-100 is a standard benchmark for fine-grained [image classification](https://www.ultralytics.com/glossary/image-classification) and object recognition research.

## Dataset Structure

CIFAR-100 ships with an official, predefined split, so no automatic or manual partitioning is needed:

- **Classes**: 100 fine-grained classes, grouped into 20 coarse superclasses
- **Total images**: 60,000 (32x32 color)
- **Training set**: 50,000 images (500 per class)
- **Test set**: 10,000 images (100 per class)

!!! note "Validation split"

    CIFAR-100 has no separate validation folder, so Ultralytics uses the 10,000-image test set as the validation split during training by default. Training with `data="cifar100"` learns the 100 fine-grained classes.

## Applications

CIFAR-100 is widely used to train and evaluate [image classification](https://www.ultralytics.com/glossary/image-classification) models, from classic [Convolutional Neural Networks](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) (CNNs) and [Support Vector Machines](https://www.ultralytics.com/glossary/support-vector-machine-svm) (SVMs) to modern deep architectures. Its 100 fine-grained classes and small image size make it a demanding benchmark for [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) research, algorithm comparison, and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) experimentation.

## Usage

Train a YOLO model on CIFAR-100 for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at an image size of 32. For the full list of available arguments, see the [Training](../../modes/train.md) page and the [image classification](../../tasks/classify.md) task guide.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="cifar100", epochs=100, imgsz=32)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=cifar100 model=yolo26n-cls.pt epochs=100 imgsz=32
        ```

## Sample Images and Annotations

Sample images from the CIFAR-100 dataset:

![CIFAR-100 image classification dataset samples](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/cifar100-sample-image.avif)

The samples show the diversity of the objects in the CIFAR-100 dataset, underlining the value of a varied dataset for training robust image classification models.

## Citations and Acknowledgments

If you use the CIFAR-100 dataset in your research or development work, please cite the following paper:

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

We would like to acknowledge Alex Krizhevsky for creating and maintaining the CIFAR-100 dataset as a valuable resource for the [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) research community. For more information about the CIFAR-100 dataset and its creator, visit the [CIFAR-100 dataset website](https://cave.cs.toronto.edu/kriz/cifar.html).

## FAQ

### What is the CIFAR-100 dataset used for in machine learning?

The [CIFAR-100](https://cave.cs.toronto.edu/kriz/cifar.html) dataset is widely used to train and benchmark fine-grained [image classification](https://www.ultralytics.com/glossary/image-classification) and object recognition models. It contains 60,000 32x32 color images across 100 classes grouped into 20 superclasses, making it a more challenging benchmark than CIFAR-10 for algorithms such as Convolutional [Neural Networks](https://www.ultralytics.com/glossary/neural-network-nn) (CNNs) and Support Vector Machines (SVMs), and for evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models built with [Ultralytics YOLO](../../models/yolo26.md).

### How can I train an Ultralytics YOLO model on the CIFAR-100 dataset?

To train an Ultralytics YOLO model on CIFAR-100, use the code snippets below. The dataset downloads automatically on first use. For a full list of arguments, see the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="cifar100", epochs=100, imgsz=32)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=cifar100 model=yolo26n-cls.pt epochs=100 imgsz=32
        ```

### How many classes does the CIFAR-100 dataset have?

CIFAR-100 has 100 fine-grained classes — such as apple, dolphin, maple tree, motorcycle, and rocket — with exactly 600 images each, for 60,000 images in total. These 100 classes are additionally grouped into 20 coarse superclasses (for example, the trees superclass covers maple, oak, palm, pine, and willow). Training with `data="cifar100"` uses the 100 fine-grained classes.

### How is the CIFAR-100 dataset split into training and test sets?

CIFAR-100 ships with a predefined split of 50,000 training images and 10,000 test images, with exactly 500 training and 100 test images per class. Unlike folder-based classification datasets that Ultralytics splits automatically, CIFAR-100's official partition is used as-is, and the test set serves as the validation split during training by default.

### Can I use Ultralytics Platform for training models on the CIFAR-100 dataset?

Yes. [Ultralytics Platform](https://platform.ultralytics.com) lets you manage datasets, train [image classification](../../tasks/classify.md) models, and deploy them without extensive coding. It is a convenient way to run CIFAR-100 experiments in the cloud, and you can explore more options in our [classification datasets overview](index.md).
