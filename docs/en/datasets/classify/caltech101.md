---
title: Caltech-101 Image Classification Dataset
comments: true
description: Train YOLO image classification models on Caltech-101, a benchmark of 9,144 images across 101 object categories plus a background class, with automatic 80/20 splitting.
keywords: Caltech-101, dataset, image classification, object recognition, machine learning, computer vision, YOLO, deep learning, AI
---

# Caltech-101 Dataset

The [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) dataset is a classic [image classification](https://www.ultralytics.com/glossary/image-classification) benchmark of 9,144 images spanning 101 object categories plus one background class. Each category holds about 40 to 800 images of real-world objects — animals, vehicles, household items, and people — making it a compact yet challenging benchmark for object recognition models.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/isc06_9qnM0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train <a href="https://www.ultralytics.com/glossary/image-classification">Image Classification</a> Model using Caltech-101 Dataset with Ultralytics Platform
</p>

!!! note "Automatic Data Splitting"

    Caltech-101 ships without a predefined train/validation split. The training commands below automatically split it 80% train / 20% validation, so no manual preparation is needed.

## Key Features

- Caltech-101 contains 9,144 color images across 101 object categories plus one `BACKGROUND_Google` class (102 class folders in total).
- The categories span a wide variety of real-world objects, including animals, vehicles, household items, and people.
- Each category holds about 40 to 800 images, so class sizes are imbalanced.
- Images are of variable sizes, most roughly 300x200 pixels (medium resolution).
- Caltech-101 is widely used to benchmark [image classification](https://www.ultralytics.com/glossary/image-classification) and object recognition algorithms.

## Dataset Structure

Caltech-101 is distributed as 102 folders — one per class, covering 101 object categories plus a `BACKGROUND_Google` class — with no predefined train/validation split. When you launch training, Ultralytics automatically partitions the images so models train across all 102 classes without any manual setup:

- **Classes**: 102 (101 object categories + 1 background)
- **Total images**: 9,144
- **Train/validation split**: automatic 80% / 20% (≈7,280 train, ≈1,864 validation)
- **Images per class**: about 40 to 800 (imbalanced)

## Applications

Caltech-101 is widely used to train and evaluate [image classification](https://www.ultralytics.com/glossary/image-classification) and object recognition models, including [Convolutional Neural Networks](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) (CNNs) and [Support Vector Machines](https://www.ultralytics.com/glossary/support-vector-machine-svm) (SVMs). Its broad category coverage and clean, labeled images make it a popular benchmark for [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) research and prototyping.

## Usage

Train a YOLO model on Caltech-101 for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at an image size of 416. For the full list of available arguments, see the [Training](../../modes/train.md) page and the [image classification](../../tasks/classify.md) task guide.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="caltech101", epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=caltech101 model=yolo26n-cls.pt epochs=100 imgsz=416
        ```

## Sample Images and Annotations

The Caltech-101 dataset contains high-quality color images of various objects, providing a well-structured dataset for [image classification](https://www.ultralytics.com/glossary/image-classification) tasks. Here are some examples of images from the dataset:

![Caltech-101 image classification dataset samples](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/caltech101-sample-image.avif)

The samples show the variety of categories and the natural, centered framing typical of Caltech-101, which makes it a clean starting point for training robust object recognition models.

## Citations and Acknowledgments

If you use the Caltech-101 dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{fei2007learning,
          title={Learning generative visual models from few training examples: An incremental Bayesian approach tested on 101 object categories},
          author={Fei-Fei, Li and Fergus, Rob and Perona, Pietro},
          journal={Computer vision and Image understanding},
          volume={106},
          number={1},
          pages={59--70},
          year={2007},
          publisher={Elsevier}
        }
        ```

We would like to acknowledge Li Fei-Fei, Rob Fergus, and Pietro Perona for creating and maintaining the Caltech-101 dataset as a valuable resource for the machine learning and computer vision research community. For more information about the Caltech-101 dataset and its creators, visit the [Caltech-101 dataset website](https://data.caltech.edu/records/mzrjq-6wc02).

## FAQ

### What is the Caltech-101 dataset used for in machine learning?

The [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) dataset is widely used to train and benchmark [image classification](https://www.ultralytics.com/glossary/image-classification) and object recognition models. It contains 9,144 images across 101 object categories plus a background class, providing a challenging benchmark for evaluating algorithms such as Convolutional [Neural Networks](https://www.ultralytics.com/glossary/neural-network-nn) (CNNs) and Support Vector Machines (SVMs).

### How can I train an Ultralytics YOLO model on the Caltech-101 dataset?

To train an Ultralytics YOLO model on Caltech-101, use the code snippets below. The dataset downloads automatically on first use. For a full list of arguments, see the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="caltech101", epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=caltech101 model=yolo26n-cls.pt epochs=100 imgsz=416
        ```

### How many classes does the Caltech-101 dataset have?

Caltech-101 contains 101 object categories plus one `BACKGROUND_Google` class, for 102 class folders and 9,144 images in total. When you train with Ultralytics, the model learns all 102 classes. Category sizes are imbalanced, ranging from about 40 to 800 images each.

### How is the Caltech-101 dataset split into training and validation sets?

Caltech-101 has no predefined split. The first time you train, Ultralytics automatically divides it 80% training / 20% validation — about 7,280 training and 1,864 validation images — so you do not need to create splits manually. To control the split yourself, organize the images into `train/` and `val/` folders before training.

### Can I use Ultralytics Platform for training models on the Caltech-101 dataset?

Yes. [Ultralytics Platform](https://platform.ultralytics.com) lets you manage datasets, train [image classification](../../tasks/classify.md) models, and deploy them without extensive coding. It is a convenient way to run Caltech-101 experiments in the cloud, and you can explore more options in our [classification datasets overview](index.md).
