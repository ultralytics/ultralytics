---
title: Caltech-256 Image Classification Dataset
comments: true
creator:
    name: California Institute of Technology
    url: https://data.caltech.edu/records/nyy15-4j048
license:
    name: CC-BY-4.0
description: Train YOLO image classification models on Caltech-256, a benchmark of 30,607 images across 256 object categories plus a background class, with automatic 80/20 splitting.
keywords: Caltech-256, dataset, image classification, object recognition, machine learning, computer vision, YOLO, deep learning, AI
---

# Caltech-256 Dataset

The [Caltech-256](https://data.caltech.edu/records/nyy15-4j048) dataset is a classic [image classification](https://www.ultralytics.com/glossary/image-classification) benchmark of 30,607 images spanning 256 object categories plus one background class. Each category holds at least 80 images of real-world objects — animals, vehicles, household items, and people — making it a larger, more challenging successor to [Caltech-101](caltech101.md) for object recognition models.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Y7cfNkqSdMg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train <a href="https://www.ultralytics.com/glossary/image-classification">Image Classification</a> Model using Caltech-256 Dataset with Ultralytics YOLO26
</p>

!!! note "Automatic Data Splitting"

    Caltech-256 ships without a predefined train/validation split. The training commands below automatically split it 80% train / 20% validation, so no manual preparation is needed.

## Key Features

- Caltech-256 contains 30,607 color images across 256 object categories plus one `257.clutter` background class (257 class folders in total).
- The categories span a wide variety of real-world objects, including animals, vehicles, household items, and people.
- Each category holds at least 80 images, with the largest holding up to about 800, so class sizes are imbalanced.
- Images are of variable sizes and resolutions.
- Caltech-256 is widely used to benchmark [image classification](https://www.ultralytics.com/glossary/image-classification) and object recognition algorithms.

## Dataset Structure

Caltech-256 is distributed as 257 folders — one per class, covering 256 object categories plus a `257.clutter` background class — with no predefined train/validation split. When you launch training, Ultralytics automatically partitions the images so models train across all 257 classes without any manual setup:

- **Classes**: 257 (256 object categories + 1 background)
- **Total images**: 30,607
- **Train/validation split**: automatic 80% / 20% (≈24,385 train, ≈6,222 validation)
- **Images per class**: at least 80 (imbalanced, up to about 800)

## Applications

The Caltech-256 dataset is widely used to train and evaluate [image classification](https://www.ultralytics.com/glossary/image-classification) and object recognition models, including [Convolutional Neural Networks](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) (CNNs) and [Support Vector Machines](https://www.ultralytics.com/glossary/support-vector-machine-svm) (SVMs). Its large category count and high-quality images make it a popular benchmark for [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) research and prototyping.

## Usage

Train a YOLO model on Caltech-256 for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at an image size of 416. For the full list of available arguments, see the [Training](../../modes/train.md) page and the [image classification](../../tasks/classify.md) task guide.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="caltech256", epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=caltech256 model=yolo26n-cls.pt epochs=100 imgsz=416
        ```

## Sample Images and Annotations

The Caltech-256 dataset contains high-quality color images of various objects, providing a well-structured dataset for [image classification](https://www.ultralytics.com/glossary/image-classification) tasks. Here are some examples of images from the dataset ([credit](https://ml4a.github.io/demos/tsne_viewer.html)):

![Caltech-256 image classification dataset samples](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/caltech256-sample-image.avif)

The samples show the diversity and complexity of the objects in the Caltech-256 dataset, underlining the value of a varied dataset for training robust object recognition models.

## Citations and Acknowledgments

If you use the Caltech-256 dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{griffin2007caltech,
                 title={Caltech-256 object category dataset},
                 author={Griffin, Gregory and Holub, Alex and Perona, Pietro},
                 year={2007}
        }
        ```

We would like to acknowledge Gregory Griffin, Alex Holub, and Pietro Perona for creating and maintaining the Caltech-256 dataset as a valuable resource for the [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and computer vision research community. For more information about the Caltech-256 dataset and its creators, visit the [Caltech-256 dataset website](https://data.caltech.edu/records/nyy15-4j048).

## FAQ

### What is the Caltech-256 dataset used for in machine learning?

The [Caltech-256](https://data.caltech.edu/records/nyy15-4j048) dataset is widely used to train and benchmark [image classification](https://www.ultralytics.com/glossary/image-classification) and object recognition models. It contains 30,607 images across 256 object categories plus a background class, providing a larger and more challenging benchmark than Caltech-101 for algorithms such as Convolutional [Neural Networks](https://www.ultralytics.com/glossary/neural-network-nn) (CNNs) and Support Vector Machines (SVMs).

### How can I train an Ultralytics YOLO model on the Caltech-256 dataset?

To train an Ultralytics YOLO model on Caltech-256, use the code snippets below. The dataset downloads automatically on first use. For a full list of arguments, see the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="caltech256", epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=caltech256 model=yolo26n-cls.pt epochs=100 imgsz=416
        ```

### How many classes does the Caltech-256 dataset have?

Caltech-256 contains 256 object categories plus one `257.clutter` background class, for 257 class folders and 30,607 images in total. When you train with Ultralytics, the model learns all 257 classes. Each category holds at least 80 images, but class sizes are imbalanced, with the largest holding up to about 800 images.

### How is the Caltech-256 dataset split into training and validation sets?

Caltech-256 has no predefined split. The first time you train, Ultralytics automatically divides it 80% training / 20% validation — about 24,385 training and 6,222 validation images — so you do not need to create splits manually. To control the split yourself, organize the images into `train/` and `val/` folders before training.

### Can I use Ultralytics Platform for training models on the Caltech-256 dataset?

Yes. [Ultralytics Platform](https://platform.ultralytics.com) lets you manage datasets, train [image classification](../../tasks/classify.md) models, and deploy them without extensive coding. It is a convenient way to run Caltech-256 experiments in the cloud, and you can explore more options in our [classification datasets overview](index.md).
