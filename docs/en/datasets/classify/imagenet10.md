---
title: ImageNet10 Image Classification Dataset
comments: true
description: ImageNet10 is a tiny 24-image subset of ImageNet across its first 10 classes, built by Ultralytics for fast CI tests, sanity checks, and pipeline validation.
keywords: ImageNet10, ImageNet subset, Ultralytics, CI tests, sanity checks, training pipelines, image classification, computer vision, deep learning, dataset
---

# ImageNet10 Dataset

The Ultralytics **ImageNet10** dataset (`data="imagenet10"`) is a tiny 24-image subset of [ImageNet](imagenet.md) spanning its first 10 classes, built for continuous-integration tests, sanity checks, and fast validation of training pipelines. It holds 12 training and 12 validation images organized in the same WordNet-synset folder structure as the full dataset, so a model that trains on ImageNet trains on ImageNet10 unchanged — in seconds instead of hours. It is designed for verifying that a pipeline runs end to end, not for benchmarking [accuracy](https://www.ultralytics.com/glossary/accuracy).

## Key Features

- ImageNet10 contains just 24 images (12 training, 12 validation) drawn from the first 10 classes of ImageNet.
- The dataset is organized according to the WordNet hierarchy, mirroring the per-class synset folders of the full ImageNet dataset.
- It is purpose-built for [continuous integration](../../help/CI.md) tests, sanity checks, and rapid debugging of training pipelines in [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks.
- Although not designed for model benchmarking, it gives a fast indication of a model's basic functionality and correctness.

## Dataset Structure

ImageNet10 ships with a predefined split, using the first 10 classes of the full [ImageNet](imagenet.md) dataset (`n01440764` tench through `n01518878` ostrich):

| Split      | Images | Classes |
| ---------- | ------ | ------- |
| Train      | 12     | 10      |
| Validation | 12     | 10      |

Each of the 10 classes is a WordNet synset (a set of synonymous terms), and images sit in per-class folders named by synset ID — the exact layout Ultralytics classification training expects. This makes ImageNet10 a compact, structurally faithful stand-in for the full dataset when testing that a model recognizes the expected folder format.

## Applications

The ImageNet10 dataset is useful for quickly testing and debugging computer vision models and pipelines. Its small size allows for rapid iteration, making it ideal for [continuous integration](../../help/CI.md) tests and sanity checks. It is also handy for fast preliminary testing of new models or code changes before moving on to full-scale runs with the complete [ImageNet](imagenet.md) dataset.

## Usage

To test a classification model on the ImageNet10 dataset at an image size of 224x224, use the code snippets below. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Test Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="imagenet10", epochs=5, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=imagenet10 model=yolo26n-cls.pt epochs=5 imgsz=224
        ```

## Sample Images and Annotations

The ImageNet10 dataset contains a subset of images from the original ImageNet dataset, chosen to represent its first 10 classes and provide a diverse yet compact resource for quick testing and evaluation.

![ImageNet-10 classification dataset sample images](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/imagenet10-sample-images.avif)

The example showcases the variety and complexity of the images in the ImageNet10 dataset, highlighting its usefulness for sanity checks and quick testing of computer vision models.

## Citations and Acknowledgments

If you use the ImageNet10 dataset in your research or development work, please cite the original ImageNet paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{ILSVRC15,
                 author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
                 title={ImageNet Large Scale Visual Recognition Challenge},
                 year={2015},
                 journal={International Journal of Computer Vision (IJCV)},
                 volume={115},
                 number={3},
                 pages={211-252}
        }
        ```

We would like to acknowledge the ImageNet team, led by Olga Russakovsky, Jia Deng, and Li Fei-Fei, for creating and maintaining the ImageNet dataset. The ImageNet10 subset, created by Ultralytics, is a valuable resource for quick testing and debugging in the [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and computer vision research community. For more information about the ImageNet dataset and its creators, visit the [ImageNet website](https://www.image-net.org/).

## FAQ

### What is the ImageNet10 dataset and how is it different from the full ImageNet dataset?

The ImageNet10 dataset is a compact subset of the [ImageNet](imagenet.md) database, created by Ultralytics for rapid CI tests, sanity checks, and training-pipeline evaluations. It contains 24 images (12 training and 12 validation) from the first 10 classes of ImageNet. Despite its small size, it preserves the WordNet folder structure of the full dataset, making it ideal for quick pipeline testing but not for benchmarking model accuracy.

### How many images and classes does ImageNet10 have?

ImageNet10 contains 24 images in total — 12 for training and 12 for validation — spread across the first 10 classes of ImageNet. Each class is a WordNet synset stored in its own folder, so the dataset mirrors the layout of the full ImageNet dataset in a fraction of the size.

### How can I use the ImageNet10 dataset to test my deep learning model?

To test your classification model on ImageNet10 at an image size of 224x224, use the following code snippets.

!!! example "Test Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="imagenet10", epochs=5, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=imagenet10 model=yolo26n-cls.pt epochs=5 imgsz=224
        ```

Refer to the [Training](../../modes/train.md) page for a comprehensive list of available arguments.

### Why should I use the ImageNet10 dataset for CI tests and sanity checks?

ImageNet10 is designed specifically for CI tests, sanity checks, and quick evaluations in [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) pipelines. Its 24-image size allows for near-instant iteration, making it ideal for continuous integration where speed is crucial. By preserving the folder structure of the full ImageNet dataset, it provides a reliable check of a model's basic functionality and correctness without the overhead of processing a large dataset.

### How does ImageNet10 compare to other small datasets like ImageNette?

While both [ImageNet10](imagenet10.md) and [ImageNette](imagenette.md) are subsets of ImageNet, they serve different purposes. ImageNet10 contains just 24 images from the first 10 classes, making it extremely lightweight for CI testing and quick sanity checks. In contrast, ImageNette contains over 13,000 images across 10 easily distinguishable classes, making it suitable for actual model training and development. ImageNet10 verifies pipeline functionality, while ImageNette is better for meaningful but faster-than-full-ImageNet training experiments.
