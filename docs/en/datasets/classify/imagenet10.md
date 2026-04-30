---
comments: true
description: Discover ImageNet10 a compact version of ImageNet for rapid model testing and CI checks. Perfect for quick evaluations in computer vision tasks.
keywords: ImageNet10, ImageNet, Ultralytics, CI tests, sanity checks, training pipelines, computer vision, deep learning, dataset
---

# ImageNet10 Dataset

The [ImageNet10](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/imagenet10.zip) dataset is a small-scale subset of the [ImageNet](https://www.image-net.org/) database, developed by [Ultralytics](https://www.ultralytics.com/) and designed for CI tests, sanity checks, and fast testing of training pipelines. This dataset is composed of the first image in the training set and the first image from the validation set of the first 10 classes in ImageNet. Although significantly smaller, it retains the structure and diversity of the original ImageNet dataset.

## Key Features

- ImageNet10 is a compact version of ImageNet, with 20 images representing the first 10 classes of the original dataset.
- The dataset is organized according to the WordNet hierarchy, mirroring the structure of the full ImageNet dataset.
- It is ideally suited for CI tests, sanity checks, and rapid testing of training pipelines in [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks.
- Although not designed for model benchmarking, it can provide a quick indication of a model's basic functionality and correctness.

## Dataset Structure

The ImageNet10 dataset, like the original [ImageNet](../classify/imagenet.md), is organized using the WordNet hierarchy. Each of the 10 classes in ImageNet10 is described by a synset (a collection of synonymous terms). The images in ImageNet10 are annotated with one or more synsets, providing a compact resource for testing models to recognize various objects and their relationships.

## Applications

The ImageNet10 dataset is useful for quickly testing and debugging computer vision models and pipelines. Its small size allows for rapid iteration, making it ideal for [continuous integration](../../help/CI.md) tests and sanity checks. It can also be used for fast preliminary testing of new models or changes to existing models before moving on to full-scale testing with the complete [ImageNet dataset](../classify/imagenet.md).

## Usage

To test a deep learning model on the ImageNet10 dataset with an image size of 224x224, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

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

The ImageNet10 dataset contains a subset of images from the original ImageNet dataset. These images are chosen to represent the first 10 classes in the dataset, providing a diverse yet compact dataset for quick testing and evaluation.

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

We would like to acknowledge the ImageNet team, led by Olga Russakovsky, Jia Deng, and Li Fei-Fei, for creating and maintaining the ImageNet dataset. The ImageNet10 dataset, while a compact subset, is a valuable resource for quick testing and debugging in the [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and computer vision research community. For more information about the ImageNet dataset and its creators, visit the [ImageNet website](https://www.image-net.org/).

## FAQ

### What is the ImageNet10 dataset and how is it different from the full ImageNet dataset?

The [ImageNet10](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/imagenet10.zip) dataset is a compact subset of the original [ImageNet](https://www.image-net.org/) database, created by Ultralytics for rapid CI tests, sanity checks, and training pipeline evaluations. ImageNet10 comprises only 20 images, representing the first image in the training and validation sets of the first 10 classes in ImageNet. Despite its small size, it maintains the structure and diversity of the full dataset, making it ideal for quick testing but not for benchmarking models.

### How can I use the ImageNet10 dataset to test my deep learning model?

To test your deep learning model on the ImageNet10 dataset with an image size of 224x224, use the following code snippets.

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

The ImageNet10 dataset is designed specifically for CI tests, sanity checks, and quick evaluations in [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) pipelines. Its small size allows for rapid iteration and testing, making it perfect for continuous integration processes where speed is crucial. By maintaining the structural complexity and diversity of the original ImageNet dataset, ImageNet10 provides a reliable indication of a model's basic functionality and correctness without the overhead of processing a large dataset.

### What are the main features of the ImageNet10 dataset?

The ImageNet10 dataset has several key features:

- **Compact Size**: With only 20 images, it allows for rapid testing and debugging.
- **Structured Organization**: Follows the WordNet hierarchy, similar to the full ImageNet dataset.
- **CI and Sanity Checks**: Ideally suited for continuous integration tests and sanity checks.
- **Not for Benchmarking**: While useful for quick model evaluations, it is not designed for extensive benchmarking.

### How does ImageNet10 compare to other small datasets like ImageNette?

While both [ImageNet10](imagenet10.md) and [ImageNette](imagenette.md) are subsets of ImageNet, they serve different purposes. ImageNet10 contains just 20 images (2 per class) from the first 10 classes of ImageNet, making it extremely lightweight for CI testing and quick sanity checks. In contrast, ImageNette contains thousands of images across 10 easily distinguishable classes, making it more suitable for actual model training and development. ImageNet10 is designed for verification of pipeline functionality, while ImageNette is better for meaningful but faster-than-full-ImageNet training experiments.
