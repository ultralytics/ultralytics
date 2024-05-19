---
comments: true
description: Explore the compact ImageNet10 Dataset developed by Ultralytics. Ideal for fast testing of computer vision training pipelines and CV model sanity checks.
keywords: Ultralytics, YOLO, ImageNet10 Dataset, Image detection, Deep Learning, ImageNet, AI model testing, Computer vision, Machine learning
---

# ImageNet10 Dataset

The [ImageNet10](https://github.com/ultralytics/yolov5/releases/download/v1.0/imagenet10.zip) dataset is a small-scale subset of the [ImageNet](https://www.image-net.org/) database, developed by [Ultralytics](https://ultralytics.com) and designed for CI tests, sanity checks, and fast testing of training pipelines. This dataset is composed of the first image in the training set and the first image from the validation set of the first 10 classes in ImageNet. Although significantly smaller, it retains the structure and diversity of the original ImageNet dataset.

## Key Features

- ImageNet10 is a compact version of ImageNet, with 20 images representing the first 10 classes of the original dataset.
- The dataset is organized according to the WordNet hierarchy, mirroring the structure of the full ImageNet dataset.
- It is ideally suited for CI tests, sanity checks, and rapid testing of training pipelines in computer vision tasks.
- Although not designed for model benchmarking, it can provide a quick indication of a model's basic functionality and correctness.

## Dataset Structure

The ImageNet10 dataset, like the original ImageNet, is organized using the WordNet hierarchy. Each of the 10 classes in ImageNet10 is described by a synset (a collection of synonymous terms). The images in ImageNet10 are annotated with one or more synsets, providing a compact resource for testing models to recognize various objects and their relationships.

## Applications

The ImageNet10 dataset is useful for quickly testing and debugging computer vision models and pipelines. Its small size allows for rapid iteration, making it ideal for continuous integration tests and sanity checks. It can also be used for fast preliminary testing of new models or changes to existing models before moving on to full-scale testing with the complete ImageNet dataset.

## Usage

To test a deep learning model on the ImageNet10 dataset with an image size of 224x224, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! Example "Test Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="imagenet10", epochs=5, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo train data=imagenet10 model=yolov8n-cls.pt epochs=5 imgsz=224
        ```

## Sample Images and Annotations

The ImageNet10 dataset contains a subset of images from the original ImageNet dataset. These images are chosen to represent the first 10 classes in the dataset, providing a diverse yet compact dataset for quick testing and evaluation.

![Dataset sample images](https://user-images.githubusercontent.com/26833433/239689723-16f9b4a7-becc-4deb-b875-d3e5c28eb03b.png) The example showcases the variety and complexity of the images in the ImageNet10 dataset, highlighting its usefulness for sanity checks and quick testing of computer vision models.

## Citations and Acknowledgments

If you use the ImageNet10 dataset in your research or development work, please cite the original ImageNet paper:

!!! Quote ""

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

We would like to acknowledge the ImageNet team, led by Olga Russakovsky, Jia Deng, and Li Fei-Fei, for creating and maintaining the ImageNet dataset. The ImageNet10 dataset, while a compact subset, is a valuable resource for quick testing and debugging in the machine learning and computer vision research community. For more information about the ImageNet dataset and its creators, visit the [ImageNet website](https://www.image-net.org/).
