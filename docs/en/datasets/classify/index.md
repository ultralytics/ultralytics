---
comments: true
description: Explore image classification datasets supported by Ultralytics, learn the standard dataset format, and set up your own dataset for training models.
keywords: Ultralytics, image classification, dataset, machine learning, CIFAR-10, ImageNet, MNIST, torchvision
---

# Image Classification Datasets Overview

## Dataset format

The folder structure for classification datasets in torchvision typically follows a standard format:

```
root/
|-- class1/
|   |-- img1.jpg
|   |-- img2.jpg
|   |-- ...
|
|-- class2/
|   |-- img1.jpg
|   |-- img2.jpg
|   |-- ...
|
|-- class3/
|   |-- img1.jpg
|   |-- img2.jpg
|   |-- ...
|
|-- ...
```

In this folder structure, the `root` directory contains one subdirectory for each class in the dataset. Each subdirectory is named after the corresponding class and contains all the images for that class. Each image file is named uniquely and is typically in a common image file format such as JPEG or PNG.

** Example **

For example, in the CIFAR10 dataset, the folder structure would look like this:

```
cifar-10-/
|
|-- train/
|   |-- airplane/
|   |   |-- 10008_airplane.png
|   |   |-- 10009_airplane.png
|   |   |-- ...
|   |
|   |-- automobile/
|   |   |-- 1000_automobile.png
|   |   |-- 1001_automobile.png
|   |   |-- ...
|   |
|   |-- bird/
|   |   |-- 10014_bird.png
|   |   |-- 10015_bird.png
|   |   |-- ...
|   |
|   |-- ...
|
|-- test/
|   |-- airplane/
|   |   |-- 10_airplane.png
|   |   |-- 11_airplane.png
|   |   |-- ...
|   |
|   |-- automobile/
|   |   |-- 100_automobile.png
|   |   |-- 101_automobile.png
|   |   |-- ...
|   |
|   |-- bird/
|   |   |-- 1000_bird.png
|   |   |-- 1001_bird.png
|   |   |-- ...
|   |
|   |-- ...
```

In this example, the `train` directory contains subdirectories for each class in the dataset, and each class subdirectory contains all the images for that class. The `test` directory has a similar structure. The `root` directory also contains other files that are part of the CIFAR10 dataset.

## Usage

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data='path/to/dataset', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=path/to/data model=yolov8n-cls.pt epochs=100 imgsz=640
        ```

## Supported Datasets

Ultralytics supports the following datasets with automatic download:

- [Caltech 101](caltech101.md): A dataset containing images of 101 object categories for image classification tasks.
- [Caltech 256](caltech256.md): An extended version of Caltech 101 with 256 object categories and more challenging images.
- [CIFAR-10](cifar10.md): A dataset of 60K 32x32 color images in 10 classes, with 6K images per class.
- [CIFAR-100](cifar100.md): An extended version of CIFAR-10 with 100 object categories and 600 images per class.
- [Fashion-MNIST](fashion-mnist.md): A dataset consisting of 70,000 grayscale images of 10 fashion categories for image classification tasks.
- [ImageNet](imagenet.md): A large-scale dataset for object detection and image classification with over 14 million images and 20,000 categories.
- [ImageNet-10](imagenet10.md): A smaller subset of ImageNet with 10 categories for faster experimentation and testing.
- [Imagenette](imagenette.md): A smaller subset of ImageNet that contains 10 easily distinguishable classes for quicker training and testing.
- [Imagewoof](imagewoof.md): A more challenging subset of ImageNet containing 10 dog breed categories for image classification tasks.
- [MNIST](mnist.md): A dataset of 70,000 grayscale images of handwritten digits for image classification tasks.

### Adding your own dataset

If you have your own dataset and would like to use it for training classification models with Ultralytics, ensure that it follows the format specified above under "Dataset format" and then point your `data` argument to the dataset directory.
