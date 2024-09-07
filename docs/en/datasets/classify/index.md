---
comments: true
description: Learn how to structure datasets for YOLO classification tasks. Detailed folder structure and usage examples for effective training.
keywords: YOLO, image classification, dataset structure, CIFAR-10, Ultralytics, machine learning, training data, model evaluation
---

# Image Classification Datasets Overview

### Dataset Structure for YOLO Classification Tasks

For [Ultralytics](https://www.ultralytics.com/) YOLO classification tasks, the dataset must be organized in a specific split-directory structure under the `root` directory to facilitate proper training, testing, and optional validation processes. This structure includes separate directories for training (`train`) and testing (`test`) phases, with an optional directory for validation (`val`).

Each of these directories should contain one subdirectory for each class in the dataset. The subdirectories are named after the corresponding class and contain all the images for that class. Ensure that each image file is named uniquely and stored in a common format such as JPEG or PNG.

**Folder Structure Example**

Consider the CIFAR-10 dataset as an example. The folder structure should look like this:

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
|
|-- val/ (optional)
|   |-- airplane/
|   |   |-- 105_airplane.png
|   |   |-- 106_airplane.png
|   |   |-- ...
|   |
|   |-- automobile/
|   |   |-- 102_automobile.png
|   |   |-- 103_automobile.png
|   |   |-- ...
|   |
|   |-- bird/
|   |   |-- 1045_bird.png
|   |   |-- 1046_bird.png
|   |   |-- ...
|   |
|   |-- ...
```

This structured approach ensures that the model can effectively learn from well-organized classes during the training phase and accurately evaluate performance during testing and validation phases.

## Usage

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="path/to/dataset", epochs=100, imgsz=640)
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

## FAQ

### How do I structure my dataset for YOLO classification tasks?

To structure your dataset for Ultralytics YOLO classification tasks, you should follow a specific split-directory format. Organize your dataset into separate directories for `train`, `test`, and optionally `val`. Each of these directories should contain subdirectories named after each class, with the corresponding images inside. This facilitates smooth training and evaluation processes. For an example, consider the CIFAR-10 dataset format:

```
cifar-10-/
|-- train/
|   |-- airplane/
|   |-- automobile/
|   |-- bird/
|   ...
|-- test/
|   |-- airplane/
|   |-- automobile/
|   |-- bird/
|   ...
|-- val/ (optional)
|   |-- airplane/
|   |-- automobile/
|   |-- bird/
|   ...
```

For more details, visit [Dataset Structure for YOLO Classification Tasks](#dataset-structure-for-yolo-classification-tasks).

### What datasets are supported by Ultralytics YOLO for image classification?

Ultralytics YOLO supports automatic downloading of several datasets for image classification, including:

- [Caltech 101](caltech101.md)
- [Caltech 256](caltech256.md)
- [CIFAR-10](cifar10.md)
- [CIFAR-100](cifar100.md)
- [Fashion-MNIST](fashion-mnist.md)
- [ImageNet](imagenet.md)
- [ImageNet-10](imagenet10.md)
- [Imagenette](imagenette.md)
- [Imagewoof](imagewoof.md)
- [MNIST](mnist.md)

These datasets are structured in a way that makes them easy to use with YOLO. Each dataset's page provides further details about its structure and applications.

### How do I add my own dataset for YOLO image classification?

To use your own dataset with Ultralytics YOLO, ensure it follows the specified directory format required for the classification task, with separate `train`, `test`, and optionally `val` directories, and subdirectories for each class containing the respective images. Once your dataset is structured correctly, point the `data` argument to your dataset's root directory when initializing the training script. Here's an example in Python:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="path/to/your/dataset", epochs=100, imgsz=640)
```

More details can be found in the [Adding your own dataset](#adding-your-own-dataset) section.

### Why should I use Ultralytics YOLO for image classification?

Ultralytics YOLO offers several benefits for image classification, including:

- **Pretrained Models**: Load pretrained models like `yolov8n-cls.pt` to jump-start your training process.
- **Ease of Use**: Simple API and CLI commands for training and evaluation.
- **High Performance**: State-of-the-art accuracy and speed, ideal for real-time applications.
- **Support for Multiple Datasets**: Seamless integration with various popular datasets like CIFAR-10, ImageNet, and more.
- **Community and Support**: Access to extensive documentation and an active community for troubleshooting and improvements.

For additional insights and real-world applications, you can explore [Ultralytics YOLO](https://www.ultralytics.com/yolo).

### How can I train a model using Ultralytics YOLO?

Training a model using Ultralytics YOLO can be done easily in both Python and CLI. Here's an example:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load a pretrained model

        # Train the model
        results = model.train(data="path/to/dataset", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=path/to/data model=yolov8n-cls.pt epochs=100 imgsz=640
        ```

These examples demonstrate the straightforward process of training a YOLO model using either approach. For more information, visit the [Usage](#usage) section.
