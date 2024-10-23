---
comments: true
description: Explore the CIFAR-10 dataset, featuring 60,000 color images in 10 classes. Learn about its structure, applications, and how to train models using YOLO.
keywords: CIFAR-10, dataset, machine learning, computer vision, image classification, YOLO, deep learning, neural networks
---

# CIFAR-10 Dataset

The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (Canadian Institute For Advanced Research) dataset is a collection of images used widely for machine learning and computer vision algorithms. It was developed by researchers at the CIFAR institute and consists of 60,000 32x32 color images in 10 different classes.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/fLBbyhPbWzY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train an Image Classification Model with CIFAR-10 Dataset using Ultralytics YOLOv8
</p>

## Key Features

- The CIFAR-10 dataset consists of 60,000 images, divided into 10 classes.
- Each class contains 6,000 images, split into 5,000 for training and 1,000 for testing.
- The images are colored and of size 32x32 pixels.
- The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.
- CIFAR-10 is commonly used for training and testing in the field of machine learning and computer vision.

## Dataset Structure

The CIFAR-10 dataset is split into two subsets:

1. **Training Set**: This subset contains 50,000 images used for training machine learning models.
2. **Testing Set**: This subset consists of 10,000 images used for testing and benchmarking the trained models.

## Applications

The CIFAR-10 dataset is widely used for training and evaluating deep learning models in image classification tasks, such as Convolutional Neural Networks (CNNs), Support Vector Machines (SVMs), and various other machine learning algorithms. The diversity of the dataset in terms of classes and the presence of color images make it a well-rounded dataset for research and development in the field of machine learning and computer vision.

## Usage

To train a YOLO model on the CIFAR-10 dataset for 100 epochs with an image size of 32x32, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="cifar10", epochs=100, imgsz=32)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=cifar10 model=yolov8n-cls.pt epochs=100 imgsz=32
        ```

## Sample Images and Annotations

The CIFAR-10 dataset contains color images of various objects, providing a well-structured dataset for image classification tasks. Here are some examples of images from the dataset:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/cifar10-sample-image.avif)

The example showcases the variety and complexity of the objects in the CIFAR-10 dataset, highlighting the importance of a diverse dataset for training robust image classification models.

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

We would like to acknowledge Alex Krizhevsky for creating and maintaining the CIFAR-10 dataset as a valuable resource for the machine learning and computer vision research community. For more information about the CIFAR-10 dataset and its creator, visit the [CIFAR-10 dataset website](https://www.cs.toronto.edu/~kriz/cifar.html).

## FAQ

### How can I train a YOLO model on the CIFAR-10 dataset?

To train a YOLO model on the CIFAR-10 dataset using Ultralytics, you can follow the examples provided for both Python and CLI. Here is a basic example to train your model for 100 epochs with an image size of 32x32 pixels:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="cifar10", epochs=100, imgsz=32)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=cifar10 model=yolov8n-cls.pt epochs=100 imgsz=32
        ```

For more details, refer to the model [Training](../../modes/train.md) page.

### What are the key features of the CIFAR-10 dataset?

The CIFAR-10 dataset consists of 60,000 color images divided into 10 classes. Each class contains 6,000 images, with 5,000 for training and 1,000 for testing. The images are 32x32 pixels in size and vary across the following categories:

- Airplanes
- Cars
- Birds
- Cats
- Deer
- Dogs
- Frogs
- Horses
- Ships
- Trucks

This diverse dataset is essential for training image classification models in fields such as machine learning and computer vision. For more information, visit the CIFAR-10 sections on [dataset structure](#dataset-structure) and [applications](#applications).

### Why use the CIFAR-10 dataset for image classification tasks?

The CIFAR-10 dataset is an excellent benchmark for image classification due to its diversity and structure. It contains a balanced mix of 60,000 labeled images across 10 different categories, which helps in training robust and generalized models. It is widely used for evaluating deep learning models, including Convolutional Neural Networks (CNNs) and other machine learning algorithms. The dataset is relatively small, making it suitable for quick experimentation and algorithm development. Explore its numerous applications in the [applications](#applications) section.

### How is the CIFAR-10 dataset structured?

The CIFAR-10 dataset is structured into two main subsets:

1. **Training Set**: Contains 50,000 images used for training machine learning models.
2. **Testing Set**: Consists of 10,000 images for testing and benchmarking the trained models.

Each subset comprises images categorized into 10 classes, with their annotations readily available for model training and evaluation. For more detailed information, refer to the [dataset structure](#dataset-structure) section.

### How can I cite the CIFAR-10 dataset in my research?

If you use the CIFAR-10 dataset in your research or development projects, make sure to cite the following paper:

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

Acknowledging the dataset's creators helps support continued research and development in the field. For more details, see the [citations and acknowledgments](#citations-and-acknowledgments) section.

### What are some practical examples of using the CIFAR-10 dataset?

The CIFAR-10 dataset is often used for training image classification models, such as Convolutional Neural Networks (CNNs) and Support Vector Machines (SVMs). These models can be employed in various computer vision tasks including object detection, image recognition, and automated tagging. To see some practical examples, check the code snippets in the [usage](#usage) section.
