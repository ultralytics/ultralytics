---
comments: true
description: Discover how to leverage the CIFAR-100 dataset for machine learning and computer vision tasks with YOLO. Gain insights on its structure, use, and utilization for model training.
keywords: Ultralytics, YOLO, CIFAR-100 dataset, image classification, machine learning, computer vision, YOLO model training
---

# CIFAR-100 Dataset

The [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) (Canadian Institute For Advanced Research) dataset is a significant extension of the CIFAR-10 dataset, composed of 60,000 32x32 color images in 100 different classes. It was developed by researchers at the CIFAR institute, offering a more challenging dataset for more complex machine learning and computer vision tasks.

## Key Features

- The CIFAR-100 dataset consists of 60,000 images, divided into 100 classes.
- Each class contains 600 images, split into 500 for training and 100 for testing.
- The images are colored and of size 32x32 pixels.
- The 100 different classes are grouped into 20 coarse categories for higher level classification.
- CIFAR-100 is commonly used for training and testing in the field of machine learning and computer vision.

## Dataset Structure

The CIFAR-100 dataset is split into two subsets:

1. **Training Set**: This subset contains 50,000 images used for training machine learning models.
2. **Testing Set**: This subset consists of 10,000 images used for testing and benchmarking the trained models.

## Applications

The CIFAR-100 dataset is extensively used for training and evaluating deep learning models in image classification tasks, such as Convolutional Neural Networks (CNNs), Support Vector Machines (SVMs), and various other machine learning algorithms. The diversity of the dataset in terms of classes and the presence of color images make it a more challenging and comprehensive dataset for research and development in the field of machine learning and computer vision.

## Usage

To train a YOLO model on the CIFAR-100 dataset for 100 epochs with an image size of 32x32, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data='cifar100', epochs=100, imgsz=32)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=cifar100 model=yolov8n-cls.pt epochs=100 imgsz=32
        ```

## Sample Images and Annotations

The CIFAR-100 dataset contains color images of various objects, providing a well-structured dataset for image classification tasks. Here are some examples of images from the dataset:

![Dataset sample image](https://user-images.githubusercontent.com/26833433/239363319-62ebf02f-7469-4178-b066-ccac3cd334db.jpg)

The example showcases the variety and complexity of the objects in the CIFAR-100 dataset, highlighting the importance of a diverse dataset for training robust image classification models.

## Citations and Acknowledgments

If you use the CIFAR-100 dataset in your research or development work, please cite the following paper:

!!! note ""

    === "BibTeX"

        ```bibtex
        @TECHREPORT{Krizhevsky09learningmultiple,
                    author={Alex Krizhevsky},
                    title={Learning multiple layers of features from tiny images},
                    institution={},
                    year={2009}
        }
        ```

We would like to acknowledge Alex Krizhevsky for creating and maintaining the CIFAR-100 dataset as a valuable resource for the machine learning and computer vision research community. For more information about the CIFAR-100 dataset and its creator, visit the [CIFAR-100 dataset website](https://www.cs.toronto.edu/~kriz/cifar.html).
