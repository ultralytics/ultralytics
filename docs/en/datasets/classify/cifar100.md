---
comments: true
description: Explore the CIFAR-100 dataset, consisting of 60,000 32x32 color images across 100 classes. Ideal for machine learning and computer vision tasks.
keywords: CIFAR-100, dataset, machine learning, computer vision, image classification, deep learning, YOLO, training, testing, Alex Krizhevsky
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
        model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="cifar100", epochs=100, imgsz=32)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=cifar100 model=yolov8n-cls.pt epochs=100 imgsz=32
        ```

## Sample Images and Annotations

The CIFAR-100 dataset contains color images of various objects, providing a well-structured dataset for image classification tasks. Here are some examples of images from the dataset:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/cifar100-sample-image.avif)

The example showcases the variety and complexity of the objects in the CIFAR-100 dataset, highlighting the importance of a diverse dataset for training robust image classification models.

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

We would like to acknowledge Alex Krizhevsky for creating and maintaining the CIFAR-100 dataset as a valuable resource for the machine learning and computer vision research community. For more information about the CIFAR-100 dataset and its creator, visit the [CIFAR-100 dataset website](https://www.cs.toronto.edu/~kriz/cifar.html).

## FAQ

### What is the CIFAR-100 dataset and why is it significant?

The [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) is a large collection of 60,000 32x32 color images classified into 100 classes. Developed by the Canadian Institute For Advanced Research (CIFAR), it provides a challenging dataset ideal for complex machine learning and computer vision tasks. Its significance lies in the diversity of classes and the small size of the images, making it a valuable resource for training and testing deep learning models, like Convolutional Neural Networks (CNNs), using frameworks such as Ultralytics YOLO.

### How do I train a YOLO model on the CIFAR-100 dataset?

You can train a YOLO model on the CIFAR-100 dataset using either Python or CLI commands. Here's how:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="cifar100", epochs=100, imgsz=32)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=cifar100 model=yolov8n-cls.pt epochs=100 imgsz=32
        ```

For a comprehensive list of available arguments, please refer to the model [Training](../../modes/train.md) page.

### What are the primary applications of the CIFAR-100 dataset?

The CIFAR-100 dataset is extensively used in training and evaluating deep learning models for image classification. Its diverse set of 100 classes, grouped into 20 coarse categories, provides a challenging environment for testing algorithms such as Convolutional Neural Networks (CNNs), Support Vector Machines (SVMs), and various other machine learning approaches. This dataset is a key resource in research and development within machine learning and computer vision fields.

### How is the CIFAR-100 dataset structured?

The CIFAR-100 dataset is split into two main subsets:

1. **Training Set**: Contains 50,000 images used for training machine learning models.
2. **Testing Set**: Consists of 10,000 images used for testing and benchmarking the trained models.

Each of the 100 classes contains 600 images, with 500 images for training and 100 for testing, making it uniquely suited for rigorous academic and industrial research.

### Where can I find sample images and annotations from the CIFAR-100 dataset?

The CIFAR-100 dataset includes a variety of color images of various objects, making it a structured dataset for image classification tasks. You can refer to the documentation page to see [sample images and annotations](#sample-images-and-annotations). These examples highlight the dataset's diversity and complexity, important for training robust image classification models.
