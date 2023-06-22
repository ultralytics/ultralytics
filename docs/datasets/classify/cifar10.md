---
comments: true
description: Learn about the CIFAR-10 dataset, a collection of images that are commonly used to train machine learning and computer vision algorithms.
keywords: CIFAR-10 dataset, YOLO model training, image classification, deep learning, computer vision, object detection, machine learning, convolutional neural networks, Alex Krizhevsky
---

# CIFAR-10 Dataset

The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (Canadian Institute For Advanced Research) dataset is a collection of images used widely for machine learning and computer vision algorithms. It was developed by researchers at the CIFAR institute and consists of 60,000 32x32 color images in 10 different classes.

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
        model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
        
        # Train the model
        model.train(data='cifar10', epochs=100, imgsz=32)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=cifar10 model=yolov8n-cls.pt epochs=100 imgsz=32
        ```

## Sample Images and Annotations

The CIFAR-10 dataset contains color images of various objects, providing a well-structured dataset for image classification tasks. Here are some examples of images from the dataset:

![Dataset sample image](https://miro.medium.com/max/1100/1*SZnidBt7CQ4Xqcag6rd8Ew.png)

The example showcases the variety and complexity of the objects in the CIFAR-10 dataset, highlighting the importance of a diverse dataset for training robust image classification models.

## Citations and Acknowledgments

If you use the CIFAR-10 dataset in your research or development work, please cite the following paper:

```bibtex
@TECHREPORT{Krizhevsky09learningmultiple,
            author={Alex Krizhevsky},
            title={Learning multiple layers of features from tiny images},
            institution={},
            year={2009}
}
```

We would like to acknowledge Alex Krizhevsky for creating and maintaining the CIFAR-10 dataset as a valuable resource for the machine learning and computer vision research community. For more information about the CIFAR-10 dataset and its creator, visit the [CIFAR-10 dataset website](https://www.cs.toronto.edu/~kriz/cifar.html).