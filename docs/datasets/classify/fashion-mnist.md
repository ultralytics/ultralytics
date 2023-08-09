---
comments: true
description: Learn how to use the Fashion-MNIST dataset for image classification with the Ultralytics YOLO model. Covers dataset structure, labels, applications, and usage.
keywords: Ultralytics, YOLO, Fashion-MNIST, dataset, image classification, machine learning, deep learning, neural networks, training, testing
---

# Fashion-MNIST Dataset

The [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset is a database of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Fashion-MNIST is intended to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms.

## Key Features

- Fashion-MNIST contains 60,000 training images and 10,000 testing images of Zalando's article images.
- The dataset comprises grayscale images of size 28x28 pixels.
- Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255.
- Fashion-MNIST is widely used for training and testing in the field of machine learning, especially for image classification tasks.

## Dataset Structure

The Fashion-MNIST dataset is split into two subsets:

1. **Training Set**: This subset contains 60,000 images used for training machine learning models.
2. **Testing Set**: This subset consists of 10,000 images used for testing and benchmarking the trained models.

## Labels

Each training and test example is assigned to one of the following labels:

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

## Applications

The Fashion-MNIST dataset is widely used for training and evaluating deep learning models in image classification tasks, such as Convolutional Neural Networks (CNNs), Support Vector Machines (SVMs), and various other machine learning algorithms. The dataset's simple and well-structured format makes it an essential resource for researchers and practitioners in the field of machine learning and computer vision.

## Usage

To train a CNN model on the Fashion-MNIST dataset for 100 epochs with an image size of 28x28, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

        # Train the model
        model.train(data='fashion-mnist', epochs=100, imgsz=28)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=fashion-mnist model=yolov8n-cls.pt epochs=100 imgsz=28
        ```

## Sample Images and Annotations

The Fashion-MNIST dataset contains grayscale images of Zalando's article images, providing a well-structured dataset for image classification tasks. Here are some examples of images from the dataset:

![Dataset sample image](https://user-images.githubusercontent.com/26833433/239359139-ce0a434e-9056-43e0-a306-3214f193dcce.png)

The example showcases the variety and complexity of the images in the Fashion-MNIST dataset, highlighting the importance of a diverse dataset for training robust image classification models.

## Acknowledgments

If you use the Fashion-MNIST dataset in your research or development work, please acknowledge the dataset by linking to the [GitHub repository](https://github.com/zalandoresearch/fashion-mnist). This dataset was made available by Zalando Research.
