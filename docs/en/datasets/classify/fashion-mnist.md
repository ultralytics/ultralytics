---
comments: true
description: Explore the Fashion-MNIST dataset, a modern replacement for MNIST with 70,000 Zalando article images. Ideal for benchmarking machine learning models.
keywords: Fashion-MNIST, image classification, Zalando dataset, machine learning, deep learning, CNN, dataset overview
---

# Fashion-MNIST Dataset

The [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset is a database of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Fashion-MNIST is intended to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) algorithms.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/eX5ad6udQ9Q"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to do <a href="https://www.ultralytics.com/glossary/image-classification">Image Classification</a> on Fashion MNIST Dataset using Ultralytics YOLO11
</p>

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

The Fashion-MNIST dataset is widely used for training and evaluating deep learning models in image classification tasks, such as [Convolutional Neural Networks](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) (CNNs), [Support Vector Machines](https://www.ultralytics.com/glossary/support-vector-machine-svm) (SVMs), and various other machine learning algorithms. The dataset's simple and well-structured format makes it an essential resource for researchers and practitioners in the field of machine learning and computer vision.

## Usage

To train a CNN model on the Fashion-MNIST dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 28x28, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="fashion-mnist", epochs=100, imgsz=28)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=fashion-mnist model=yolo11n-cls.pt epochs=100 imgsz=28
        ```

## Sample Images and Annotations

The Fashion-MNIST dataset contains grayscale images of Zalando's article images, providing a well-structured dataset for image classification tasks. Here are some examples of images from the dataset:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/fashion-mnist-sample.avif)

The example showcases the variety and complexity of the images in the Fashion-MNIST dataset, highlighting the importance of a diverse dataset for training robust image classification models.

## Acknowledgments

If you use the Fashion-MNIST dataset in your research or development work, please acknowledge the dataset by linking to the [GitHub repository](https://github.com/zalandoresearch/fashion-mnist). This dataset was made available by Zalando Research.

## FAQ

### What is the Fashion-MNIST dataset and how is it different from MNIST?

The [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset is a collection of 70,000 grayscale images of Zalando's article images, intended as a modern replacement for the original MNIST dataset. It serves as a benchmark for machine learning models in the context of image classification tasks. Unlike MNIST, which contains handwritten digits, Fashion-MNIST consists of 28x28-pixel images categorized into 10 fashion-related classes, such as T-shirt/top, trouser, and ankle boot.

### How can I train a YOLO model on the Fashion-MNIST dataset?

To train an Ultralytics YOLO model on the Fashion-MNIST dataset, you can use both Python and CLI commands. Here's a quick example to get you started:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo11n-cls.pt")

        # Train the model on Fashion-MNIST
        results = model.train(data="fashion-mnist", epochs=100, imgsz=28)
        ```


    === "CLI"

        ```bash
        yolo classify train data=fashion-mnist model=yolo11n-cls.pt epochs=100 imgsz=28
        ```

For more detailed training parameters, refer to the [Training page](../../modes/train.md).

### Why should I use the Fashion-MNIST dataset for benchmarking my machine learning models?

The [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset is widely recognized in the [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) community as a robust alternative to MNIST. It offers a more complex and varied set of images, making it an excellent choice for benchmarking image classification models. The dataset's structure, comprising 60,000 training images and 10,000 testing images, each labeled with one of 10 classes, makes it ideal for evaluating the performance of different machine learning algorithms in a more challenging context.

### Can I use Ultralytics YOLO for image classification tasks like Fashion-MNIST?

Yes, Ultralytics YOLO models can be used for image classification tasks, including those involving the Fashion-MNIST dataset. YOLO11, for example, supports various vision tasks such as detection, segmentation, and classification. To get started with image classification tasks, refer to the [Classification page](https://docs.ultralytics.com/tasks/classify/).

### What are the key features and structure of the Fashion-MNIST dataset?

The Fashion-MNIST dataset is divided into two main subsets: 60,000 training images and 10,000 testing images. Each image is a 28x28-pixel grayscale picture representing one of 10 fashion-related classes. The simplicity and well-structured format make it ideal for training and evaluating models in machine learning and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks. For more details on the dataset structure, see the [Dataset Structure section](#dataset-structure).

### How can I acknowledge the use of the Fashion-MNIST dataset in my research?

If you utilize the Fashion-MNIST dataset in your research or development projects, it's important to acknowledge it by linking to the [GitHub repository](https://github.com/zalandoresearch/fashion-mnist). This helps in attributing the data to Zalando Research, who made the dataset available for public use.
