---
comments: true
description: Learn about the ImageNette dataset and its usage in deep learning model training. Find code snippets for model training and explore ImageNette datatypes.
keywords: ImageNette dataset, Ultralytics, YOLO, Image classification, Machine Learning, Deep learning, Training code snippets, CNN, ImageNette160, ImageNette320
---

# ImageNette Dataset

The [ImageNette](https://github.com/fastai/imagenette) dataset is a subset of the larger [Imagenet](https://www.image-net.org/) dataset, but it only includes 10 easily distinguishable classes. It was created to provide a quicker, easier-to-use version of Imagenet for software development and education.

## Key Features

- ImageNette contains images from 10 different classes such as tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute.
- The dataset comprises colored images of varying dimensions.
- ImageNette is widely used for training and testing in the field of machine learning, especially for image classification tasks.

## Dataset Structure

The ImageNette dataset is split into two subsets:

1. **Training Set**: This subset contains several thousands of images used for training machine learning models. The exact number varies per class.
2. **Validation Set**: This subset consists of several hundreds of images used for validating and benchmarking the trained models. Again, the exact number varies per class.

## Applications

The ImageNette dataset is widely used for training and evaluating deep learning models in image classification tasks, such as Convolutional Neural Networks (CNNs), and various other machine learning algorithms. The dataset's straightforward format and well-chosen classes make it a handy resource for both beginner and experienced practitioners in the field of machine learning and computer vision.

## Usage

To train a model on the ImageNette dataset for 100 epochs with a standard image size of 224x224, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! Example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data='imagenette', epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=imagenette model=yolov8n-cls.pt epochs=100 imgsz=224
        ```

## Sample Images and Annotations

The ImageNette dataset contains colored images of various objects and scenes, providing a diverse dataset for image classification tasks. Here are some examples of images from the dataset:

![Dataset sample image](https://docs.fast.ai/22_tutorial.imagenette_files/figure-html/cell-21-output-1.png)

The example showcases the variety and complexity of the images in the ImageNette dataset, highlighting the importance of a diverse dataset for training robust image classification models.

## ImageNette160 and ImageNette320

For faster prototyping and training, the ImageNette dataset is also available in two reduced sizes: ImageNette160 and ImageNette320. These datasets maintain the same classes and structure as the full ImageNette dataset, but the images are resized to a smaller dimension. As such, these versions of the dataset are particularly useful for preliminary model testing, or when computational resources are limited.

To use these datasets, simply replace 'imagenette' with 'imagenette160' or 'imagenette320' in the training command. The following code snippets illustrate this:

!!! Example "Train Example with ImageNette160"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

        # Train the model with ImageNette160
        results = model.train(data='imagenette160', epochs=100, imgsz=160)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model with ImageNette160
        yolo detect train data=imagenette160 model=yolov8n-cls.pt epochs=100 imgsz=160
        ```

!!! Example "Train Example with ImageNette320"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

        # Train the model with ImageNette320
        results = model.train(data='imagenette320', epochs=100, imgsz=320)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model with ImageNette320
        yolo detect train data=imagenette320 model=yolov8n-cls.pt epochs=100 imgsz=320
        ```

These smaller versions of the dataset allow for rapid iterations during the development process while still providing valuable and realistic image classification tasks.

## Citations and Acknowledgments

If you use the ImageNette dataset in your research or development work, please acknowledge it appropriately. For more information about the ImageNette dataset, visit the [ImageNette dataset GitHub page](https://github.com/fastai/imagenette).
