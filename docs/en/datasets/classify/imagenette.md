---
comments: true
description: Explore the ImageNette dataset, a subset of ImageNet with 10 classes for efficient training and evaluation of image classification models. Ideal for ML and CV projects.
keywords: ImageNette dataset, ImageNet subset, image classification, machine learning, deep learning, YOLO, Convolutional Neural Networks, ML dataset, education, training
---

# ImageNette Dataset

The [ImageNette](https://github.com/fastai/imagenette) dataset is a subset of the larger [ImageNet](https://www.image-net.org/) dataset, but it only includes 10 easily distinguishable classes. It was created to provide a quicker, easier-to-use version of ImageNet for software development and education.

## Key Features

- ImageNette contains images from 10 different classes such as tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute.
- The dataset comprises colored images of varying dimensions.
- ImageNette is widely used for training and testing in the field of machine learning, especially for image classification tasks.

## Dataset Structure

The ImageNette dataset is split into two subsets:

1. **Training Set**: This subset contains several thousands of images used for training machine learning models. The exact number varies per class.
2. **Validation Set**: This subset consists of several hundreds of images used for validating and benchmarking the trained models. Again, the exact number varies per class.

## Applications

The ImageNette dataset is widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in image classification tasks, such as [Convolutional Neural Networks](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) (CNNs), and various other machine learning algorithms. The dataset's straightforward format and well-chosen classes make it a handy resource for both beginner and experienced practitioners in the field of [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv).

## Usage

To train a model on the ImageNette dataset for 100 epochs with a standard image size of 224x224, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="imagenette", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=imagenette model=yolo26n-cls.pt epochs=100 imgsz=224
        ```

## Sample Images and Annotations

The ImageNette dataset contains colored images of various objects and scenes, providing a diverse dataset for [image classification](https://www.ultralytics.com/glossary/image-classification) tasks. Here are some examples of images from the dataset:

![ImageNette classification dataset sample images](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/imagenette-sample-image.avif)

The example showcases the variety and complexity of the images in the ImageNette dataset, highlighting the importance of a diverse dataset for training robust image classification models.

## ImageNette160 and ImageNette320

For faster prototyping and training, the ImageNette dataset is also available in two reduced sizes: [ImageNette160](https://github.com/fastai/imagenette) and [ImageNette320](https://github.com/fastai/imagenette). These datasets maintain the same classes and structure as the full ImageNette dataset, but the images are resized to a smaller dimension. As such, these versions of the dataset are particularly useful for preliminary model testing, or when computational resources are limited.

To use these datasets, simply replace 'imagenette' with 'imagenette160' or 'imagenette320' in the training command. The following code snippets illustrate this:

!!! example "Train Example with ImageNette160"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model with ImageNette160
        results = model.train(data="imagenette160", epochs=100, imgsz=160)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model with ImageNette160
        yolo classify train data=imagenette160 model=yolo26n-cls.pt epochs=100 imgsz=160
        ```

!!! example "Train Example with ImageNette320"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model with ImageNette320
        results = model.train(data="imagenette320", epochs=100, imgsz=320)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model with ImageNette320
        yolo classify train data=imagenette320 model=yolo26n-cls.pt epochs=100 imgsz=320
        ```

These smaller versions of the dataset allow for rapid iterations during the development process while still providing valuable and realistic image classification tasks.

## Citations and Acknowledgments

If you use the ImageNette dataset in your research or development work, please acknowledge it appropriately. For more information about the ImageNette dataset, visit the [ImageNette dataset GitHub page](https://github.com/fastai/imagenette).

## FAQ

### What is the ImageNette dataset?

The [ImageNette dataset](https://github.com/fastai/imagenette) is a simplified subset of the larger [ImageNet dataset](https://www.image-net.org/), featuring only 10 easily distinguishable classes such as tench, English springer, and French horn. It was created to offer a more manageable dataset for efficient training and evaluation of image classification models. This dataset is particularly useful for quick software development and educational purposes in [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and computer vision.

### How can I use the ImageNette dataset for training a YOLO model?

To train a YOLO model on the ImageNette dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch), you can use the following commands. Make sure to have the Ultralytics YOLO environment set up.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="imagenette", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=imagenette model=yolo26n-cls.pt epochs=100 imgsz=224
        ```

For more details, see the [Training](../../modes/train.md) documentation page.

### Why should I use ImageNette for image classification tasks?

The ImageNette dataset is advantageous for several reasons:

- **Quick and Simple**: It contains only 10 classes, making it less complex and time-consuming compared to larger datasets.
- **Educational Use**: Ideal for learning and teaching the basics of image classification since it requires less computational power and time.
- **Versatility**: Widely used to train and benchmark various machine learning models, especially in image classification.

For more details on model training and dataset management, explore the [Dataset Structure](#dataset-structure) section.

### Can the ImageNette dataset be used with different image sizes?

Yes, the ImageNette dataset is also available in two resized versions: ImageNette160 and ImageNette320. These versions help in faster prototyping and are especially useful when computational resources are limited.

!!! example "Train Example with ImageNette160"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")

        # Train the model with ImageNette160
        results = model.train(data="imagenette160", epochs=100, imgsz=160)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model with ImageNette160
        yolo classify train data=imagenette160 model=yolo26n-cls.pt epochs=100 imgsz=160
        ```

For more information, refer to [Training with ImageNette160 and ImageNette320](#imagenette160-and-imagenette320).

### What are some practical applications of the ImageNette dataset?

The ImageNette dataset is extensively used in:

- **Educational Settings**: To educate beginners in machine learning and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv).
- **Software Development**: For rapid prototyping and development of image classification models.
- **Deep Learning Research**: To evaluate and benchmark the performance of various deep learning models, especially Convolutional [Neural Networks](https://www.ultralytics.com/glossary/neural-network-nn) (CNNs).

Explore the [Applications](#applications) section for detailed use cases.
