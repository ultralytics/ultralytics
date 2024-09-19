---
comments: true
description: Explore the ImageWoof dataset, a challenging subset of ImageNet focusing on 10 dog breeds, designed to enhance image classification models. Learn more on Ultralytics Docs.
keywords: ImageWoof dataset, ImageNet subset, dog breeds, image classification, deep learning, machine learning, Ultralytics, training dataset, noisy labels
---

# ImageWoof Dataset

The [ImageWoof](https://github.com/fastai/imagenette) dataset is a subset of the ImageNet consisting of 10 classes that are challenging to classify, since they're all dog breeds. It was created as a more difficult task for image classification algorithms to solve, aiming at encouraging development of more advanced models.

## Key Features

- ImageWoof contains images of 10 different dog breeds: Australian terrier, Border terrier, Samoyed, Beagle, Shih-Tzu, English foxhound, Rhodesian ridgeback, Dingo, Golden retriever, and Old English sheepdog.
- The dataset provides images at various resolutions (full size, 320px, 160px), accommodating for different computational capabilities and research needs.
- It also includes a version with noisy labels, providing a more realistic scenario where labels might not always be reliable.

## Dataset Structure

The ImageWoof dataset structure is based on the dog breed classes, with each breed having its own directory of images.

## Applications

The ImageWoof dataset is widely used for training and evaluating deep learning models in image classification tasks, especially when it comes to more complex and similar classes. The dataset's challenge lies in the subtle differences between the dog breeds, pushing the limits of model's performance and generalization.

## Usage

To train a CNN model on the ImageWoof dataset for 100 epochs with an image size of 224x224, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="imagewoof", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=imagewoof model=yolov8n-cls.pt epochs=100 imgsz=224
        ```

## Dataset Variants

ImageWoof dataset comes in three different sizes to accommodate various research needs and computational capabilities:

1. **Full Size (imagewoof)**: This is the original version of the ImageWoof dataset. It contains full-sized images and is ideal for final training and performance benchmarking.

2. **Medium Size (imagewoof320)**: This version contains images resized to have a maximum edge length of 320 pixels. It's suitable for faster training without significantly sacrificing model performance.

3. **Small Size (imagewoof160)**: This version contains images resized to have a maximum edge length of 160 pixels. It's designed for rapid prototyping and experimentation where training speed is a priority.

To use these variants in your training, simply replace 'imagewoof' in the dataset argument with 'imagewoof320' or 'imagewoof160'. For example:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

        # For medium-sized dataset
        model.train(data="imagewoof320", epochs=100, imgsz=224)

        # For small-sized dataset
        model.train(data="imagewoof160", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Load a pretrained model and train on the small-sized dataset
        yolo classify train model=yolov8n-cls.pt data=imagewoof320 epochs=100 imgsz=224
        ```

It's important to note that using smaller images will likely yield lower performance in terms of classification accuracy. However, it's an excellent way to iterate quickly in the early stages of model development and prototyping.

## Sample Images and Annotations

The ImageWoof dataset contains colorful images of various dog breeds, providing a challenging dataset for image classification tasks. Here are some examples of images from the dataset:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/imagewoof-dataset-sample.avif)

The example showcases the subtle differences and similarities among the different dog breeds in the ImageWoof dataset, highlighting the complexity and difficulty of the classification task.

## Citations and Acknowledgments

If you use the ImageWoof dataset in your research or development work, please make sure to acknowledge the creators of the dataset by linking to the [official dataset repository](https://github.com/fastai/imagenette).

We would like to acknowledge the FastAI team for creating and maintaining the ImageWoof dataset as a valuable resource for the machine learning and computer vision research community. For more information about the ImageWoof dataset, visit the [ImageWoof dataset repository](https://github.com/fastai/imagenette).

## FAQ

### What is the ImageWoof dataset in Ultralytics?

The [ImageWoof](https://github.com/fastai/imagenette) dataset is a challenging subset of ImageNet focusing on 10 specific dog breeds. Created to push the limits of image classification models, it features breeds like Beagle, Shih-Tzu, and Golden Retriever. The dataset includes images at various resolutions (full size, 320px, 160px) and even noisy labels for more realistic training scenarios. This complexity makes ImageWoof ideal for developing more advanced deep learning models.

### How can I train a model using the ImageWoof dataset with Ultralytics YOLO?

To train a Convolutional Neural Network (CNN) model on the ImageWoof dataset using Ultralytics YOLO for 100 epochs at an image size of 224x224, you can use the following code:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolov8n-cls.pt")  # Load a pretrained model
        results = model.train(data="imagewoof", epochs=100, imgsz=224)
        ```


    === "CLI"

        ```bash
        yolo classify train data=imagewoof model=yolov8n-cls.pt epochs=100 imgsz=224
        ```

For more details on available training arguments, refer to the [Training](../../modes/train.md) page.

### What versions of the ImageWoof dataset are available?

The ImageWoof dataset comes in three sizes:

1. **Full Size (imagewoof)**: Ideal for final training and benchmarking, containing full-sized images.
2. **Medium Size (imagewoof320)**: Resized images with a maximum edge length of 320 pixels, suited for faster training.
3. **Small Size (imagewoof160)**: Resized images with a maximum edge length of 160 pixels, perfect for rapid prototyping.

Use these versions by replacing 'imagewoof' in the dataset argument accordingly. Note, however, that smaller images may yield lower classification accuracy but can be useful for quicker iterations.

### How do noisy labels in the ImageWoof dataset benefit training?

Noisy labels in the ImageWoof dataset simulate real-world conditions where labels might not always be accurate. Training models with this data helps develop robustness and generalization in image classification tasks. This prepares the models to handle ambiguous or mislabeled data effectively, which is often encountered in practical applications.

### What are the key challenges of using the ImageWoof dataset?

The primary challenge of the ImageWoof dataset lies in the subtle differences among the dog breeds it includes. Since it focuses on 10 closely related breeds, distinguishing between them requires more advanced and fine-tuned image classification models. This makes ImageWoof an excellent benchmark to test the capabilities and improvements of deep learning models.
