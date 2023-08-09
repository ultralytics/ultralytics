---
comments: true
description: Explore the ImageWoof dataset, designed for challenging dog breed classification. Train AI models with Ultralytics YOLO using this dataset.
keywords: ImageWoof, image classification, dog breeds, machine learning, deep learning, Ultralytics, YOLO, dataset
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
        model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

        # Train the model
        model.train(data='imagewoof', epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=imagewoof model=yolov8n-cls.pt epochs=100 imgsz=224
        ```

## Dataset Variants

ImageWoof dataset comes in three different sizes to accommodate various research needs and computational capabilities:

1. **Full Size (imagewoof)**: This is the original version of the ImageWoof dataset. It contains full-sized images and is ideal for final training and performance benchmarking.

2. **Medium Size (imagewoof320)**: This version contains images resized to have a maximum edge length of 320 pixels. It's suitable for faster training without significantly sacrificing model performance.

3. **Small Size (imagewoof160)**: This version contains images resized to have a maximum edge length of 160 pixels. It's designed for rapid prototyping and experimentation where training speed is a priority.

To use these variants in your training, simply replace 'imagewoof' in the dataset argument with 'imagewoof320' or 'imagewoof160'. For example:

```python
# For medium-sized dataset
model.train(data='imagewoof320', epochs=100, imgsz=224)

# For small-sized dataset
model.train(data='imagewoof160', epochs=100, imgsz=224)
```

It's important to note that using smaller images will likely yield lower performance in terms of classification accuracy. However, it's an excellent way to iterate quickly in the early stages of model development and prototyping.

## Sample Images and Annotations

The ImageWoof dataset contains colorful images of various dog breeds, providing a challenging dataset for image classification tasks. Here are some examples of images from the dataset:

![Dataset sample image](https://user-images.githubusercontent.com/26833433/239357533-ec833254-4351-491b-8cb3-59578ea5d0b2.png)

The example showcases the subtle differences and similarities among the different dog breeds in the ImageWoof dataset, highlighting the complexity and difficulty of the classification task.

## Citations and Acknowledgments

If you use the ImageWoof dataset in your research or development work, please make sure to acknowledge the creators of the dataset by linking to the [official dataset repository](https://github.com/fastai/imagenette). As of my knowledge cutoff in September 2021, there is no official publication specifically about ImageWoof for citation.

We would like to acknowledge the FastAI team for creating and maintaining the ImageWoof dataset as a valuable resource for the machine learning and computer vision research community. For more information about the ImageWoof dataset, visit the [ImageWoof dataset repository](https://github.com/fastai/imagenette).
