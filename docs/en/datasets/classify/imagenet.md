---
comments: true
description: Understand how to use ImageNet, an extensive annotated image dataset for object recognition research, with Ultralytics YOLO models. Learn about its structure, usage, and significance in computer vision.
keywords: Ultralytics, YOLO, ImageNet, dataset, object recognition, deep learning, computer vision, machine learning, dataset training, model training, image classification, object detection
---

# ImageNet Dataset

[ImageNet](https://www.image-net.org/) is a large-scale database of annotated images designed for use in visual object recognition research. It contains over 14 million images, with each image annotated using WordNet synsets, making it one of the most extensive resources available for training deep learning models in computer vision tasks.

## Key Features

- ImageNet contains over 14 million high-resolution images spanning thousands of object categories.
- The dataset is organized according to the WordNet hierarchy, with each synset representing a category.
- ImageNet is widely used for training and benchmarking in the field of computer vision, particularly for image classification and object detection tasks.
- The annual ImageNet Large Scale Visual Recognition Challenge (ILSVRC) has been instrumental in advancing computer vision research.

## Dataset Structure

The ImageNet dataset is organized using the WordNet hierarchy. Each node in the hierarchy represents a category, and each category is described by a synset (a collection of synonymous terms). The images in ImageNet are annotated with one or more synsets, providing a rich resource for training models to recognize various objects and their relationships.

## ImageNet Large Scale Visual Recognition Challenge (ILSVRC)

The annual [ImageNet Large Scale Visual Recognition Challenge (ILSVRC)](http://image-net.org/challenges/LSVRC/) has been an important event in the field of computer vision. It has provided a platform for researchers and developers to evaluate their algorithms and models on a large-scale dataset with standardized evaluation metrics. The ILSVRC has led to significant advancements in the development of deep learning models for image classification, object detection, and other computer vision tasks.

## Applications

The ImageNet dataset is widely used for training and evaluating deep learning models in various computer vision tasks, such as image classification, object detection, and object localization. Some popular deep learning architectures, such as AlexNet, VGG, and ResNet, were developed and benchmarked using the ImageNet dataset.

## Usage

To train a deep learning model on the ImageNet dataset for 100 epochs with an image size of 224x224, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! Example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data='imagenet', epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo train data=imagenet model=yolov8n-cls.pt epochs=100 imgsz=224
        ```

## Sample Images and Annotations

The ImageNet dataset contains high-resolution images spanning thousands of object categories, providing a diverse and extensive dataset for training and evaluating computer vision models. Here are some examples of images from the dataset:

![Dataset sample images](https://user-images.githubusercontent.com/26833433/239280348-3d8f30c7-6f05-4dda-9cfe-d62ad9faecc9.png)

The example showcases the variety and complexity of the images in the ImageNet dataset, highlighting the importance of a diverse dataset for training robust computer vision models.

## Citations and Acknowledgments

If you use the ImageNet dataset in your research or development work, please cite the following paper:

!!! Note ""

    === "BibTeX"

        ```bibtex
        @article{ILSVRC15,
                 author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
                 title={ImageNet Large Scale Visual Recognition Challenge},
                 year={2015},
                 journal={International Journal of Computer Vision (IJCV)},
                 volume={115},
                 number={3},
                 pages={211-252}
        }
        ```

We would like to acknowledge the ImageNet team, led by Olga Russakovsky, Jia Deng, and Li Fei-Fei, for creating and maintaining the ImageNet dataset as a valuable resource for the machine learning and computer vision research community. For more information about the ImageNet dataset and its creators, visit the [ImageNet website](https://www.image-net.org/).
