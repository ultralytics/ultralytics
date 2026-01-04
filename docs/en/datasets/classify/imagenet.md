---
comments: true
description: Explore the extensive ImageNet dataset and discover its role in advancing deep learning in computer vision. Access pretrained models and training examples.
keywords: ImageNet, deep learning, visual recognition, computer vision, pretrained models, YOLO, dataset, object detection, image classification
---

# ImageNet Dataset

[ImageNet](https://www.image-net.org/) is a large-scale database of annotated images designed for use in visual object recognition research. It contains over 14 million images, with each image annotated using WordNet synsets, making it one of the most extensive resources available for training [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks.

## ImageNet Pretrained Models

{% include "macros/yolo-cls-perf.md" %}

## Key Features

- ImageNet contains over 14 million high-resolution images spanning thousands of object categories.
- The dataset is organized according to the WordNet hierarchy, with each synset representing a category.
- ImageNet is widely used for training and benchmarking in the field of computer vision, particularly for [image classification](https://www.ultralytics.com/glossary/image-classification) and [object detection](https://www.ultralytics.com/glossary/object-detection) tasks.
- The annual ImageNet Large Scale Visual Recognition Challenge (ILSVRC) has been instrumental in advancing computer vision research.

## Dataset Structure

The ImageNet dataset is organized using the WordNet hierarchy. Each node in the hierarchy represents a category, and each category is described by a synset (a collection of synonymous terms). The images in ImageNet are annotated with one or more synsets, providing a rich resource for training models to recognize various objects and their relationships.

## ImageNet Large Scale Visual Recognition Challenge (ILSVRC)

The annual [ImageNet Large Scale Visual Recognition Challenge (ILSVRC)](https://image-net.org/challenges/LSVRC/) has been an important event in the field of computer vision. It has provided a platform for researchers and developers to evaluate their algorithms and models on a large-scale dataset with standardized evaluation metrics. The ILSVRC has led to significant advancements in the development of deep learning models for image classification, object detection, and other computer vision tasks.

## Applications

The ImageNet dataset is widely used for training and evaluating deep learning models in various computer vision tasks, such as image classification, object detection, and object localization. Some popular deep learning architectures, such as [AlexNet](https://en.wikipedia.org/wiki/AlexNet), [VGG](https://arxiv.org/abs/1409.1556), and [ResNet](https://arxiv.org/abs/1512.03385), were developed and benchmarked using the ImageNet dataset.

## Usage

To train a deep learning model on the ImageNet dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 224x224, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="imagenet", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=imagenet model=yolo11n-cls.pt epochs=100 imgsz=224
        ```

## Sample Images and Annotations

The ImageNet dataset contains high-resolution images spanning thousands of object categories, providing a diverse and extensive dataset for training and evaluating computer vision models. Here are some examples of images from the dataset:

![Dataset sample images](https://github.com/ultralytics/docs/releases/download/0/imagenet-sample-images.avif)

The example showcases the variety and complexity of the images in the ImageNet dataset, highlighting the importance of a diverse dataset for training robust computer vision models.

## Citations and Acknowledgments

If you use the ImageNet dataset in your research or development work, please cite the following paper:

!!! quote ""

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

We would like to acknowledge the ImageNet team, led by Olga Russakovsky, Jia Deng, and Li Fei-Fei, for creating and maintaining the ImageNet dataset as a valuable resource for the [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and computer vision research community. For more information about the ImageNet dataset and its creators, visit the [ImageNet website](https://www.image-net.org/).

## FAQ

### What is the ImageNet dataset and how is it used in computer vision?

The [ImageNet dataset](https://www.image-net.org/) is a large-scale database consisting of over 14 million high-resolution images categorized using WordNet synsets. It is extensively used in visual object recognition research, including image classification and object detection. The dataset's annotations and sheer volume provide a rich resource for training deep learning models. Notably, models like AlexNet, VGG, and ResNet have been trained and benchmarked using ImageNet, showcasing its role in advancing computer vision.

### How can I use a pretrained YOLO model for image classification on the ImageNet dataset?

To use a pretrained Ultralytics YOLO model for image classification on the ImageNet dataset, follow these steps:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="imagenet", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=imagenet model=yolo11n-cls.pt epochs=100 imgsz=224
        ```

For more in-depth training instruction, refer to our [Training page](../../modes/train.md).

### Why should I use the Ultralytics YOLO11 pretrained models for my ImageNet dataset projects?

Ultralytics YOLO11 pretrained models offer state-of-the-art performance in terms of speed and [accuracy](https://www.ultralytics.com/glossary/accuracy) for various computer vision tasks. For example, the YOLO11n-cls model, with a top-1 accuracy of 70.0% and a top-5 accuracy of 89.4%, is optimized for real-time applications. Pretrained models reduce the computational resources required for training from scratch and accelerate development cycles. Learn more about the performance metrics of YOLO11 models in the [ImageNet Pretrained Models section](#imagenet-pretrained-models).

### How is the ImageNet dataset structured, and why is it important?

The ImageNet dataset is organized using the WordNet hierarchy, where each node in the hierarchy represents a category described by a synset (a collection of synonymous terms). This structure allows for detailed annotations, making it ideal for training models to recognize a wide variety of objects. The diversity and annotation richness of ImageNet make it a valuable dataset for developing robust and generalizable deep learning models. More about this organization can be found in the [Dataset Structure](#dataset-structure) section.

### What role does the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) play in computer vision?

The annual [ImageNet Large Scale Visual Recognition Challenge (ILSVRC)](https://image-net.org/challenges/LSVRC/) has been pivotal in driving advancements in computer vision by providing a competitive platform for evaluating algorithms on a large-scale, standardized dataset. It offers standardized evaluation metrics, fostering innovation and development in areas such as image classification, object detection, and [image segmentation](https://www.ultralytics.com/glossary/image-segmentation). The challenge has continuously pushed the boundaries of what is possible with deep learning and computer vision technologies.
