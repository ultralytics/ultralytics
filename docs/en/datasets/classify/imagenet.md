---
comments: true
description: Explore the extensive ImageNet dataset and discover its role in advancing deep learning in computer vision. Access pretrained models and training examples.
keywords: ImageNet, deep learning, visual recognition, computer vision, pretrained models, YOLO, dataset, object detection, image classification
---

# ImageNet Dataset

[ImageNet](https://www.image-net.org/) is a large-scale database of annotated images designed for use in visual object recognition research. It contains over 14 million images, with each image annotated using WordNet synsets, making it one of the most extensive resources available for training deep learning models in computer vision tasks.

## ImageNet Pretrained Models

| Model                                                                                        | size<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
|----------------------------------------------------------------------------------------------|-----------------------|------------------|------------------|--------------------------------|-------------------------------------|--------------------|--------------------------|
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-cls.pt) | 224                   | 69.0             | 88.3             | 12.9                           | 0.31                                | 2.7                | 4.3                      |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-cls.pt) | 224                   | 73.8             | 91.7             | 23.4                           | 0.35                                | 6.4                | 13.5                     |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-cls.pt) | 224                   | 76.8             | 93.5             | 85.4                           | 0.62                                | 17.0               | 42.7                     |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-cls.pt) | 224                   | 76.8             | 93.5             | 163.0                          | 0.87                                | 37.5               | 99.7                     |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-cls.pt) | 224                   | 79.0             | 94.6             | 232.0                          | 1.01                                | 57.4               | 154.8                    |

## Key Features

- ImageNet contains over 14 million high-resolution images spanning thousands of object categories.
- The dataset is organized according to the WordNet hierarchy, with each synset representing a category.
- ImageNet is widely used for training and benchmarking in the field of computer vision, particularly for image classification and object detection tasks.
- The annual ImageNet Large Scale Visual Recognition Challenge (ILSVRC) has been instrumental in advancing computer vision research.

## Dataset Structure

The ImageNet dataset is organized using the WordNet hierarchy. Each node in the hierarchy represents a category, and each category is described by a synset (a collection of synonymous terms). The images in ImageNet are annotated with one or more synsets, providing a rich resource for training models to recognize various objects and their relationships.

## ImageNet Large Scale Visual Recognition Challenge (ILSVRC)

The annual [ImageNet Large Scale Visual Recognition Challenge (ILSVRC)](https://image-net.org/challenges/LSVRC/) has been an important event in the field of computer vision. It has provided a platform for researchers and developers to evaluate their algorithms and models on a large-scale dataset with standardized evaluation metrics. The ILSVRC has led to significant advancements in the development of deep learning models for image classification, object detection, and other computer vision tasks.

## Applications

The ImageNet dataset is widely used for training and evaluating deep learning models in various computer vision tasks, such as image classification, object detection, and object localization. Some popular deep learning architectures, such as AlexNet, VGG, and ResNet, were developed and benchmarked using the ImageNet dataset.

## Usage

To train a deep learning model on the ImageNet dataset for 100 epochs with an image size of 224x224, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! Example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="imagenet", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=imagenet model=yolov8n-cls.pt epochs=100 imgsz=224
        ```

## Sample Images and Annotations

The ImageNet dataset contains high-resolution images spanning thousands of object categories, providing a diverse and extensive dataset for training and evaluating computer vision models. Here are some examples of images from the dataset:

![Dataset sample images](https://user-images.githubusercontent.com/26833433/239280348-3d8f30c7-6f05-4dda-9cfe-d62ad9faecc9.png)

The example showcases the variety and complexity of the images in the ImageNet dataset, highlighting the importance of a diverse dataset for training robust computer vision models.

## Citations and Acknowledgments

If you use the ImageNet dataset in your research or development work, please cite the following paper:

!!! Quote ""

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

## FAQ

### What is the ImageNet dataset and how is it used in computer vision?

The [ImageNet dataset](https://www.image-net.org/) is a large-scale database consisting of over 14 million high-resolution images categorized using WordNet synsets. It is extensively used in visual object recognition research, including image classification and object detection. The dataset's annotations and sheer volume provide a rich resource for training deep learning models. Notably, models like AlexNet, VGG, and ResNet have been trained and benchmarked using ImageNet, showcasing its role in advancing computer vision.

### How can I use a pretrained YOLO model for image classification on the ImageNet dataset?

To use a pretrained Ultralytics YOLO model for image classification on the ImageNet dataset, follow these steps:

!!! Example "Train Example"

    === "Python"
    
        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="imagenet", epochs=100, imgsz=224)
        ```

    === "CLI"
    
        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=imagenet model=yolov8n-cls.pt epochs=100 imgsz=224
        ```

For more in-depth training instruction, refer to our [Training page](../../modes/train.md).

### Why should I use the Ultralytics YOLOv8 pretrained models for my ImageNet dataset projects?

Ultralytics YOLOv8 pretrained models offer state-of-the-art performance in terms of speed and accuracy for various computer vision tasks. For example, the YOLOv8n-cls model, with a top-1 accuracy of 69.0% and a top-5 accuracy of 88.3%, is optimized for real-time applications. Pretrained models reduce the computational resources required for training from scratch and accelerate development cycles. Learn more about the performance metrics of YOLOv8 models in the [ImageNet Pretrained Models section](#imagenet-pretrained-models).

### How is the ImageNet dataset structured, and why is it important?

The ImageNet dataset is organized using the WordNet hierarchy, where each node in the hierarchy represents a category described by a synset (a collection of synonymous terms). This structure allows for detailed annotations, making it ideal for training models to recognize a wide variety of objects. The diversity and annotation richness of ImageNet make it a valuable dataset for developing robust and generalizable deep learning models. More about this organization can be found in the [Dataset Structure](#dataset-structure) section.

### What role does the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) play in computer vision?

The annual [ImageNet Large Scale Visual Recognition Challenge (ILSVRC)](https://image-net.org/challenges/LSVRC/) has been pivotal in driving advancements in computer vision by providing a competitive platform for evaluating algorithms on a large-scale, standardized dataset. It offers standardized evaluation metrics, fostering innovation and development in areas such as image classification, object detection, and image segmentation. The challenge has continuously pushed the boundaries of what is possible with deep learning and computer vision technologies.
