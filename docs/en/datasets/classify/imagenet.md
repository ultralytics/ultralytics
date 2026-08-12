---
title: ImageNet Image Classification Dataset
comments: true
creator:
    name: ImageNet
    url: https://www.image-net.org/
license:
    name: Research-Only
    url: https://www.image-net.org/download.php
description: "ImageNet (ILSVRC-2012) image classification dataset: 1,000 classes, 1.28M training images. Train Ultralytics YOLO classification models with data=imagenet."
keywords: ImageNet, ILSVRC-2012, image classification, deep learning, computer vision, pretrained models, YOLO, dataset, WordNet
---

# ImageNet Dataset

The **Ultralytics ImageNet** dataset (`data="imagenet"`) is the ImageNet-1k / ILSVRC-2012 subset used to train and benchmark [image classification](../../tasks/classify.md) models. It contains **1,000 object classes** with **1,281,167 training images** and **50,000 validation images** at a **224x224** image size, and downloads to roughly **144 GB** of data. The broader [ImageNet](https://www.image-net.org/) database is far larger — over 14 million high-resolution images annotated with WordNet synsets across more than 20,000 categories — but Ultralytics trains on the standardized 1,000-class ILSVRC subset that became the de-facto benchmark for [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) in [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv).

## ImageNet Pretrained Models

{% include "macros/yolo-cls-perf.md" %}

## Key Features

- The Ultralytics `imagenet` dataset provides 1,000 classes with 1,281,167 training and 50,000 validation images (ILSVRC-2012), the standard pretraining benchmark for image classification.
- Classes are organized according to the WordNet hierarchy, where each class corresponds to a synset (a set of synonymous terms).
- Images are trained at 224x224, and the full dataset is a large ~144 GB download.
- The annual ImageNet Large Scale Visual Recognition Challenge (ILSVRC) has been instrumental in advancing computer vision research.

## Dataset Structure

The Ultralytics ImageNet dataset uses the ILSVRC-2012 split:

| Split      | Images    | Classes |
| ---------- | --------- | ------- |
| Train      | 1,281,167 | 1,000   |
| Validation | 50,000    | 1,000   |

Images are stored in per-class folders named by WordNet synset ID (for example, `n01440764`), the layout Ultralytics classification training expects. Each of the 1,000 classes maps to a WordNet synset, and there is no separate test split, so the 50,000-image validation set is used to measure [accuracy](https://www.ultralytics.com/glossary/accuracy).

!!! note "Download size"

    ImageNet-1k is a ~144 GB download, so make sure you have enough disk space before training. For quick experiments, the smaller [ImageNette](imagenette.md) and [ImageNet10](imagenet10.md) subsets use the same folder format and train in a fraction of the time.

## ImageNet Large Scale Visual Recognition Challenge (ILSVRC)

The annual [ImageNet Large Scale Visual Recognition Challenge (ILSVRC)](https://image-net.org/challenges/LSVRC/) let researchers benchmark algorithms on a large-scale, standardized dataset with consistent evaluation metrics. It drove major advances in deep learning for image classification, [object detection](https://www.ultralytics.com/glossary/object-detection), and other vision tasks — most notably AlexNet's 2012 win, which helped launch the modern deep-learning era.

## Applications

The ImageNet dataset is widely used to train and evaluate deep learning models for image classification, object detection, and object localization. Landmark architectures such as [AlexNet](https://en.wikipedia.org/wiki/AlexNet), [VGG](https://arxiv.org/abs/1409.1556), and [ResNet](https://arxiv.org/abs/1512.03385) were all developed and benchmarked on ImageNet, and ImageNet-pretrained weights remain a common starting point for transfer learning across vision tasks.

## Usage

To train a YOLO classification model on ImageNet for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at a 224x224 image size, use the code snippets below. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="imagenet", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=imagenet model=yolo26n-cls.pt epochs=100 imgsz=224
        ```

You can also manage classification datasets and run training in the cloud with [Ultralytics Platform](https://platform.ultralytics.com/).

## Sample Images and Annotations

The ImageNet dataset spans the 1,000 ILSVRC-2012 classes, providing a diverse and extensive resource for training and evaluating computer vision models. Here are some example images from the dataset:

![ImageNet classification dataset sample images](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/imagenet-sample-images.avif)

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

The [ImageNet dataset](https://www.image-net.org/) is a large-scale image database whose broader collection holds over 14 million high-resolution images annotated with WordNet synsets. In Ultralytics, `data="imagenet"` trains on the standardized 1,000-class ILSVRC-2012 subset, which is the de-facto benchmark for [image classification](https://www.ultralytics.com/glossary/image-classification) pretraining. Landmark models such as AlexNet, VGG, and ResNet were trained and benchmarked on ImageNet, underscoring its role in advancing computer vision.

### How many classes and images does the ImageNet dataset have?

The Ultralytics `imagenet` dataset uses the ILSVRC-2012 subset with **1,000 classes**, **1,281,167 training images**, and **50,000 validation images** at a 224x224 image size, for a total download of roughly 144 GB. The full ImageNet database is much larger (over 14 million images across more than 20,000 WordNet synsets), but the 1,000-class subset is the one used for classification training and benchmarking.

### How can I train a YOLO model for image classification on the ImageNet dataset?

To train an Ultralytics YOLO model on ImageNet, load a pretrained classification model and point `data` at `imagenet`:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="imagenet", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=imagenet model=yolo26n-cls.pt epochs=100 imgsz=224
        ```

For more in-depth training instruction, refer to our [Training page](../../modes/train.md).

### Why should I use the Ultralytics YOLO26 pretrained models for my ImageNet dataset projects?

Ultralytics YOLO26 pretrained models offer state-of-the-art performance in terms of speed and [accuracy](https://www.ultralytics.com/glossary/accuracy) for various computer vision tasks. For example, the YOLO26n-cls model, with a top-1 accuracy of 71.4% and a top-5 accuracy of 90.1%, is optimized for real-time applications. Pretrained models reduce the computational resources required for training from scratch and accelerate development cycles. Learn more about the performance metrics of YOLO26 models in the [ImageNet Pretrained Models section](#imagenet-pretrained-models).

### What role does the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) play in computer vision?

The annual [ImageNet Large Scale Visual Recognition Challenge (ILSVRC)](https://image-net.org/challenges/LSVRC/) drove advances in computer vision by providing a competitive platform for evaluating algorithms on a large-scale, standardized dataset. Its consistent evaluation metrics fostered innovation in image classification, object detection, and [image segmentation](https://www.ultralytics.com/glossary/image-segmentation), continuously pushing the boundaries of deep learning and computer vision.
