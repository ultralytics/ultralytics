---
title: Objects365 Detection Dataset
comments: true
creator:
    name: Objects365 Consortium
    url: https://www.objects365.org/
license:
    name: CC-BY-4.0
    url: https://www.objects365.org/download.html
description: Objects365 is a large-scale object detection dataset with 1,742,289 training and 80,000 validation images across 365 classes. Train Ultralytics YOLO on it.
keywords: Objects365 dataset, object detection, Megvii, large-scale dataset, model pretraining, computer vision, annotated images, bounding boxes, YOLO26, 365 classes
---

# Objects365 Dataset

The [Objects365](https://www.objects365.org/) dataset is a large-scale [object detection](../../tasks/detect.md) benchmark with 1,742,289 training images and 80,000 validation images spanning 365 object classes — from people, cars, and chairs to bottles, dogs, and street lights. Created by [Megvii](https://en.megvii.com/) researchers and presented at ICCV 2019, it focuses on diverse objects in the wild and is widely used to pretrain [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) [models](../../models/index.md) that generalize better than ImageNet-pretrained ones.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/J-RH22rwx1A"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train Ultralytics YOLO on the Objects365 Dataset
</p>

## Key Features

- Objects365 defines 365 object classes, and the upstream release reports around 2 million images with 30 million bounding boxes in total.
- The dataset includes diverse objects in various real-world scenarios, providing a rich and challenging benchmark for object detection tasks.
- Annotations include bounding boxes for objects, making it suitable for training and evaluating object detection models.
- According to the [ICCV 2019 paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shao_Objects365_A_Large-Scale_High-Quality_Dataset_for_Object_Detection_ICCV_2019_paper.pdf), Objects365 pretraining outperforms [ImageNet](../classify/imagenet.md) pretraining by 5.6 points (42.0 vs 36.4 mAP) on the COCO benchmark.

## Dataset Structure

The Ultralytics `Objects365.yaml` configuration defines two splits:

| Split      | Images    | Description                                     |
| ---------- | --------- | ----------------------------------------------- |
| Train      | 1,742,289 | Labeled images for model training               |
| Validation | 80,000    | Held-out images for evaluation and benchmarking |

The download retrieves the train and validation splits — 1,822,289 images in total — and the `test:` key in the configuration is left empty.

## Applications

The Objects365 dataset supports a wide range of [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) applications in object detection:

- **Pretraining detection backbones**: With 365 classes and dense box annotations, Objects365 pretraining improves downstream fine-tuning on smaller datasets such as [COCO](coco.md) and [VOC](voc.md).
- **Retail and inventory recognition**: Hundreds of everyday categories — bottles, cups, sneakers, handbags — support shelf monitoring and automated checkout systems.
- **Robotics and smart environments**: Broad household and street-object coverage helps robots and smart cameras recognize objects in unstructured scenes.
- **Detector benchmarking**: The long class list and in-the-wild imagery make it a demanding benchmark for evaluating how well detection models generalize.

To label your own images, train, and manage large-scale datasets in your browser, run the full workflow with [Ultralytics Platform](https://platform.ultralytics.com/).

## Dataset YAML

The `Objects365.yaml` file defines the dataset configuration — the dataset paths, class names, and other metadata. It is maintained in the Ultralytics repository at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Objects365.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Objects365.yaml).

!!! example "ultralytics/cfg/datasets/Objects365.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/Objects365.yaml"
    ```

## Usage

!!! warning "712 GB download"

    Objects365 downloads automatically on first use and requires about 712 GB of free disk space — 345 GB of downloaded zip archives plus 367 GB for the extracted dataset. The download script installs the `faster-coco-eval` package and converts the annotations to YOLO format, which can take a long time depending on your connection and hardware.

To train a YOLO26n model on the Objects365 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="Objects365.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=Objects365.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

The Objects365 dataset contains diverse, high-resolution images with dense bounding-box annotations across its 365 classes. The sample below shows the in-the-wild scenes and multi-object annotations typical of the dataset:

![Objects365 dataset sample with diverse object annotations](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/objects365-sample-image.avif)

## Citations and Acknowledgments

If you use the Objects365 dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @InProceedings{Shao_2019_ICCV,
          author = {Shao, Shuai and Li, Zeming and Zhang, Tianyuan and Peng, Chao and Yu, Gang and Zhang, Xiangyu and Li, Jing and Sun, Jian},
          title = {Objects365: A Large-Scale, High-Quality Dataset for Object Detection},
          booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
          month = {October},
          year = {2019}
        }
        ```

We would like to acknowledge the team of researchers who created and maintain the Objects365 dataset as a valuable resource for the computer vision research community. For more information about the Objects365 dataset and its creators, visit the [Objects365 dataset website](https://www.objects365.org/).

## FAQ

### What is the Objects365 dataset used for?

The [Objects365 dataset](https://www.objects365.org/) is used to train and evaluate object detection models in [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and computer vision. It provides 1,742,289 training images and 80,000 validation images across 365 object classes, and it is especially popular for pretraining detectors that are later fine-tuned on smaller, task-specific datasets.

### How many images and classes are in the Objects365 dataset?

The Ultralytics `Objects365.yaml` configuration covers 365 object classes split into 1,742,289 training images and 80,000 validation images — 1,822,289 in total, with no test split. The upstream release reports around 2 million images with 30 million bounding boxes overall.

### How big is the Objects365 dataset download?

Objects365 requires approximately 712 GB of disk space — about 345 GB of zip archives that download automatically the first time you train with `data="Objects365.yaml"`, plus 367 GB for the extracted dataset. The download script installs the `faster-coco-eval` package and converts the annotations to YOLO format. You can browse smaller alternatives in the [detection datasets overview](index.md).

### How can I train a YOLO26 model on the Objects365 dataset?

To train a YOLO26n model using the Objects365 dataset for 100 epochs with an image size of 640, follow these instructions:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="Objects365.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=Objects365.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

Refer to the [Training](../../modes/train.md) page for a comprehensive list of available arguments.

### Why should I use the Objects365 dataset for my object detection projects?

Objects365's 365-class vocabulary and dense annotations make it one of the strongest pretraining datasets for object detection — the ICCV 2019 paper reports a 5.6-point gain (42.0 vs 36.4 mAP) over [ImageNet](../classify/imagenet.md) pretraining on COCO. Its images cover diverse real-world scenarios, which helps models generalize well to downstream detection tasks.

### Where can I find the YAML configuration file for the Objects365 dataset?

The YAML configuration file for the Objects365 dataset is available at [Objects365.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Objects365.yaml). This file contains essential information such as dataset paths and class labels, crucial for setting up your training environment.
