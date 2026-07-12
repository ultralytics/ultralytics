---
title: Carparts-Seg Dataset
comments: true
description: Train Ultralytics YOLO segmentation models on Carparts-Seg — 3,833 annotated images across 23 car-part classes for automotive AI applications.
keywords: Carparts Segmentation Dataset, computer vision, automotive AI, vehicle maintenance, Ultralytics, YOLO, segmentation models, deep learning, object segmentation
---

# Carparts Segmentation Dataset

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-carparts-segmentation-dataset.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Carparts Segmentation Dataset In Colab"></a>

The [Ultralytics](https://www.ultralytics.com/) Carparts Segmentation Dataset provides 3,833 annotated images across 23 car-part classes — including bumpers, doors, lights, mirrors, hood, and trunk — for training [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) models on automotive [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks. Captured from multiple perspectives and [annotated](https://www.ultralytics.com/glossary/data-labeling) with pixel-level masks, it pairs directly with [Ultralytics YOLO](../../models/yolo26.md) for use cases ranging from automotive quality control and auto repair to insurance-claim damage assessment and autonomous-vehicle perception.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/FvWl00sD4rc"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Segment Carparts with Ultralytics Platform | Train, Deploy & Inference | Ultralytics YOLO26 🚀
</p>

## Dataset Structure

The Carparts Segmentation Dataset splits its 3,833 images as follows:

- **Training set**: 3,156 images used for [training](https://www.ultralytics.com/glossary/training-data) the [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) [model](https://www.ultralytics.com/glossary/foundation-model).
- **Validation set**: 401 images used during training to tune [hyperparameters](../../guides/hyperparameter-tuning.md) and prevent [overfitting](https://www.ultralytics.com/glossary/overfitting) on [validation data](https://www.ultralytics.com/glossary/validation-data).
- **Testing set**: 276 images used to evaluate the model on held-out [test data](https://www.ultralytics.com/glossary/test-data) after training.
- **Classes**: 23 in total — 22 named car-part categories (bumpers, doors, lights, glass, mirrors, hood, tailgate, trunk, and wheels) plus a catch-all `object` class for parts outside those categories.
- **Download size**: ~133 MB.

## Applications

Carparts Segmentation finds applications in various domains including:

- **Automotive Quality Control**: Identifying defects or inconsistencies in car parts during manufacturing ([AI in Manufacturing](https://www.ultralytics.com/solutions/computer-vision-in-manufacturing)).
- **Auto Repair**: Assisting mechanics in identifying parts for repair or replacement.
- **E-commerce Cataloging**: Automatically tagging and categorizing car parts in online stores for [e-commerce](https://en.wikipedia.org/wiki/E-commerce) platforms.
- **Traffic Monitoring**: Analyzing vehicle components in traffic surveillance footage.
- **Autonomous Vehicles**: Enhancing the perception systems of [self-driving cars](https://www.ultralytics.com/blog/ai-in-self-driving-cars) to better understand surrounding vehicles.
- **Insurance Processing**: Automating damage assessment by identifying affected car parts during insurance claims.
- **Recycling**: Sorting vehicle components for efficient recycling processes.
- **Smart City Initiatives**: Contributing data for urban planning and traffic management systems within [Smart Cities](https://en.wikipedia.org/wiki/Smart_city).

The complete Carparts Segmentation Dataset can also be browsed and managed on [Ultralytics Platform](https://platform.ultralytics.com/).

## Dataset YAML

A [YAML](https://www.ultralytics.com/glossary/yaml) (Yet Another Markup Language) file defines the dataset configuration, including paths, class names, and other essential details. For the Carparts Segmentation dataset, the `carparts-seg.yaml` file is available at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/carparts-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/carparts-seg.yaml). You can learn more about the YAML format at [yaml.org](https://yaml.org/).

!!! example "ultralytics/cfg/datasets/carparts-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/carparts-seg.yaml"
    ```

## Usage

To train an [Ultralytics YOLO26](../../models/yolo26.md) model on the Carparts Segmentation dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the following code snippets. Refer to the model [Training guide](../../modes/train.md) for a comprehensive list of available arguments and explore [model training tips](../../guides/model-training-tips.md) for best practices.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained segmentation model like YOLO26n-seg
        model = YOLO("yolo26n-seg.pt")  # load a pretrained model (recommended for training)

        # Train the model on the Carparts Segmentation dataset
        results = model.train(data="carparts-seg.yaml", epochs=100, imgsz=640)

        # After training, you can validate the model's performance on the validation set
        results = model.val()

        # Or perform prediction on new images or videos
        results = model.predict("path/to/your/image.jpg")
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model using the Command Line Interface
        # Specify the dataset config file, model, number of epochs, and image size
        yolo segment train data=carparts-seg.yaml model=yolo26n-seg.pt epochs=100 imgsz=640

        # Validate the trained model using the validation set
        yolo segment val data=carparts-seg.yaml model=path/to/best.pt

        # Predict using the trained model on a specific image source
        yolo segment predict model=path/to/best.pt source=path/to/your/image.jpg
        ```

## Sample Data and Annotations

Below is an example image from the Carparts Segmentation Dataset with its [object segmentation](../../tasks/segment.md) masks overlaid, showing how individual car parts are outlined and labeled:

![Car parts segmentation dataset sample image](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/carparts-seg-sample.avif)

The dataset spans varied locations, lighting conditions, and object densities, giving models trained on it exposure to the range of real-world scenes they'll need to generalize across.

## Citations and Acknowledgments

If you utilize the Carparts Segmentation dataset in your research or development efforts, please cite the original source:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{car-seg-un1pm_dataset,
              title = { car-seg Dataset },
              type = { Open Source Dataset },
              author = { Gianmarco Russo },
              publisher = { Roboflow },
              year = { 2023 },
              month = { nov },
        }
        ```

We acknowledge the contribution of Gianmarco Russo and the Roboflow team in creating and maintaining this valuable dataset for the computer vision community. For more datasets, visit the [Ultralytics Datasets collection](../index.md).

## FAQ

### What is the Carparts Segmentation Dataset, and how is it used in Ultralytics YOLO26?

The **Carparts Segmentation Dataset** is a curated collection of 3,833 annotated images spanning 23 car-part classes — bumpers, doors, lights, mirrors, hood, trunk, and more — for training and evaluating [instance segmentation](../../tasks/segment.md) models. It's built for automotive computer vision applications like quality control, auto repair, and damage assessment, and is used directly with Ultralytics [YOLO26](../../models/yolo26.md) via the `carparts-seg.yaml` configuration file.

### How many images and classes does the Carparts Segmentation Dataset contain?

The dataset totals 3,833 images — 3,156 for training, 401 for validation, and 276 for testing — across 23 classes: 22 named car-part categories plus a catch-all `object` class for parts outside them. The full archive downloads automatically as a ~133 MB `.zip` on first use.

### How can I train an Ultralytics YOLO26 model on the Carparts Segmentation Dataset?

Load a pretrained segmentation model (e.g., `yolo26n-seg.pt`) and train it with the `carparts-seg.yaml` configuration using the Python or CLI snippets in the [Usage](#usage) section above. See the [Training guide](../../modes/train.md) for the full list of available arguments.

### What are some applications of the Carparts Segmentation Dataset?

Carparts segmentation supports automotive quality control, auto repair, e-commerce cataloging, traffic monitoring, autonomous-vehicle perception, insurance damage assessment, recycling, and smart-city initiatives — see the [Applications](#applications) section above for details on each use case.

### Where can I find the dataset configuration file for Carparts Segmentation?

The dataset configuration file, `carparts-seg.yaml`, which contains details about the dataset paths and classes, is located in the Ultralytics GitHub repository: [carparts-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/carparts-seg.yaml).

### Why should I use the Carparts Segmentation Dataset?

This dataset offers rich, annotated data crucial for developing accurate [segmentation models](../../tasks/segment.md) for automotive applications. Its diversity helps improve model robustness and performance in real-world scenarios like automated vehicle inspection, enhancing safety systems, and supporting autonomous driving technology. Using high-quality, domain-specific datasets like this accelerates AI development.
