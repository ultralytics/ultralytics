---
comments: true
description: Explore the Carparts Segmentation Dataset for automotive AI applications. Enhance your segmentation models with rich, annotated data using Ultralytics YOLO.
keywords: Carparts Segmentation Dataset, computer vision, automotive AI, vehicle maintenance, Ultralytics, YOLO, segmentation models, deep learning, object segmentation
---

# Carparts Segmentation Dataset

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-carparts-segmentation-dataset.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Carparts Segmentation Dataset In Colab"></a>

The Carparts Segmentation Dataset, available on Roboflow Universe, is a curated collection of images and videos designed for [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) applications, specifically focusing on [segmentation tasks](https://docs.ultralytics.com/tasks/segment/). Hosted on Roboflow Universe, this dataset provides a diverse set of visuals captured from multiple perspectives, offering valuable [annotated](https://www.ultralytics.com/glossary/data-labeling) examples for training and testing segmentation models.

Whether you're working on [automotive research](https://www.ultralytics.com/solutions/ai-in-automotive), developing AI solutions for vehicle maintenance, or exploring computer vision applications, the Carparts Segmentation Dataset serves as a valuable resource for enhancing the [accuracy](https://www.ultralytics.com/glossary/accuracy) and efficiency of your projects using models like [Ultralytics YOLO](../../models/yolo11.md).

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/HATMPgLYAPU"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Carparts <a href="https://www.ultralytics.com/glossary/instance-segmentation">Instance Segmentation</a> with Ultralytics YOLO11.
</p>

## Dataset Structure

The data distribution within the Carparts Segmentation Dataset is organized as follows:

- **Training set**: Includes 3156 images, each accompanied by its corresponding annotations. This set is used for [training](https://www.ultralytics.com/glossary/training-data) the [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) [model](https://www.ultralytics.com/glossary/foundation-model).
- **Testing set**: Comprises 276 images, with each one paired with its respective annotations. This set is used to evaluate the model's performance after training using [test data](https://www.ultralytics.com/glossary/test-data).
- **Validation set**: Consists of 401 images, each having corresponding annotations. This set is used during training to tune [hyperparameters](https://docs.ultralytics.com/guides/hyperparameter-tuning/) and prevent [overfitting](https://www.ultralytics.com/glossary/overfitting) using [validation data](https://www.ultralytics.com/glossary/validation-data).

## Applications

Carparts Segmentation finds applications in various domains including:

- **Automotive Quality Control**: Identifying defects or inconsistencies in car parts during manufacturing ([AI in Manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing)).
- **Auto Repair**: Assisting mechanics in identifying parts for repair or replacement.
- **E-commerce Cataloging**: Automatically tagging and categorizing car parts in online stores for [e-commerce](https://en.wikipedia.org/wiki/E-commerce) platforms.
- **Traffic Monitoring**: Analyzing vehicle components in traffic surveillance footage.
- **Autonomous Vehicles**: Enhancing the perception systems of [self-driving cars](https://www.ultralytics.com/blog/ai-in-self-driving-cars) to better understand surrounding vehicles.
- **Insurance Processing**: Automating damage assessment by identifying affected car parts during insurance claims.
- **Recycling**: Sorting vehicle components for efficient recycling processes.
- **Smart City Initiatives**: Contributing data for urban planning and traffic management systems within [Smart Cities](https://en.wikipedia.org/wiki/Smart_city).

By accurately identifying and categorizing different vehicle components, carparts segmentation streamlines processes and contributes to increased efficiency and automation across these industries.

## Dataset YAML

A [YAML](https://www.ultralytics.com/glossary/yaml) (Yet Another Markup Language) file defines the dataset configuration, including paths, class names, and other essential details. For the Carparts Segmentation dataset, the `carparts-seg.yaml` file is available at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/carparts-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/carparts-seg.yaml). You can learn more about the YAML format at [yaml.org](https://yaml.org/).

!!! example "ultralytics/cfg/datasets/carparts-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/carparts-seg.yaml"
    ```

## Usage

To train an [Ultralytics YOLO11](../../models/yolo11.md) model on the Carparts Segmentation dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the following code snippets. Refer to the model [Training guide](../../modes/train.md) for a comprehensive list of available arguments and explore [model training tips](https://docs.ultralytics.com/guides/model-training-tips/) for best practices.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained segmentation model like YOLO11n-seg
        model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

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
        yolo segment train data=carparts-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640

        # Validate the trained model using the validation set
        # yolo segment val model=path/to/best.pt

        # Predict using the trained model on a specific image source
        # yolo segment predict model=path/to/best.pt source=path/to/your/image.jpg
        ```

## Sample Data and Annotations

The Carparts Segmentation dataset includes a diverse array of images and videos captured from various perspectives. Below are examples showcasing the data and its corresponding annotations:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/dataset-sample-image.avif)

- The image demonstrates [object segmentation](https://docs.ultralytics.com/tasks/segment/) within a car image sample. Annotated [bounding boxes](https://www.ultralytics.com/glossary/bounding-box) with masks highlight the identified car parts (e.g., headlights, grille).
- The dataset features a variety of images captured under different conditions (locations, lighting, object densities), providing a comprehensive resource for training robust car part segmentation models.
- This example underscores the dataset's complexity and the importance of [high-quality data](https://www.ultralytics.com/blog/the-importance-of-high-quality-computer-vision-datasets) for computer vision tasks, especially in specialized domains like automotive component analysis. Techniques like [data augmentation](https://www.ultralytics.com/glossary/data-augmentation) can further enhance model generalization.

## Citations and Acknowledgments

If you utilize the Carparts Segmentation dataset in your research or development efforts, please cite the original source:

!!! quote ""

    === "BibTeX"

        ```bibtex
           @misc{ car-seg-un1pm_dataset,
                title = { car-seg Dataset },
                type = { Open Source Dataset },
                author = { Gianmarco Russo },
                url = { https://universe.roboflow.com/gianmarco-russo-vt9xr/car-seg-un1pm },
                journal = { Roboflow Universe },
                publisher = { Roboflow },
                year = { 2023 },
                month = { nov },
                note = { visited on 2024-01-24 },
            }
        ```

We acknowledge the contribution of Gianmarco Russo and the Roboflow team in creating and maintaining this valuable dataset for the computer vision community. For more datasets, visit the [Ultralytics Datasets collection](https://docs.ultralytics.com/datasets/).

## FAQ

### What is the Carparts Segmentation Dataset?

The Carparts Segmentation Dataset is a specialized collection of images and videos for training computer vision models to perform [segmentation](https://docs.ultralytics.com/tasks/segment/) on car parts. It includes diverse visuals with detailed annotations, suitable for automotive AI applications.

### How can I use the Carparts Segmentation Dataset with Ultralytics YOLO11?

You can train an [Ultralytics YOLO11](../../models/yolo11.md) segmentation model using this dataset. Load a pretrained model (e.g., `yolo11n-seg.pt`) and initiate training using the provided Python or CLI examples, referencing the `carparts-seg.yaml` configuration file. Check the [Training Guide](../../modes/train.md) for detailed instructions.

!!! example "Train Example Snippet"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="carparts-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo segment train data=carparts-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640
        ```

### What are some applications of Carparts Segmentation?

Carparts Segmentation is useful in:

- **Automotive Quality Control**: Ensuring parts meet standards ([AI in Manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing)).
- **Auto Repair**: Identifying parts needing service.
- **E-commerce**: Cataloging parts online.
- **Autonomous Vehicles**: Improving vehicle perception ([AI in Automotive](https://www.ultralytics.com/solutions/ai-in-automotive)).
- **Insurance**: Assessing vehicle damage automatically.
- **Recycling**: Sorting parts efficiently.

### Where can I find the dataset configuration file for Carparts Segmentation?

The dataset configuration file, `carparts-seg.yaml`, which contains details about the dataset paths and classes, is located in the Ultralytics GitHub repository: [carparts-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/carparts-seg.yaml).

### Why should I use the Carparts Segmentation Dataset?

This dataset offers rich, annotated data crucial for developing accurate [segmentation models](https://docs.ultralytics.com/tasks/segment/) for automotive applications. Its diversity helps improve model robustness and performance in real-world scenarios like automated vehicle inspection, enhancing safety systems, and supporting autonomous driving technology. Using high-quality, domain-specific datasets like this accelerates AI development.
