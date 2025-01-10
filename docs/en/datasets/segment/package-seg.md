---
comments: true
description: Explore the Roboflow Package Segmentation Dataset. Optimize logistics and enhance vision models with curated images for package identification and sorting.
keywords: Roboflow, Package Segmentation Dataset, computer vision, package identification, logistics, warehouse automation, segmentation models, training data
---

# Roboflow Universe Package Segmentation Dataset

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-package-segmentation-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Package Segmentation Dataset In Colab"></a>

The [Roboflow](https://roboflow.com/?ref=ultralytics) [Package Segmentation Dataset](https://universe.roboflow.com/factorypackage/factory_package?ref=ultralytics) is a curated collection of images specifically tailored for tasks related to package segmentation in the field of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv). This dataset is designed to assist researchers, developers, and enthusiasts working on projects related to package identification, sorting, and handling.

Containing a diverse set of images showcasing various packages in different contexts and environments, the dataset serves as a valuable resource for training and evaluating segmentation models. Whether you are engaged in logistics, warehouse automation, or any application requiring precise package analysis, the Package Segmentation Dataset provides a targeted and comprehensive set of images to enhance the performance of your computer vision algorithms.

## Dataset Structure

The distribution of data in the Package Segmentation Dataset is structured as follows:

- **Training set**: Encompasses 1920 images accompanied by their corresponding annotations.
- **Testing set**: Consists of 89 images, each paired with its respective annotations.
- **Validation set**: Comprises 188 images, each with corresponding annotations.

## Applications

Package segmentation, facilitated by the Package Segmentation Dataset, is crucial for optimizing logistics, enhancing last-mile delivery, improving manufacturing quality control, and contributing to smart city solutions. From e-commerce to security applications, this dataset is a key resource, fostering innovation in computer vision for diverse and efficient package analysis applications.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the Package Segmentation dataset, the `package-seg.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/package-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/package-seg.yaml).

!!! example "ultralytics/cfg/datasets/package-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/package-seg.yaml"
    ```

## Usage

To train Ultralytics YOLO11n model on the Package Segmentation dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="package-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo segment train data=package-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

The Package Segmentation dataset comprises a varied collection of images and videos captured from multiple perspectives. Below are instances of data from the dataset, accompanied by their respective annotations:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/dataset-sample-image-1.avif)

- This image displays an instance of image [object detection](https://www.ultralytics.com/glossary/object-detection), featuring annotated bounding boxes with masks outlining recognized objects. The dataset incorporates a diverse collection of images taken in different locations, environments, and densities. It serves as a comprehensive resource for developing models specific to this task.
- The example emphasizes the diversity and complexity present in the VisDrone dataset, underscoring the significance of high-quality sensor data for computer vision tasks involving drones.

## Citations and Acknowledgments

If you integrate the crack segmentation dataset into your research or development initiatives, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{ factory_package_dataset,
            title = { factory_package Dataset },
            type = { Open Source Dataset },
            author = { factorypackage },
            howpublished = { \url{ https://universe.roboflow.com/factorypackage/factory_package } },
            url = { https://universe.roboflow.com/factorypackage/factory_package },
            journal = { Roboflow Universe },
            publisher = { Roboflow },
            year = { 2024 },
            month = { jan },
            note = { visited on 2024-01-24 },
        }
        ```

We express our gratitude to the Roboflow team for their efforts in creating and maintaining the Package Segmentation dataset, a valuable asset for logistics and research projects. For additional details about the Package Segmentation dataset and its creators, please visit the [Package Segmentation Dataset Page](https://universe.roboflow.com/factorypackage/factory_package?ref=ultralytics).

## FAQ

### What is the Roboflow Package Segmentation Dataset and how can it help in computer vision projects?

The [Roboflow Package Segmentation Dataset](https://universe.roboflow.com/factorypackage/factory_package?ref=ultralytics) is a curated collection of images tailored for tasks involving package segmentation. It includes diverse images of packages in various contexts, making it invaluable for training and evaluating segmentation models. This dataset is particularly useful for applications in logistics, warehouse automation, and any project requiring precise package analysis. It helps optimize logistics and enhance vision models for accurate package identification and sorting.

### How do I train an Ultralytics YOLO11 model on the Package Segmentation Dataset?

You can train an Ultralytics YOLO11n model using both Python and CLI methods. Use the snippets below:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-seg.pt")  # load a pretrained model

        # Train the model
        results = model.train(data="package-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo segment train data=package-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640
        ```

Refer to the model [Training](../../modes/train.md) page for more details.

### What are the components of the Package Segmentation Dataset, and how is it structured?

The dataset is structured into three main components:

- **Training set**: Contains 1920 images with annotations.
- **Testing set**: Comprises 89 images with corresponding annotations.
- **Validation set**: Includes 188 images with annotations.

This structure ensures a balanced dataset for thorough model training, validation, and testing, enhancing the performance of segmentation algorithms.

### Why should I use Ultralytics YOLO11 with the Package Segmentation Dataset?

Ultralytics YOLO11 provides state-of-the-art [accuracy](https://www.ultralytics.com/glossary/accuracy) and speed for real-time object detection and segmentation tasks. Using it with the Package Segmentation Dataset allows you to leverage YOLO11's capabilities for precise package segmentation. This combination is especially beneficial for industries like logistics and warehouse automation, where accurate package identification is critical. For more information, check out our [page on YOLO11 segmentation](https://docs.ultralytics.com/models/yolo11/).

### How can I access and use the package-seg.yaml file for the Package Segmentation Dataset?

The `package-seg.yaml` file is hosted on Ultralytics' GitHub repository and contains essential information about the dataset's paths, classes, and configuration. You can download it from [here](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/package-seg.yaml). This file is crucial for configuring your models to utilize the dataset efficiently.

For more insights and practical examples, explore our [Usage](https://docs.ultralytics.com/usage/python/) section.
