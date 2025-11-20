---
comments: true
description: Explore the Package Segmentation Dataset. Optimize logistics and enhance vision models with curated images for package identification and sorting.
keywords: Package Segmentation Dataset, computer vision, package identification, logistics, warehouse automation, segmentation models, training data, Ultralytics YOLO
---

# Package Segmentation Dataset

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-package-segmentation-dataset.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Package Segmentation Dataset In Colab"></a>

The Package Segmentation Dataset, available on Roboflow Universe, is a curated collection of images specifically tailored for tasks related to package segmentation within the field of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv). This dataset is designed to assist researchers, developers, and enthusiasts working on projects involving package identification, sorting, and handling, primarily focusing on [image segmentation](https://www.ultralytics.com/glossary/image-segmentation) tasks.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/im7xBCnPURg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Train Package Segmentation Model using Ultralytics YOLO11 | Industrial Packages 🎉
</p>

Containing a diverse set of images showcasing various packages in different contexts and environments, the dataset serves as a valuable resource for training and evaluating segmentation models. Whether you are engaged in logistics, warehouse automation, or any application requiring precise package analysis, the Package Segmentation Dataset provides a targeted and comprehensive set of images to enhance the performance of your computer vision algorithms. Explore more datasets for segmentation tasks on our [datasets overview page](https://docs.ultralytics.com/datasets/segment/).

## Dataset Structure

The distribution of data in the Package Segmentation Dataset is structured as follows:

- **Training set**: Encompasses 1920 images accompanied by their corresponding annotations.
- **Testing set**: Consists of 89 images, each paired with its respective annotations.
- **Validation set**: Comprises 188 images, each with corresponding annotations.

## Applications

Package segmentation, facilitated by the Package Segmentation Dataset, is crucial for optimizing logistics, enhancing last-mile delivery, improving manufacturing quality control, and contributing to smart city solutions. From e-commerce to security applications, this dataset is a key resource, fostering innovation in computer vision for diverse and efficient package analysis applications.

### Smart Warehouses and Logistics

In modern warehouses, [vision AI solutions](https://www.ultralytics.com/solutions) can streamline operations by automating package identification and sorting. Computer vision models trained on this dataset can quickly detect and segment packages in real-time, even in challenging environments with dim lighting or cluttered spaces. This leads to faster processing times, reduced errors, and improved overall efficiency in [logistics operations](https://www.ultralytics.com/blog/ultralytics-yolo11-the-key-to-computer-vision-in-logistics).

### Quality Control and Damage Detection

Package segmentation models can be used to identify damaged packages by analyzing their shape and appearance. By detecting irregularities or deformations in package outlines, these models help ensure that only intact packages proceed through the supply chain, reducing customer complaints and return rates. This is a key aspect of [quality control in manufacturing](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision) and is vital for maintaining product integrity.

## Dataset YAML

A YAML (Yet Another Markup Language) file defines the dataset configuration, including paths, classes, and other essential details. For the Package Segmentation dataset, the `package-seg.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/package-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/package-seg.yaml).

!!! example "ultralytics/cfg/datasets/package-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/package-seg.yaml"
    ```

## Usage

To train an [Ultralytics YOLO11n](https://docs.ultralytics.com/models/yolo11/) model on the Package Segmentation dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training page](../../modes/train.md).

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-seg.pt")  # load a pretrained segmentation model (recommended for training)

        # Train the model on the Package Segmentation dataset
        results = model.train(data="package-seg.yaml", epochs=100, imgsz=640)

        # Validate the model
        results = model.val()

        # Perform inference on an image
        results = model("path/to/image.jpg")
        ```

    === "CLI"

        ```bash
        # Load a pretrained segmentation model and start training
        yolo segment train data=package-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640

        # Resume training from the last checkpoint
        yolo segment train data=package-seg.yaml model=path/to/last.pt resume=True

        # Validate the trained model
        yolo segment val data=package-seg.yaml model=path/to/best.pt

        # Perform inference using the trained model
        yolo segment predict model=path/to/best.pt source=path/to/image.jpg
        ```

## Sample Data and Annotations

The Package Segmentation dataset comprises a varied collection of images captured from multiple perspectives. Below are instances of data from the dataset, accompanied by their respective segmentation masks:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/dataset-sample-image-1.avif)

- This image displays an instance of package segmentation, featuring annotated masks outlining recognized package objects. The dataset incorporates a diverse collection of images taken in different locations, environments, and densities. It serves as a comprehensive resource for developing models specific to this [segmentation task](https://docs.ultralytics.com/tasks/segment/).
- The example emphasizes the diversity and complexity present in the dataset, underscoring the significance of high-quality data for computer vision tasks involving package segmentation.

## Benefits of Using YOLO11 for Package Segmentation

[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) offers several advantages for package segmentation tasks:

1.  **Speed and Accuracy Balance**: YOLO11 achieves high precision and efficiency, making it ideal for [real-time inference](https://www.ultralytics.com/glossary/real-time-inference) in fast-paced logistics environments. It provides a strong balance compared to models like [YOLOv8](https://docs.ultralytics.com/models/yolov8/).

2.  **Adaptability**: Models trained with YOLO11 can adapt to various warehouse conditions, from dim lighting to cluttered spaces, ensuring robust performance.

3.  **Scalability**: During peak periods like holiday seasons, YOLO11 models can efficiently scale to handle increased package volumes without compromising performance or [accuracy](https://www.ultralytics.com/glossary/accuracy).

4.  **Integration Capabilities**: YOLO11 can be easily integrated with existing warehouse management systems and deployed across various platforms using formats like [ONNX](https://docs.ultralytics.com/integrations/onnx/) or [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/), facilitating end-to-end automated solutions.

## Citations and Acknowledgments

If you integrate the Package Segmentation dataset into your research or development initiatives, please cite the source appropriately:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{ factory_package_dataset,
            title = { factory_package Dataset },
            type = { Open Source Dataset },
            author = { factorypackage },
            url = { https://universe.roboflow.com/factorypackage/factory_package },
            journal = { Roboflow Universe },
            publisher = { Roboflow },
            year = { 2024 },
            month = { jan },
            note = { visited on 2024-01-24 },
        }
        ```

We express our gratitude to the creators of the Package Segmentation dataset for their contribution to the computer vision community. For further exploration of datasets and model training, consider visiting our [Ultralytics Datasets](https://docs.ultralytics.com/datasets/) page and our guide on [model training tips](https://docs.ultralytics.com/guides/model-training-tips/).

## FAQ

### What is the Package Segmentation Dataset and how can it help in computer vision projects?

- The Package Segmentation Dataset is a curated collection of images tailored for tasks involving package [image segmentation](https://www.ultralytics.com/glossary/image-segmentation). It includes diverse images of packages in various contexts, making it invaluable for training and evaluating segmentation models. This dataset is particularly useful for applications in logistics, warehouse automation, and any project requiring precise package analysis.

### How do I train an Ultralytics YOLO11 model on the Package Segmentation Dataset?

- You can train an [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) model using both Python and CLI methods. Use the code snippets provided in the [Usage](#usage) section. Refer to the model [Training page](../../modes/train.md) for more details on arguments and configurations.

### What are the components of the Package Segmentation Dataset, and how is it structured?

- The dataset is structured into three main components:
    - **Training set**: Contains 1920 images with annotations.
    - **Testing set**: Comprises 89 images with corresponding annotations.
    - **Validation set**: Includes 188 images with annotations.
- This structure ensures a balanced dataset for thorough model training, validation, and testing, following best practices outlined in [model evaluation guides](https://docs.ultralytics.com/guides/model-evaluation-insights/).

### Why should I use Ultralytics YOLO11 with the Package Segmentation Dataset?

- Ultralytics YOLO11 provides state-of-the-art [accuracy](https://www.ultralytics.com/glossary/accuracy) and speed for real-time [object detection](https://www.ultralytics.com/glossary/object-detection) and segmentation tasks. Using it with the Package Segmentation Dataset allows you to leverage YOLO11's capabilities for precise package segmentation, which is especially beneficial for industries like [logistics](https://www.ultralytics.com/blog/ultralytics-yolo11-the-key-to-computer-vision-in-logistics) and warehouse automation.

### How can I access and use the package-seg.yaml file for the Package Segmentation Dataset?

- The `package-seg.yaml` file is hosted on Ultralytics' GitHub repository and contains essential information about the dataset's paths, classes, and configuration. You can view or download it at <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/package-seg.yaml>. This file is crucial for configuring your models to utilize the dataset efficiently. For more insights and practical examples, explore our [Python Usage](https://docs.ultralytics.com/usage/python/) section.
