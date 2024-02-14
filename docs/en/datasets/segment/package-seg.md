---
comments: true
description: Explore the Package Segmentation using Ultralytics YOLOv8 Dataset, a large-scale benchmark for logistics, and learn how to train a YOLO model using it.
keywords: Packet Segmentation Dataset, Ultralytics, Manufacturing, Logistics, YOLO model, object detection, object tracking
---

# Roboflow Universe Package Segmentation Dataset

The [Roboflow](https://roboflow.com/?ref=ultralytics) [Package Segmentation Dataset](https://universe.roboflow.com/factorypackage/factory_package) is a curated collection of images specifically tailored for tasks related to package segmentation in the field of computer vision. This dataset is designed to assist researchers, developers, and enthusiasts working on projects related to package identification, sorting, and handling.

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

!!! Example "ultralytics/cfg/datasets/package-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/package-seg.yaml"
    ```

## Usage

To train Ultralytics YOLOv8n model on the Package Segmentation dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! Example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data='package-seg.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo segment train data=package-seg.yaml model=yolov8n-seg.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

The Package Segmentation dataset comprises a varied collection of images and videos captured from multiple perspectives. Below are instances of data from the dataset, accompanied by their respective annotations:

![Dataset sample image](https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/55bdf5c8-4ae4-4824-8d08-63c15bdd9a92)

- This image displays an instance of image object detection, featuring annotated bounding boxes with masks outlining recognized objects. The dataset incorporates a diverse collection of images taken in different locations, environments, and densities. It serves as a comprehensive resource for developing models specific to this task.
- The example emphasizes the diversity and complexity present in the VisDrone dataset, underscoring the significance of high-quality sensor data for computer vision tasks involving drones.

## Citations and Acknowledgments

If you integrate the crack segmentation dataset into your research or development initiatives, please cite the following paper:

!!! Quote ""

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

We express our gratitude to the Roboflow team for their efforts in creating and maintaining the Package Segmentation dataset, a valuable asset for logistics and research projects. For additional details about the Package Segmentation dataset and its creators, please visit the [Package Segmentation Dataset Page](https://universe.roboflow.com/factorypackage/factory_package).
