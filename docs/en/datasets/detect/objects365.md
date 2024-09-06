---
comments: true
description: Explore the Objects365 Dataset with 2M images and 30M bounding boxes across 365 categories. Enhance your object detection models with diverse, high-quality data.
keywords: Objects365 dataset, object detection, machine learning, deep learning, computer vision, annotated images, bounding boxes, YOLOv8, high-resolution images, dataset configuration
---

# Objects365 Dataset

The [Objects365](https://www.objects365.org/) dataset is a large-scale, high-quality dataset designed to foster object detection research with a focus on diverse objects in the wild. Created by a team of [Megvii](https://en.megvii.com/) researchers, the dataset offers a wide range of high-resolution images with a comprehensive set of annotated bounding boxes covering 365 object categories.

## Key Features

- Objects365 contains 365 object categories, with 2 million images and over 30 million bounding boxes.
- The dataset includes diverse objects in various scenarios, providing a rich and challenging benchmark for object detection tasks.
- Annotations include bounding boxes for objects, making it suitable for training and evaluating object detection models.
- Objects365 pre-trained models significantly outperform ImageNet pre-trained models, leading to better generalization on various tasks.

## Dataset Structure

The Objects365 dataset is organized into a single set of images with corresponding annotations:

- **Images**: The dataset includes 2 million high-resolution images, each containing a variety of objects across 365 categories.
- **Annotations**: The images are annotated with over 30 million bounding boxes, providing comprehensive ground truth information for object detection tasks.

## Applications

The Objects365 dataset is widely used for training and evaluating deep learning models in object detection tasks. The dataset's diverse set of object categories and high-quality annotations make it a valuable resource for researchers and practitioners in the field of computer vision.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. For the case of the Objects365 Dataset, the `Objects365.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Objects365.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Objects365.yaml).

!!! example "ultralytics/cfg/datasets/Objects365.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/Objects365.yaml"
    ```

## Usage

To train a YOLOv8n model on the Objects365 dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="Objects365.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=Objects365.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

The Objects365 dataset contains a diverse set of high-resolution images with objects from 365 categories, providing rich context for object detection tasks. Here are some examples of the images in the dataset:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/objects365-sample-image.avif)

- **Objects365**: This image demonstrates an example of object detection, where objects are annotated with bounding boxes. The dataset provides a wide range of images to facilitate the development of models for this task.

The example showcases the variety and complexity of the data in the Objects365 dataset and highlights the importance of accurate object detection for computer vision applications.

## Citations and Acknowledgments

If you use the Objects365 dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{shao2019objects365,
          title={Objects365: A Large-scale, High-quality Dataset for Object Detection},
          author={Shao, Shuai and Li, Zeming and Zhang, Tianyuan and Peng, Chao and Yu, Gang and Li, Jing and Zhang, Xiangyu and Sun, Jian},
          booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
          pages={8425--8434},
          year={2019}
        }
        ```

We would like to acknowledge the team of researchers who created and maintain the Objects365 dataset as a valuable resource for the computer vision research community. For more information about the Objects365 dataset and its creators, visit the [Objects365 dataset website](https://www.objects365.org/).

## FAQ

### What is the Objects365 dataset used for?

The [Objects365 dataset](https://www.objects365.org/) is designed for object detection tasks in machine learning and computer vision. It provides a large-scale, high-quality dataset with 2 million annotated images and 30 million bounding boxes across 365 categories. Leveraging such a diverse dataset helps improve the performance and generalization of object detection models, making it invaluable for research and development in the field.

### How can I train a YOLOv8 model on the Objects365 dataset?

To train a YOLOv8n model using the Objects365 dataset for 100 epochs with an image size of 640, follow these instructions:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="Objects365.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=Objects365.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

Refer to the [Training](../../modes/train.md) page for a comprehensive list of available arguments.

### Why should I use the Objects365 dataset for my object detection projects?

The Objects365 dataset offers several advantages for object detection tasks:

1. **Diversity**: It includes 2 million images with objects in diverse scenarios, covering 365 categories.
2. **High-quality Annotations**: Over 30 million bounding boxes provide comprehensive ground truth data.
3. **Performance**: Models pre-trained on Objects365 significantly outperform those trained on datasets like ImageNet, leading to better generalization.

### Where can I find the YAML configuration file for the Objects365 dataset?

The YAML configuration file for the Objects365 dataset is available at [Objects365.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Objects365.yaml). This file contains essential information such as dataset paths and class labels, crucial for setting up your training environment.

### How does the dataset structure of Objects365 enhance object detection modeling?

The [Objects365 dataset](https://www.objects365.org/) is organized with 2 million high-resolution images and comprehensive annotations of over 30 million bounding boxes. This structure ensures a robust dataset for training deep learning models in object detection, offering a wide variety of objects and scenarios. Such diversity and volume help in developing models that are more accurate and capable of generalizing well to real-world applications. For more details on the dataset structure, refer to the [Dataset YAML](#dataset-yaml) section.
