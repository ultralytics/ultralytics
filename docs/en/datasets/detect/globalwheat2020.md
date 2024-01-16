---
comments: true
description: Understand how to utilize the vast Global Wheat Head Dataset for building wheat head detection models. Features, structure, applications, usage, sample data, and citation.
keywords: Ultralytics, YOLO, Global Wheat Head Dataset, wheat head detection, plant phenotyping, crop management, deep learning, outdoor images, annotations, YAML configuration
---

# Global Wheat Head Dataset

The [Global Wheat Head Dataset](https://www.global-wheat.com/) is a collection of images designed to support the development of accurate wheat head detection models for applications in wheat phenotyping and crop management. Wheat heads, also known as spikes, are the grain-bearing parts of the wheat plant. Accurate estimation of wheat head density and size is essential for assessing crop health, maturity, and yield potential. The dataset, created by a collaboration of nine research institutes from seven countries, covers multiple growing regions to ensure models generalize well across different environments.

## Key Features

- The dataset contains over 3,000 training images from Europe (France, UK, Switzerland) and North America (Canada).
- It includes approximately 1,000 test images from Australia, Japan, and China.
- Images are outdoor field images, capturing the natural variability in wheat head appearances.
- Annotations include wheat head bounding boxes to support object detection tasks.

## Dataset Structure

The Global Wheat Head Dataset is organized into two main subsets:

1. **Training Set**: This subset contains over 3,000 images from Europe and North America. The images are labeled with wheat head bounding boxes, providing ground truth for training object detection models.
2. **Test Set**: This subset consists of approximately 1,000 images from Australia, Japan, and China. These images are used for evaluating the performance of trained models on unseen genotypes, environments, and observational conditions.

## Applications

The Global Wheat Head Dataset is widely used for training and evaluating deep learning models in wheat head detection tasks. The dataset's diverse set of images, capturing a wide range of appearances, environments, and conditions, make it a valuable resource for researchers and practitioners in the field of plant phenotyping and crop management.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. For the case of the Global Wheat Head Dataset, the `GlobalWheat2020.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/GlobalWheat2020.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/GlobalWheat2020.yaml).

!!! Example "ultralytics/cfg/datasets/GlobalWheat2020.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/GlobalWheat2020.yaml"
    ```

## Usage

To train a YOLOv8n model on the Global Wheat Head Dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! Example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data='GlobalWheat2020.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=GlobalWheat2020.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

The Global Wheat Head Dataset contains a diverse set of outdoor field images, capturing the natural variability in wheat head appearances, environments, and conditions. Here are some examples of data from the dataset, along with their corresponding annotations:

![Dataset sample image](https://i.ytimg.com/vi/yqvMuw-uedU/maxresdefault.jpg)

- **Wheat Head Detection**: This image demonstrates an example of wheat head detection, where wheat heads are annotated with bounding boxes. The dataset provides a variety of images to facilitate the development of models for this task.

The example showcases the variety and complexity of the data in the Global Wheat Head Dataset and highlights the importance of accurate wheat head detection for applications in wheat phenotyping and crop management.

## Citations and Acknowledgments

If you use the Global Wheat Head Dataset in your research or development work, please cite the following paper:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{david2020global,
                 title={Global Wheat Head Detection (GWHD) Dataset: A Large and Diverse Dataset of High-Resolution RGB-Labelled Images to Develop and Benchmark Wheat Head Detection Methods},
                 author={David, Etienne and Madec, Simon and Sadeghi-Tehran, Pouria and Aasen, Helge and Zheng, Bangyou and Liu, Shouyang and Kirchgessner, Norbert and Ishikawa, Goro and Nagasawa, Koichi and Badhon, Minhajul and others},
                 journal={arXiv preprint arXiv:2005.02162},
                 year={2020}
        }
        ```

We would like to acknowledge the researchers and institutions that contributed to the creation and maintenance of the Global Wheat Head Dataset as a valuable resource for the plant phenotyping and crop management research community. For more information about the dataset and its creators, visit the [Global Wheat Head Dataset website](https://www.global-wheat.com/).
