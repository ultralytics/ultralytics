---
comments: true
description: Explore the Global Wheat Head Dataset to develop accurate wheat head detection models. Includes training images, annotations, and usage for crop management.
keywords: Global Wheat Head Dataset, wheat head detection, wheat phenotyping, crop management, deep learning, object detection, training datasets
---

# Global Wheat Head Dataset

The [Global Wheat Head Dataset](https://www.global-wheat.com/) is a collection of images designed to support the development of accurate wheat head detection models for applications in wheat phenotyping and crop management. Wheat heads, also known as spikes, are the grain-bearing parts of the wheat plant. Accurate estimation of wheat head density and size is essential for assessing crop health, maturity, and yield potential. The dataset, created by a collaboration of nine research institutes from seven countries, covers multiple growing regions to ensure models generalize well across different environments.

## Key Features

- The dataset contains over 3,000 training images from Europe (France, UK, Switzerland) and North America (Canada).
- It includes approximately 1,000 test images from Australia, Japan, and China.
- Images are outdoor field images, capturing the natural variability in wheat head appearances.
- Annotations include wheat head bounding boxes to support [object detection](https://docs.ultralytics.com/tasks/detect/) tasks.

## Dataset Structure

The Global Wheat Head Dataset is organized into two main subsets:

1. **Training Set**: This subset contains over 3,000 images from Europe and North America. The images are labeled with wheat head bounding boxes, providing ground truth for training object detection models.
2. **Test Set**: This subset consists of approximately 1,000 images from Australia, Japan, and China. These images are used for evaluating the performance of trained models on unseen genotypes, environments, and observational conditions.

## Applications

The Global Wheat Head Dataset is widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in wheat head detection tasks. The dataset's diverse set of images, capturing a wide range of appearances, environments, and conditions, make it a valuable resource for researchers and practitioners in the field of [plant phenotyping](https://www.ultralytics.com/blog/computer-vision-in-agriculture-transforming-fruit-detection-and-precision-farming) and crop management.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. For the case of the Global Wheat Head Dataset, the `GlobalWheat2020.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/GlobalWheat2020.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/GlobalWheat2020.yaml).

!!! example "ultralytics/cfg/datasets/GlobalWheat2020.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/GlobalWheat2020.yaml"
    ```

## Usage

To train a YOLO11n model on the Global Wheat Head Dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="GlobalWheat2020.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=GlobalWheat2020.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

The Global Wheat Head Dataset contains a diverse set of outdoor field images, capturing the natural variability in wheat head appearances, environments, and conditions. Here are some examples of data from the dataset, along with their corresponding annotations:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/wheat-head-detection-sample.avif)

- **Wheat Head Detection**: This image demonstrates an example of wheat head detection, where wheat heads are annotated with bounding boxes. The dataset provides a variety of images to facilitate the development of models for this task.

The example showcases the variety and complexity of the data in the Global Wheat Head Dataset and highlights the importance of accurate wheat head detection for applications in wheat phenotyping and crop management.

## Citations and Acknowledgments

If you use the Global Wheat Head Dataset in your research or development work, please cite the following paper:

!!! quote ""

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

## FAQ

### What is the Global Wheat Head Dataset used for?

The Global Wheat Head Dataset is primarily used for developing and training deep learning models aimed at wheat head detection. This is crucial for applications in [wheat phenotyping](https://www.ultralytics.com/blog/from-farm-to-table-how-ai-drives-innovation-in-agriculture) and crop management, allowing for more accurate estimations of wheat head density, size, and overall crop yield potential. Accurate detection methods help in assessing crop health and maturity, essential for efficient crop management.

### How do I train a YOLO11n model on the Global Wheat Head Dataset?

To train a YOLO11n model on the Global Wheat Head Dataset, you can use the following code snippets. Make sure you have the `GlobalWheat2020.yaml` configuration file specifying dataset paths and classes:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pre-trained model (recommended for training)
        model = YOLO("yolo11n.pt")

        # Train the model
        results = model.train(data="GlobalWheat2020.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=GlobalWheat2020.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

### What are the key features of the Global Wheat Head Dataset?

Key features of the Global Wheat Head Dataset include:

- Over 3,000 training images from Europe (France, UK, Switzerland) and North America (Canada).
- Approximately 1,000 test images from Australia, Japan, and China.
- High variability in wheat head appearances due to different growing environments.
- Detailed annotations with wheat head bounding boxes to aid [object detection](https://www.ultralytics.com/glossary/object-detection) models.

These features facilitate the development of robust models capable of generalization across multiple regions.

### Where can I find the configuration YAML file for the Global Wheat Head Dataset?

The configuration YAML file for the Global Wheat Head Dataset, named `GlobalWheat2020.yaml`, is available on GitHub. You can access it at this [link](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/GlobalWheat2020.yaml). This file contains necessary information about dataset paths, classes, and other configuration details needed for model training in [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo11/).

### Why is wheat head detection important in crop management?

Wheat head detection is critical in crop management because it enables accurate estimation of wheat head density and size, which are essential for evaluating crop health, maturity, and yield potential. By leveraging [deep learning models](https://docs.ultralytics.com/models/) trained on datasets like the Global Wheat Head Dataset, farmers and researchers can better monitor and manage crops, leading to improved productivity and optimized resource use in agricultural practices. This technological advancement supports [sustainable agriculture](https://www.ultralytics.com/blog/real-time-crop-health-monitoring-with-ultralytics-yolo11) and food security initiatives.

For more information on applications of AI in agriculture, visit [AI in Agriculture](https://www.ultralytics.com/solutions/ai-in-agriculture).
