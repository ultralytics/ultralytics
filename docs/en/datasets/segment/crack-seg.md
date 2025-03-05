---
comments: true
description: Explore the extensive Roboflow Crack Segmentation Dataset, perfect for transportation and public safety studies or self-driving car model development.
keywords: Roboflow, Crack Segmentation Dataset, Ultralytics, transportation safety, public safety, self-driving cars, computer vision, road safety, infrastructure maintenance, dataset
---

# Roboflow Universe Crack Segmentation Dataset

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-crack-segmentation-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Crack Segmentation Dataset In Colab"></a>

The [Roboflow](https://roboflow.com/?ref=ultralytics) [Crack Segmentation Dataset](https://universe.roboflow.com/university-bswxt/crack-bphdr?ref=ultralytics) stands out as an extensive resource designed specifically for individuals involved in transportation and public safety studies. It is equally beneficial for those working on the development of self-driving car models or simply exploring [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) applications for recreational purposes.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/C4mc40YKm-g"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Crack segmentation using Ultralytics YOLOv9
</p>

Comprising a total of 4029 static images captured from diverse road and wall scenarios, this dataset emerges as a valuable asset for tasks related to crack segmentation. Whether you are delving into the intricacies of transportation research or seeking to enhance the [accuracy](https://www.ultralytics.com/glossary/accuracy) of your self-driving car models, this dataset provides a rich and varied collection of images to support your endeavors.

## Dataset Structure

The division of data within the Crack Segmentation Dataset is outlined as follows:

- **Training set**: Consists of 3717 images with corresponding annotations.
- **Testing set**: Comprises 112 images along with their respective annotations.
- **Validation set**: Includes 200 images with their corresponding annotations.

## Applications

Crack segmentation finds practical applications in infrastructure maintenance, aiding in the identification and assessment of structural damage. It also plays a crucial role in enhancing road safety by enabling automated systems to detect and address pavement cracks for timely repairs.

## Dataset YAML

A YAML (Yet Another Markup Language) file is employed to outline the configuration of the dataset, encompassing details about paths, classes, and other pertinent information. Specifically, for the Crack Segmentation dataset, the `crack-seg.yaml` file is managed and accessible at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/crack-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/crack-seg.yaml).

!!! example "ultralytics/cfg/datasets/crack-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/crack-seg.yaml"
    ```

## Usage

To train Ultralytics YOLO11n model on the Crack Segmentation dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="crack-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo segment train data=crack-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

The Crack Segmentation dataset comprises a varied collection of images and videos captured from multiple perspectives. Below are instances of data from the dataset, accompanied by their respective annotations:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/crack-segmentation-sample.avif)

- This image presents an example of image object segmentation, featuring annotated bounding boxes with masks outlining identified objects. The dataset includes a diverse array of images taken in different locations, environments, and densities, making it a comprehensive resource for developing models designed for this particular task.

- The example underscores the diversity and complexity found in the Crack segmentation dataset, emphasizing the crucial role of high-quality data in computer vision tasks.

## Citations and Acknowledgments

If you incorporate the crack segmentation dataset into your research or development endeavors, kindly reference the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{ crack-bphdr_dataset,
            title = { crack Dataset },
            type = { Open Source Dataset },
            author = { University },
            howpublished = { \url{ https://universe.roboflow.com/university-bswxt/crack-bphdr } },
            url = { https://universe.roboflow.com/university-bswxt/crack-bphdr },
            journal = { Roboflow Universe },
            publisher = { Roboflow },
            year = { 2022 },
            month = { dec },
            note = { visited on 2024-01-23 },
        }
        ```

We would like to acknowledge the Roboflow team for creating and maintaining the Crack Segmentation dataset as a valuable resource for the road safety and research projects. For more information about the Crack segmentation dataset and its creators, visit the [Crack Segmentation Dataset Page](https://universe.roboflow.com/university-bswxt/crack-bphdr?ref=ultralytics).

## FAQ

### What is the Roboflow Crack Segmentation Dataset?

The [Roboflow Crack Segmentation Dataset](https://universe.roboflow.com/university-bswxt/crack-bphdr?ref=ultralytics) is a comprehensive collection of 4029 static images designed specifically for transportation and public safety studies. It is ideal for tasks such as self-driving car model development and infrastructure maintenance. The dataset includes training, testing, and validation sets, aiding in accurate crack detection and segmentation.

### How do I train a model using the Crack Segmentation Dataset with Ultralytics YOLO11?

To train an Ultralytics YOLO11 model on the Crack Segmentation dataset, use the following code snippets. Detailed instructions and further parameters can be found on the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="crack-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo segment train data=crack-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640
        ```

### Why should I use the Crack Segmentation Dataset for my self-driving car project?

The Crack Segmentation Dataset is exceptionally suited for self-driving car projects due to its diverse collection of 4029 road and wall images, which provide a varied range of scenarios. This diversity enhances the accuracy and robustness of models trained for crack detection, crucial for maintaining road safety and ensuring timely infrastructure repairs.

### What unique features does Ultralytics YOLO offer for crack segmentation?

Ultralytics YOLO offers advanced real-time [object detection](https://www.ultralytics.com/glossary/object-detection), segmentation, and classification capabilities that make it ideal for crack segmentation tasks. Its ability to handle large datasets and complex scenarios ensures high accuracy and efficiency. For example, the model [Training](../../modes/train.md), [Predict](../../modes/predict.md), and [Export](../../modes/export.md) modes cover comprehensive functionalities from training to deployment.

### How do I cite the Roboflow Crack Segmentation Dataset in my research paper?

If you incorporate the Crack Segmentation Dataset into your research, please use the following BibTeX reference:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{ crack-bphdr_dataset,
            title = { crack Dataset },
            type = { Open Source Dataset },
            author = { University },
            howpublished = { \url{ https://universe.roboflow.com/university-bswxt/crack-bphdr } },
            url = { https://universe.roboflow.com/university-bswxt/crack-bphdr },
            journal = { Roboflow Universe },
            publisher = { Roboflow },
            year = { 2022 },
            month = { dec },
            note = { visited on 2024-01-23 },
        }
        ```

This citation format ensures proper accreditation to the creators of the dataset and acknowledges its use in your research.
