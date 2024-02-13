---
comments: true
description: Explore the Crack Segmentation using Ultralytics YOLOv8 Dataset, a large-scale benchmark for road safety analysis, and learn how to train a YOLO model using it.
keywords: Crack Segmentation Dataset, Ultralytics, road cracks monitoring, YOLO model, object detection, object tracking, road safety
---

# Roboflow Universe Crack Segmentation Dataset

The [Roboflow](https://roboflow.com/?ref=ultralytics) [Crack Segmentation Dataset](https://universe.roboflow.com/university-bswxt/crack-bphdr) stands out as an extensive resource designed specifically for individuals involved in transportation and public safety studies. It is equally beneficial for those working on the development of self-driving car models or simply exploring computer vision applications for recreational purposes.

Comprising a total of 4029 static images captured from diverse road and wall scenarios, this dataset emerges as a valuable asset for tasks related to crack segmentation. Whether you are delving into the intricacies of transportation research or seeking to enhance the accuracy of your self-driving car models, this dataset provides a rich and varied collection of images to support your endeavors.

## Dataset Structure

The division of data within the Crack Segmentation Dataset is outlined as follows:

- **Training set**: Consists of 3717 images with corresponding annotations.
- **Testing set**: Comprises 112 images along with their respective annotations.
- **Validation set**: Includes 200 images with their corresponding annotations.

## Applications

Crack segmentation finds practical applications in infrastructure maintenance, aiding in the identification and assessment of structural damage. It also plays a crucial role in enhancing road safety by enabling automated systems to detect and address pavement cracks for timely repairs.

## Dataset YAML

A YAML (Yet Another Markup Language) file is employed to outline the configuration of the dataset, encompassing details about paths, classes, and other pertinent information. Specifically, for the Crack Segmentation dataset, the `crack-seg.yaml` file is managed and accessible at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/crack-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/crack-seg.yaml).

!!! Example "ultralytics/cfg/datasets/crack-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/crack-seg.yaml"
    ```

## Usage

To train Ultralytics YOLOv8n model on the Crack Segmentation dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! Example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data='crack-seg.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo segment train data=crack-seg.yaml model=yolov8n-seg.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

The Crack Segmentation dataset comprises a varied collection of images and videos captured from multiple perspectives. Below are instances of data from the dataset, accompanied by their respective annotations:

![Dataset sample image](https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/40ccc20a-9593-412f-b028-643d4a904d0e)

- This image presents an example of image object segmentation, featuring annotated bounding boxes with masks outlining identified objects. The dataset includes a diverse array of images taken in different locations, environments, and densities, making it a comprehensive resource for developing models designed for this particular task.

- The example underscores the diversity and complexity found in the Crack segmentation dataset, emphasizing the crucial role of high-quality data in computer vision tasks.

## Citations and Acknowledgments

If you incorporate the crack segmentation dataset into your research or development endeavors, kindly reference the following paper:

!!! Quote ""

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

We would like to acknowledge the Roboflow team for creating and maintaining the Crack Segmentation dataset as a valuable resource for the road safety and research projects. For more information about the Crack segmentation dataset and its creators, visit the [Crack Segmentation Dataset Page](https://universe.roboflow.com/university-bswxt/crack-bphdr).
