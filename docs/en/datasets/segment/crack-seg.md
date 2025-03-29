---
comments: true
description: Explore the extensive Crack Segmentation Dataset, perfect for transportation and public safety studies or self-driving car model development using Ultralytics YOLO.
keywords: Crack Segmentation Dataset, Ultralytics, transportation safety, public safety, self-driving cars, computer vision, road safety, infrastructure maintenance, dataset, YOLO, segmentation
---

# Crack Segmentation Dataset

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-crack-segmentation-dataset.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Crack Segmentation Dataset In Colab"></a>

The [Crack Segmentation Dataset](https://universe.roboflow.com/university-bswxt/crack-bphdr?ref=ultralytics) is an extensive resource designed for individuals involved in transportation and public safety studies. It is also beneficial for developing [self-driving car](https://www.ultralytics.com/blog/ai-in-self-driving-cars) models or exploring [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) applications.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/C4mc40YKm-g"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Crack segmentation using Ultralytics YOLOv9.
</p>

Comprising 4029 static images captured from diverse road and wall scenarios, this dataset is a valuable asset for crack segmentation tasks. Whether you are researching transportation infrastructure or aiming to enhance the [accuracy](https://www.ultralytics.com/glossary/accuracy) of autonomous driving systems, this dataset provides a rich collection of images.

## Dataset Structure

The Crack Segmentation Dataset is divided as follows:

- **Training set**: 3717 images with corresponding annotations.
- **Testing set**: 112 images with corresponding annotations.
- **Validation set**: 200 images with corresponding annotations.

## Applications

Crack segmentation finds practical applications in [infrastructure maintenance](https://www.ultralytics.com/blog/using-ai-for-crack-detection-and-segmentation), aiding in the identification and assessment of structural damage. It also plays a crucial role in enhancing road safety by enabling automated systems to detect pavement cracks for timely repairs.

In industrial settings, crack detection using [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models like [Ultralytics YOLO](https://docs.ultralytics.com/models/yolov8/) helps ensure building integrity in construction, prevents costly downtimes in [manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing), and makes road inspections safer and more effective. Automatically identifying and classifying cracks allows maintenance teams to prioritize repairs efficiently.

## Dataset YAML

A YAML (Yet Another Markup Language) file defines the dataset configuration. It includes details about paths, classes, and other relevant information. For the Crack Segmentation dataset, the `crack-seg.yaml` file is located at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/crack-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/crack-seg.yaml).

!!! example "ultralytics/cfg/datasets/crack-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/crack-seg.yaml"
    ```

## Usage

To train the Ultralytics YOLO11n model on the Crack Segmentation dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the following code snippets. Refer to the model [Training](../../modes/train.md) page for a comprehensive list of available arguments.

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

The Crack Segmentation dataset contains a diverse collection of images captured from various perspectives. Here are some examples:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/crack-segmentation-sample.avif)

- This image demonstrates [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation), featuring annotated [bounding boxes](https://www.ultralytics.com/glossary/bounding-box) with masks outlining identified cracks. The dataset includes images from different locations and environments, making it a comprehensive resource for developing models for this task. Learn more about instance segmentation and tracking in our [guide](https://docs.ultralytics.com/guides/instance-segmentation-and-tracking/).

- The example highlights the diversity within the Crack Segmentation dataset, emphasizing the importance of high-quality data for [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks.

## Citations and Acknowledgments

If you use the Crack Segmentation dataset in your research or development work, please cite the source appropriately:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{ crack-bphdr_dataset,
            title = { crack Dataset },
            type = { Open Source Dataset },
            author = { University },
            url = { https://universe.roboflow.com/university-bswxt/crack-bphdr },
            journal = { Roboflow Universe },
            publisher = { Roboflow },
            year = { 2022 },
            month = { dec },
            note = { visited on 2024-01-23 },
        }
        ```

We acknowledge the Roboflow team for creating and maintaining the Crack Segmentation dataset, providing a valuable resource for road safety and research projects.

## FAQ

### What is the Crack Segmentation Dataset?

The Crack Segmentation Dataset is a collection of 4029 static images designed for transportation and public safety studies. It's suitable for tasks like [self-driving car](https://www.ultralytics.com/blog/ai-in-self-driving-cars) model development and [infrastructure maintenance](https://www.ultralytics.com/blog/using-ai-for-crack-detection-and-segmentation). It includes training, testing, and validation sets for crack detection and [segmentation](https://docs.ultralytics.com/tasks/segment/).

### How do I train a model using the Crack Segmentation Dataset with Ultralytics YOLO11?

To train an Ultralytics YOLO11 model on this dataset, use the provided Python or CLI examples. Detailed instructions and parameters are available on the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-seg.pt")  # load a pretrained model

        # Train the model
        results = model.train(data="crack-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained model
        yolo segment train data=crack-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640
        ```

### Why use the Crack Segmentation Dataset for self-driving car projects?

This dataset is ideal for self-driving car projects due to its diverse images of roads and walls, covering various scenarios. This diversity improves the robustness of models trained for crack detection, crucial for road safety and infrastructure assessment. The detailed annotations aid in [developing models](https://docs.ultralytics.com/guides/model-training-tips/) that identify potential road hazards.

### What features does Ultralytics YOLO offer for crack segmentation?

Ultralytics YOLO provides real-time [object detection](https://www.ultralytics.com/glossary/object-detection), segmentation, and classification, making it suitable for crack segmentation. It handles large datasets and complex scenarios efficiently. The [Training](../../modes/train.md), [Predict](../../modes/predict.md), and [Export](../../modes/export.md) modes offer comprehensive functionality. YOLO's [anchor-free detection](https://www.ultralytics.com/blog/benefits-ultralytics-yolo11-being-anchor-free-detector) approach can improve performance on irregular shapes like cracks.

### How do I cite the Crack Segmentation Dataset?

If using this dataset, please cite it using the provided BibTeX entry:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{ crack-bphdr_dataset,
            title = { crack Dataset },
            type = { Open Source Dataset },
            author = { University },
            url = { https://universe.roboflow.com/university-bswxt/crack-bphdr },
            journal = { Roboflow Universe },
            publisher = { Roboflow },
            year = { 2022 },
            month = { dec },
            note = { visited on 2024-01-23 },
        }
        ```

This ensures proper credit to the dataset creators.
