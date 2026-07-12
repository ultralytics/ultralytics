---
title: Crack-Seg Dataset
comments: true
description: Train Ultralytics YOLO segmentation models on the Crack Segmentation Dataset — 4,029 annotated road and wall images for a single crack class.
keywords: Crack Segmentation Dataset, Ultralytics, transportation safety, public safety, self-driving cars, computer vision, road safety, infrastructure maintenance, dataset, YOLO, segmentation, deep learning
---

# Crack Segmentation Dataset

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-crack-segmentation-dataset.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Crack Segmentation Dataset In Colab"></a>

The [Ultralytics](https://www.ultralytics.com/) Crack Segmentation Dataset provides 4,029 annotated images of cracks on roads and walls for training [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) models on a single `crack` class. Captured across diverse pavement and structural scenarios, it pairs directly with [Ultralytics YOLO](../../models/yolo26.md) for use cases ranging from transportation safety and [self-driving car](https://www.ultralytics.com/blog/ai-in-self-driving-cars) perception to [infrastructure maintenance](https://www.ultralytics.com/blog/using-ai-for-crack-detection-and-segmentation) and structural [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) inspection.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GAFlmuk0fZI"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train a Crack Segmentation Model using Ultralytics YOLO26 | AI in Construction 🎉
</p>

## Dataset Structure

The Crack Segmentation Dataset splits its 4,029 images as follows:

- **Training set**: 3,717 images used for [training](https://www.ultralytics.com/glossary/training-data) the [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) model.
- **Validation set**: 200 images used during training to tune [hyperparameters](../../guides/hyperparameter-tuning.md) and prevent [overfitting](https://www.ultralytics.com/glossary/overfitting).
- **Testing set**: 112 images held out to evaluate the model after training.
- **Classes**: a single `crack` class covering every annotated crack on roads and walls.
- **Download size**: ~91.6 MB.

## Applications

Crack segmentation supports [infrastructure maintenance](https://www.ultralytics.com/blog/using-ai-for-crack-detection-and-segmentation) by identifying and assessing structural damage in buildings, bridges, and roads. It also enhances [road safety](https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries) by letting automated systems detect pavement cracks for timely repairs.

In industrial settings, crack detection with models like [Ultralytics YOLO26](../../models/yolo26.md) helps verify building integrity in construction, prevents costly downtime in [manufacturing](https://www.ultralytics.com/solutions/computer-vision-in-manufacturing), and makes road inspections safer. Automatically classifying cracks lets maintenance teams prioritize the most urgent repairs.

The complete Crack Segmentation Dataset can also be browsed and managed on [Ultralytics Platform](https://platform.ultralytics.com/).

## Dataset YAML

A [YAML](https://www.ultralytics.com/glossary/yaml) (Yet Another Markup Language) file defines the dataset configuration. It includes details about the dataset's paths, classes, and other relevant information. For the Crack Segmentation dataset, the `crack-seg.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/crack-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/crack-seg.yaml).

!!! example "ultralytics/cfg/datasets/crack-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/crack-seg.yaml"
    ```

## Usage

To train the Ultralytics YOLO26n-seg model on the Crack Segmentation dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the following [Python](https://www.python.org/) or CLI snippets. Refer to the model [Training](../../modes/train.md) documentation page for a comprehensive list of available arguments and configurations like [hyperparameter tuning](../../guides/hyperparameter-tuning.md).

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        # Using a pretrained model like yolo26n-seg.pt is recommended for faster convergence
        model = YOLO("yolo26n-seg.pt")

        # Train the model on the Crack Segmentation dataset
        # Ensure 'crack-seg.yaml' is accessible or provide the full path
        results = model.train(data="crack-seg.yaml", epochs=100, imgsz=640)

        # After training, the model can be used for prediction or exported
        # results = model.predict(source='path/to/your/images')
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model using the Command Line Interface
        # Ensure the dataset YAML file 'crack-seg.yaml' is correctly configured and accessible
        yolo segment train data=crack-seg.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

Below is an example from the Crack Segmentation Dataset with its [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) masks overlaid, outlining identified cracks on road and wall surfaces:

![Crack segmentation dataset sample for infrastructure inspection](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/crack-segmentation-sample.avif)

The dataset spans varied locations, surfaces, and lighting conditions, so models trained on it see the range of real-world scenes they need to generalize across. [Data augmentation](https://www.ultralytics.com/glossary/data-augmentation) can broaden that variety further — see our [instance segmentation and tracking guide](../../guides/instance-segmentation-and-tracking.md) for related workflows.

## Citations and Acknowledgments

If you use the Crack Segmentation dataset in your research or development work, please cite the source appropriately:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{ crack-bphdr_dataset,
            title = { crack Dataset },
            type = { Open Source Dataset },
            author = { University },
            url = { https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/crack-seg.yaml },
            year = { 2022 },
            month = { dec },
            note = { visited on 2024-01-23 },
        }
        ```

We acknowledge the team at Roboflow for making the Crack Segmentation dataset available, providing a valuable resource for the computer vision community, particularly for projects related to road safety and infrastructure assessment. For more datasets, visit the [Ultralytics Datasets collection](../index.md).

## FAQ

### What is the Crack Segmentation Dataset, and how is it used in Ultralytics YOLO26?

The **Crack Segmentation Dataset** is a collection of 4,029 annotated images of cracks on roads and walls for training and evaluating [instance segmentation](../../tasks/segment.md) models on a single `crack` class. It's built for transportation-safety and infrastructure applications like structural inspection and pavement assessment, and is used directly with Ultralytics [YOLO26](../../models/yolo26.md) via the `crack-seg.yaml` configuration file.

### How many images and classes does the Crack Segmentation Dataset contain?

The dataset totals 4,029 images — 3,717 for training, 200 for validation, and 112 for testing — all annotated for a single `crack` class. The full archive downloads automatically as a ~91.6 MB `.zip` on first use.

### How do I train an Ultralytics YOLO26 model on the Crack Segmentation Dataset?

Load a pretrained segmentation model (e.g., `yolo26n-seg.pt`) and train it with the `crack-seg.yaml` configuration using the Python or CLI snippets in the [Usage](#usage) section above. See the [Training guide](../../modes/train.md) for the full list of available arguments.

### Why use the Crack Segmentation Dataset for self-driving car and infrastructure projects?

Its diverse images of cracks across roads and walls cover many real-world scenarios, improving the robustness of models trained for crack detection. Accurate segmentation supports [road safety](https://www.ultralytics.com/blog/ai-in-self-driving-cars) and infrastructure-assessment systems that must identify potential hazards reliably — see the [Applications](#applications) section above and our [model training tips](../../guides/model-training-tips.md) for best practices.

### Where can I find the dataset configuration file for Crack Segmentation?

The `crack-seg.yaml` file, which defines the dataset paths and the single `crack` class, is located in the Ultralytics GitHub repository: [crack-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/crack-seg.yaml).
