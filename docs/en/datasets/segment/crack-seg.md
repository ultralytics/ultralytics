---
comments: true
description: Explore the extensive Crack Segmentation Dataset, ideal for transportation safety, infrastructure maintenance, and self-driving car model development using Ultralytics YOLO.
keywords: Crack Segmentation Dataset, Ultralytics, transportation safety, public safety, self-driving cars, computer vision, road safety, infrastructure maintenance, dataset, YOLO, segmentation, deep learning
---

# Crack Segmentation Dataset

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-crack-segmentation-dataset.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Crack Segmentation Dataset In Colab"></a>

The Crack Segmentation Dataset is an extensive resource designed for individuals involved in transportation and public safety studies. It is also beneficial for developing [self-driving car](https://www.ultralytics.com/blog/ai-in-self-driving-cars) models or exploring various [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) applications. This dataset is part of the broader collection available on the Ultralytics [Datasets Hub](../../datasets/index.md).

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

Comprising 4029 static images captured from diverse road and wall scenarios, this dataset is a valuable asset for crack segmentation tasks. Whether you are researching transportation infrastructure or aiming to enhance the [accuracy](https://www.ultralytics.com/glossary/accuracy) of autonomous driving systems, this dataset provides a rich collection of images for training [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models.

## Dataset Structure

The Crack Segmentation Dataset is organized into three subsets:

- **Training set**: 3717 images with corresponding annotations.
- **Testing set**: 112 images with corresponding annotations.
- **Validation set**: 200 images with corresponding annotations.

## Applications

Crack segmentation finds practical applications in [infrastructure maintenance](https://www.ultralytics.com/blog/using-ai-for-crack-detection-and-segmentation), aiding in the identification and assessment of structural damage in buildings, bridges, and roads. It also plays a crucial role in enhancing [road safety](https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries) by enabling automated systems to detect pavement cracks for timely repairs.

In industrial settings, crack detection using deep learning models like [Ultralytics YOLO26](../../models/yolo26.md) helps ensure building integrity in construction, prevents costly downtimes in [manufacturing](https://www.ultralytics.com/solutions/ai-in-manufacturing), and makes road inspections safer and more effective. Automatically identifying and classifying cracks allows maintenance teams to prioritize repairs efficiently, contributing to better [model evaluation insights](../../guides/model-evaluation-insights.md).

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

The Crack Segmentation dataset contains a diverse collection of images captured from various perspectives, showcasing different types of cracks on roads and walls. Here are some examples:

![Crack segmentation dataset sample for infrastructure inspection](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/crack-segmentation-sample.avif)

- This image demonstrates [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation), featuring annotated [bounding boxes](https://www.ultralytics.com/glossary/bounding-box) with masks outlining identified cracks. The dataset includes images from different locations and environments, making it a comprehensive resource for developing robust models for this task. Techniques like [data augmentation](https://www.ultralytics.com/glossary/data-augmentation) can further enhance dataset diversity. Learn more about instance segmentation and tracking in our [guide](../../guides/instance-segmentation-and-tracking.md).

- The example highlights the diversity within the Crack Segmentation dataset, emphasizing the importance of high-quality data for training effective computer vision models.

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
            year = { 2022 },
            month = { dec },
            note = { visited on 2024-01-23 },
        }
        ```

We acknowledge the team at Roboflow for making the Crack Segmentation dataset available, providing a valuable resource for the computer vision community, particularly for projects related to road safety and infrastructure assessment.

## FAQ

### What is the Crack Segmentation Dataset?

The Crack Segmentation Dataset is a collection of 4029 static images designed for transportation and public safety studies. It's suitable for tasks like [self-driving car](https://www.ultralytics.com/blog/ai-in-self-driving-cars) model development and [infrastructure maintenance](https://www.ultralytics.com/blog/using-ai-for-crack-detection-and-segmentation). It includes training, testing, and validation sets for crack detection and [segmentation](../../tasks/segment.md) tasks.

### How do I train a model using the Crack Segmentation Dataset with Ultralytics YOLO26?

To train an [Ultralytics YOLO26](../../models/yolo26.md) model on this dataset, use the provided Python or CLI examples. Detailed instructions and parameters are available on the model [Training](../../modes/train.md) page. You can manage your training process using tools like [Ultralytics Platform](https://platform.ultralytics.com).

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model (recommended)
        model = YOLO("yolo26n-seg.pt")

        # Train the model
        results = model.train(data="crack-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained model via CLI
        yolo segment train data=crack-seg.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
        ```

### Why use the Crack Segmentation Dataset for self-driving car projects?

This dataset is valuable for self-driving car projects due to its diverse images of roads and walls, covering various real-world scenarios. This diversity improves the robustness of models trained for crack detection, which is crucial for road safety and infrastructure assessment. The detailed annotations aid in [developing models](../../guides/model-training-tips.md) that can accurately identify potential road hazards.

### What features does Ultralytics YOLO offer for crack segmentation?

Ultralytics YOLO provides real-time [object detection](https://www.ultralytics.com/glossary/object-detection), segmentation, and classification capabilities, making it highly suitable for crack segmentation tasks. It efficiently handles large datasets and complex scenarios. The framework includes comprehensive modes for [Training](../../modes/train.md), [Prediction](../../modes/predict.md), and [Exporting](../../modes/export.md) models. YOLO's [anchor-free detection](https://www.ultralytics.com/blog/benefits-ultralytics-yolo11-being-anchor-free-detector) approach can improve performance on irregular shapes like cracks, and performance can be measured using standard [metrics](../../guides/yolo-performance-metrics.md).

### How do I cite the Crack Segmentation Dataset?

If using this dataset in your work, please cite it using the provided BibTeX entry above to give appropriate credit to the creators.
