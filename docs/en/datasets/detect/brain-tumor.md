---
title: Brain Tumor Detection Dataset
comments: true
description: Train YOLO26 object detection on the Ultralytics Brain Tumor dataset — 1,116 MRI/CT scans across 2 classes for medical imaging and early diagnosis.
keywords: brain tumor dataset, MRI scans, CT scans, brain tumor detection, medical imaging, AI in healthcare, computer vision, object detection, YOLO26, early diagnosis
---

# Brain Tumor Dataset

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-brain-tumor-detection-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Brain Tumor Dataset In Colab"></a>

The Ultralytics Brain Tumor dataset is an [object detection](https://www.ultralytics.com/glossary/object-detection) dataset of 1,116 medical images (893 for training and 223 for validation) from MRI and CT scans, labeled across 2 classes: `negative` (no tumor) and `positive` (tumor present). It lets you train [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models to locate brain tumors in scans, supporting early diagnosis and treatment planning in [healthcare applications](https://www.ultralytics.com/solutions/computer-vision-in-healthcare).

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Jj7WpfiegD0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Brain Tumor Detection using Ultralytics Platform with Ultralytics YOLO26 | Object Detection 🚀
</p>

## Dataset Structure

The brain tumor dataset contains 1,116 images split into two predefined subsets, defined by the `brain-tumor.yaml` configuration:

| Split      | Images | Annotations |
| ---------- | ------ | ----------- |
| Train      | 893    | Yes         |
| Validation | 223    | Yes         |

Every image is labeled with one of 2 classes:

- **`negative`**: images without a brain tumor
- **`positive`**: images showing a brain tumor

The dataset downloads automatically (4.21 MB) from Ultralytics GitHub assets the first time you train, so no manual setup is required.

Explore [Brain Tumor on Ultralytics Platform](https://platform.ultralytics.com/ultralytics/datasets/brain-tumor) to browse the images with their annotation overlays, view the class distribution and bounding-box heatmaps in the **Charts** tab, and clone it to train your own model in the cloud.

## Applications

Brain tumor detection with [computer vision](../../tasks/detect.md) enables [early diagnosis](https://www.ultralytics.com/blog/ai-and-radiology-a-new-era-of-precision-and-efficiency), treatment planning, and tumor-progression monitoring. By analyzing MRI or CT scans, detection models accurately locate tumors, supporting timely medical intervention and personalized treatment.

Medical professionals can leverage this technology to:

- Reduce diagnostic time and improve accuracy
- Assist in surgical planning by precisely locating tumors
- Monitor treatment effectiveness over time
- Support research in oncology and neurology

## Dataset YAML

A YAML file defines the dataset configuration, including paths, classes, and other relevant information. For the brain tumor dataset, the `brain-tumor.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/brain-tumor.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/brain-tumor.yaml).

!!! example "ultralytics/cfg/datasets/brain-tumor.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/brain-tumor.yaml"
    ```

## Usage

To train a [YOLO26](../../models/yolo26.md) model on the brain tumor dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, utilize the provided code snippets. For a detailed list of available arguments, consult the model's [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="brain-tumor.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=brain-tumor.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

!!! example "Inference Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/best.pt")  # load a brain-tumor fine-tuned model

        # Inference using the model
        results = model.predict("https://ultralytics.com/assets/brain-tumor-sample.jpg")
        ```

    === "CLI"

        ```bash
        # Start prediction with a finetuned *.pt model
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/brain-tumor-sample.jpg"
        ```

## Sample Images and Annotations

The brain tumor dataset contains MRI and CT brain scans with and without tumors. Below is an example image from the dataset with its annotations.

![Brain tumor dataset sample image](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/brain-tumor-dataset-sample-image.avif)

- **Mosaiced Image**: This training batch shows mosaiced dataset images. Mosaicing combines multiple images into one during training, increasing batch diversity so the model generalizes better across tumor sizes, shapes, and locations for [medical image analysis](https://www.ultralytics.com/blog/using-yolo11-for-tumor-detection-in-medical-imaging).

## Citations and Acknowledgments

The dataset has been made available under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

If you use this dataset in your research or development work, please cite it appropriately:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @dataset{Ultralytics_Brain_Tumor_Dataset_2023,
            author = {Ultralytics},
            title = {Brain Tumor Detection Dataset},
            year = {2023},
            publisher = {Ultralytics},
            url = {https://docs.ultralytics.com/datasets/detect/brain-tumor/}
        }
        ```

## FAQ

### What is the structure of the brain tumor dataset available in Ultralytics documentation?

The brain tumor dataset contains 1,116 images divided into two subsets: a **training set** of 893 images and a **validation set** of 223 images, each with paired annotations. This structured division supports developing robust and accurate computer vision models for detecting brain tumors. For more information, see the [Dataset Structure](#dataset-structure) section.

### What classes does the brain tumor dataset contain?

The brain tumor dataset has 2 classes: `negative` (images without a brain tumor) and `positive` (images showing a brain tumor). This binary labeling lets a detection model both locate a tumor and flag scans where none is present.

### How do I download the brain tumor dataset?

The brain tumor dataset (4.21 MB) downloads automatically from Ultralytics GitHub assets the first time you train with `data="brain-tumor.yaml"` — no manual download is required. You can browse related datasets in the [detection datasets overview](index.md).

### How can I train a YOLO26 model on the brain tumor dataset using Ultralytics?

You can train a YOLO26 model on the brain tumor dataset for 100 epochs with an image size of 640px using both Python and CLI methods. Below are the examples for both:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="brain-tumor.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=brain-tumor.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

For a detailed list of available arguments, refer to the [Training](../../modes/train.md) page.

### What are the benefits of using the brain tumor dataset for AI in healthcare?

Using the brain tumor dataset in AI projects enables early diagnosis and treatment planning for brain tumors. It helps in automating brain tumor identification through computer vision, facilitating accurate and timely medical interventions, and supporting personalized treatment strategies. This application holds significant potential in improving patient outcomes and medical efficiencies. For more insights on AI applications in healthcare, see [Ultralytics' healthcare solutions](https://www.ultralytics.com/solutions/computer-vision-in-healthcare).

### How do I perform inference using a fine-tuned YOLO26 model on the brain tumor dataset?

Inference using a fine-tuned YOLO26 model can be performed with either Python or CLI approaches. Here are the examples:

!!! example "Inference Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/best.pt")  # load a brain-tumor fine-tuned model

        # Inference using the model
        results = model.predict("https://ultralytics.com/assets/brain-tumor-sample.jpg")
        ```

    === "CLI"

        ```bash
        # Start prediction with a finetuned *.pt model
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/brain-tumor-sample.jpg"
        ```

### Where can I find the YAML configuration for the brain tumor dataset?

The YAML configuration file for the brain tumor dataset can be found at [brain-tumor.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/brain-tumor.yaml). This file includes paths, classes, and additional relevant information necessary for training and evaluating models on this dataset.
