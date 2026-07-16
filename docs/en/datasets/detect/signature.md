---
title: Signature Detection Dataset
comments: true
creator:
    name: Ultralytics
    url: https://www.ultralytics.com/
license:
    name: AGPL-3.0
    url: https://www.ultralytics.com/license
description: The Ultralytics Signature Detection Dataset provides 143 training and 35 validation document images with one signature class for YOLO object detection models.
keywords: Signature Detection Dataset, signature detection, document verification, fraud detection, object detection, computer vision, YOLO26, Ultralytics, annotated signatures, document analysis
---

# Signature Detection Dataset

The Ultralytics Signature Detection Dataset is an [object detection](../../tasks/detect.md) dataset of 178 document images annotated with a single `signature` class, pre-split into 143 training and 35 validation images. The dataset downloads automatically (11.3 MB) the first time you train, making it a compact starting point for [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) applications such as document verification, fraud detection, and digital document processing.

## Dataset Structure

The dataset contains 178 images of various document types with handwritten signatures, split into two subsets:

| Split      | Images | Description                                          |
| ---------- | ------ | ---------------------------------------------------- |
| Train      | 143    | Labeled images for model training                    |
| Validation | 35     | Held-out images for [evaluation](../../modes/val.md) |

Every image carries bounding-box annotations for one class, `signature`, and the configuration defines no separate test split.

!!! tip "Automatic download"

    The Signature Detection Dataset (11.3 MB) downloads automatically from Ultralytics GitHub assets the first time you train, so no manual download or preparation is required.

Explore [Signature on Ultralytics Platform](https://platform.ultralytics.com/ultralytics/datasets/signature) to browse the images with their annotation overlays, view the class distribution and bounding-box heatmaps in the **Charts** tab, and clone it to train your own model in the cloud.

## Applications

A model trained on this dataset can identify and [track](../../modes/track.md) signatures in scanned documents and video, supporting:

- **Document Verification**: Automating signature checks in legal and financial documents
- **Fraud Detection**: Identifying potentially forged or unauthorized signatures
- **Digital Document Processing**: Streamlining workflows in administrative and legal sectors
- **Banking and Finance**: Enhancing security in check processing and loan document verification
- **Archival Research**: Supporting historical document analysis and cataloging
- **Education and Research**: Studying signature characteristics across document types in computer vision courses

## Dataset YAML

The `signature.yaml` file defines the dataset configuration — the dataset paths, class names, and other metadata. It is maintained in the Ultralytics repository at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/signature.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/signature.yaml).

!!! example "ultralytics/cfg/datasets/signature.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/signature.yaml"
    ```

## Usage

To train a [YOLO26n](../../models/yolo26.md) model on the Signature Detection Dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the provided code samples. For a comprehensive list of available parameters, refer to the model's [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="signature.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=signature.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

Once trained, you can run [inference](../../modes/predict.md) on documents or video with the fine-tuned model. The example below runs prediction on a sample video with a confidence threshold of 0.75:

!!! example "Inference Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/best.pt")  # load a signature-detection fine-tuned model

        # Inference using the model
        results = model.predict("https://ultralytics.com/assets/signature-s.mp4", conf=0.75)
        ```

    === "CLI"

        ```bash
        # Start prediction with a finetuned *.pt model
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/signature-s.mp4" conf=0.75
        ```

## Sample Images and Annotations

The dataset covers a variety of document formats, helping trained models generalize across contracts, forms, and letters. Below is a training batch from the dataset:

![Signature detection dataset sample image](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/signature-detection-mosaiced-sample.avif)

- **Mosaiced Image**: Here, we present a training batch consisting of mosaiced dataset images. Mosaicing, a training technique, combines multiple images into one, enriching batch diversity. This method helps enhance the model's ability to generalize across different signature sizes, aspect ratios, and contexts.

## Citations and Acknowledgments

The dataset has been made available under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

If you use the Signature Detection Dataset in your research or development work, please cite it appropriately:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @dataset{Ultralytics_Signature_Detection_Dataset_2024,
            author = {Ultralytics},
            title = {Signature Detection Dataset},
            year = {2024},
            publisher = {Ultralytics},
            url = {https://docs.ultralytics.com/datasets/detect/signature/}
        }
        ```

## FAQ

### What is the Signature Detection Dataset used for?

The Signature Detection Dataset is a collection of 178 annotated document images for training models to detect handwritten signatures. It supports document verification, fraud detection, and archival research, and is a practical base for building [smart document analysis](https://www.ultralytics.com/blog/using-ultralytics-yolo11-for-smart-document-analysis) systems with [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml).

### How do I download the Signature Detection Dataset?

The dataset downloads automatically (11.3 MB) from Ultralytics GitHub assets the first time you train with `data="signature.yaml"` — no manual download is required. To explore other datasets, browse the [detection datasets overview](index.md).

### How many images and classes are in the Signature Detection Dataset?

The Signature Detection Dataset contains 143 training and 35 validation images — 178 in total — each annotated with a single class, `signature`. There is no separate test split. See the [Dataset Structure](#dataset-structure) section and the [`signature.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/signature.yaml) configuration for details.

### How do I train a YOLO26n model on the Signature Detection Dataset?

You can train a YOLO26n model for 100 epochs with an image size of 640 using Python or the CLI:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n.pt")

        # Train the model
        results = model.train(data="signature.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=signature.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

For more details, refer to the [Training](../../modes/train.md) page and [model training tips](../../guides/model-training-tips.md).

### How can I run inference with a model trained on the Signature Detection Dataset?

Load your fine-tuned weights and run [prediction](../../modes/predict.md):

!!! example "Inference Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the fine-tuned model
        model = YOLO("path/to/best.pt")

        # Perform inference
        results = model.predict("https://ultralytics.com/assets/signature-s.mp4", conf=0.75)
        ```

    === "CLI"

        ```bash
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/signature-s.mp4" conf=0.75
        ```

### Can I use the Signature Detection Dataset in commercial projects?

The dataset is released under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), which permits commercial use provided derivative works — including software offered over a network — are made available under the same license. For licensing options that remove the open-source requirements, see [Ultralytics Licensing](https://www.ultralytics.com/license).
