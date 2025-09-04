---
comments: true
description: Discover the Signature Detection Dataset for training models to identify and verify human signatures in various documents. Perfect for document verification and fraud prevention.
keywords: Signature Detection Dataset, document verification, fraud detection, computer vision, YOLO11, Ultralytics, annotated signatures, training dataset
---

# Signature Detection Dataset

This dataset focuses on detecting human written signatures within documents. It includes a variety of document types with annotated signatures, providing valuable insights for applications in document verification and fraud detection. Essential for training [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) algorithms, this dataset aids in identifying signatures in various document formats, supporting research and practical applications in document analysis.

## Dataset Structure

The signature detection dataset is split into two subsets:

- **Training set**: Contains 143 images, each with corresponding annotations.
- **Validation set**: Includes 35 images, each with paired annotations.

## Applications

This dataset can be applied in various computer vision tasks such as [object detection](https://www.ultralytics.com/glossary/object-detection), [object tracking](https://docs.ultralytics.com/modes/track/), and document analysis. Specifically, it can be used to train and evaluate models for identifying signatures in documents, which has significant applications in:

- **Document Verification**: Automating the verification process for legal and financial documents
- **Fraud Detection**: Identifying potentially forged or unauthorized signatures
- **Digital Document Processing**: Streamlining workflows in administrative and legal sectors
- **Banking and Finance**: Enhancing security in check processing and loan document verification
- **Archival Research**: Supporting historical document analysis and cataloging

Additionally, it serves as a valuable resource for educational purposes, enabling students and researchers to study signature characteristics across different document types.

## Dataset YAML

A YAML (Yet Another Markup Language) file defines the dataset configuration, including paths and classes information. For the signature detection dataset, the `signature.yaml` file is located at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/signature.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/signature.yaml).

!!! example "ultralytics/cfg/datasets/signature.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/signature.yaml"
    ```

## Usage

To train a YOLO11n model on the signature detection dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the provided code samples. For a comprehensive list of available parameters, refer to the model's [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="signature.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=signature.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

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

The signature detection dataset comprises a wide variety of images showcasing different document types and annotated signatures. Below are examples of images from the dataset, each accompanied by its corresponding annotations.

![Signature detection dataset sample image](https://github.com/ultralytics/docs/releases/download/0/signature-detection-mosaiced-sample.avif)

- **Mosaiced Image**: Here, we present a training batch consisting of mosaiced dataset images. Mosaicing, a training technique, combines multiple images into one, enriching batch diversity. This method helps enhance the model's ability to generalize across different signature sizes, aspect ratios, and contexts.

This example illustrates the variety and complexity of images in the signature Detection Dataset, emphasizing the benefits of including mosaicing during the training process.

## Citations and Acknowledgments

The dataset has been released available under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

## FAQ

### What is the Signature Detection Dataset, and how can it be used?

The Signature Detection Dataset is a collection of annotated images aimed at detecting human signatures within various document types. It can be applied in computer vision tasks such as [object detection](https://www.ultralytics.com/glossary/object-detection) and tracking, primarily for document verification, fraud detection, and archival research. This dataset helps train models to recognize signatures in different contexts, making it valuable for both research and practical applications in [smart document analysis](https://www.ultralytics.com/blog/using-ultralytics-yolo11-for-smart-document-analysis).

### How do I train a YOLO11n model on the Signature Detection Dataset?

To train a YOLO11n model on the Signature Detection Dataset, follow these steps:

1. Download the `signature.yaml` dataset configuration file from [signature.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/signature.yaml).
2. Use the following Python script or CLI command to start training:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo11n.pt")

        # Train the model
        results = model.train(data="signature.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=signature.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

For more details, refer to the [Training](../../modes/train.md) page.

### What are the main applications of the Signature Detection Dataset?

The Signature Detection Dataset can be used for:

1. **Document Verification**: Automatically verifying the presence and authenticity of human signatures in documents.
2. **Fraud Detection**: Identifying forged or fraudulent signatures in legal and financial documents.
3. **Archival Research**: Assisting historians and archivists in the digital analysis and cataloging of historical documents.
4. **Education**: Supporting academic research and teaching in the fields of computer vision and [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml).
5. **Financial Services**: Enhancing security in banking transactions and loan processing by verifying signature authenticity.

### How can I perform inference using a model trained on the Signature Detection Dataset?

To perform inference using a model trained on the Signature Detection Dataset, follow these steps:

1. Load your fine-tuned model.
2. Use the below Python script or CLI command to perform inference:

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

### What is the structure of the Signature Detection Dataset, and where can I find more information?

The Signature Detection Dataset is divided into two subsets:

- **Training Set**: Contains 143 images with annotations.
- **Validation Set**: Includes 35 images with annotations.

For detailed information, you can refer to the [Dataset Structure](#dataset-structure) section. Additionally, view the complete dataset configuration in the `signature.yaml` file located at [signature.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/signature.yaml).
