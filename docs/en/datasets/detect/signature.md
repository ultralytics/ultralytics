---
comments: true
description: Signature Detection Dataset, a leading dataset for detecting signatures in documents, integrates with Ultralytics. Discover ways to use it for training YOLO models.
keywords: Ultralytics, Signature Detection Dataset, object detection, YOLO, YOLO model training, document analysis, computer vision, deep learning models, signature tracking, document verification
---

# Signature Detection Dataset

This dataset focuses on detecting human written signatures within documents. It includes a variety of document types with annotated signatures, providing valuable insights for applications in document verification and fraud detection. Essential for training computer vision algorithms, this dataset aids in identifying signatures in various document formats, supporting research and practical applications in document analysis.

## Dataset Structure

The signature detection dataset is split into three subsets:

- **Training set**: Contains 143 images, each with corresponding annotations.
- **Validation set**: Includes 35 images, each with paired annotations.

## Applications

This dataset can be applied in various computer vision tasks such as object detection, object tracking, and document analysis. Specifically, it can be used to train and evaluate models for identifying signatures in documents, which can have applications in document verification, fraud detection, and archival research. Additionally, it can serve as a valuable resource for educational purposes, enabling students and researchers to study and understand the characteristics and behaviors of signatures in different document types.

## Dataset YAML

A YAML (Yet Another Markup Language) file defines the dataset configuration, including paths and classes information. For the signature detection dataset, the `signature.yaml` file is located at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/signature.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/signature.yaml).

!!! Example "ultralytics/cfg/datasets/signature.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/signature.yaml"
    ```

## Usage

To train a YOLOv8n model on the signature detection dataset for 100 epochs with an image size of 640, use the provided code samples. For a comprehensive list of available parameters, refer to the model's [Training](../../modes/train.md) page.

!!! Example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="signature.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=signature.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

!!! Example "Inference Example"

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

![Signature detection dataset sample image](https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/88a453da-3110-4835-9ae4-97bfb8b19046)

- **Mosaiced Image**: Here, we present a training batch consisting of mosaiced dataset images. Mosaicing, a training technique, combines multiple images into one, enriching batch diversity. This method helps enhance the model's ability to generalize across different signature sizes, aspect ratios, and contexts.

This example illustrates the variety and complexity of images in the signature Detection Dataset, emphasizing the benefits of including mosaicing during the training process.

## Citations and Acknowledgments

The dataset has been released available under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
