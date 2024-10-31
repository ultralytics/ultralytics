---
comments: true
description: Explore the comprehensive Open Images V7 dataset by Google. Learn about its annotations, applications, and use YOLO11 pretrained models for computer vision tasks.
keywords: Open Images V7, Google dataset, computer vision, YOLO11 models, object detection, image segmentation, visual relationships, AI research, Ultralytics
---

# Open Images V7 Dataset

[Open Images V7](https://storage.googleapis.com/openimages/web/index.html) is a versatile and expansive dataset championed by Google. Aimed at propelling research in the realm of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv), it boasts a vast collection of images annotated with a plethora of data, including image-level labels, object bounding boxes, object segmentation masks, visual relationships, and localized narratives.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/u3pLlgzUeV8"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> <a href="https://www.ultralytics.com/glossary/object-detection">Object Detection</a> using OpenImagesV7 Pretrained Model
</p>

## Open Images V7 Pretrained Models

| Model                                                                                     | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ----------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-oiv7.pt) | 640                   | 18.4                 | 142.4                          | 1.21                                | 3.5                | 10.5              |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-oiv7.pt) | 640                   | 27.7                 | 183.1                          | 1.40                                | 11.4               | 29.7              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-oiv7.pt) | 640                   | 33.6                 | 408.5                          | 2.26                                | 26.2               | 80.6              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-oiv7.pt) | 640                   | 34.9                 | 596.9                          | 2.43                                | 44.1               | 167.4             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-oiv7.pt) | 640                   | 36.3                 | 860.6                          | 3.56                                | 68.7               | 260.6             |

You can use these pretrained for inference or fine-tuning as follows.

!!! example "Pretrained Model Usage Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load an Open Images Dataset V7 pretrained YOLOv8n model
        model = YOLO("yolov8n-oiv7.pt")

        # Run prediction
        results = model.predict(source="image.jpg")

        # Start training from the pretrained checkpoint
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Predict using an Open Images Dataset V7 pretrained model
        yolo detect predict source=image.jpg model=yolov8n-oiv7.pt

        # Start training from an Open Images Dataset V7 pretrained checkpoint
        yolo detect train data=coco8.yaml model=yolov8n-oiv7.pt epochs=100 imgsz=640
        ```

![Open Images V7 classes visual](https://github.com/ultralytics/docs/releases/download/0/open-images-v7-classes-visual.avif)

## Key Features

- Encompasses ~9M images annotated in various ways to suit multiple computer vision tasks.
- Houses a staggering 16M bounding boxes across 600 object classes in 1.9M images. These boxes are primarily hand-drawn by experts ensuring high [precision](https://www.ultralytics.com/glossary/precision).
- Visual relationship annotations totaling 3.3M are available, detailing 1,466 unique relationship triplets, object properties, and human activities.
- V5 introduced segmentation masks for 2.8M objects across 350 classes.
- V6 introduced 675k localized narratives that amalgamate voice, text, and mouse traces highlighting described objects.
- V7 introduced 66.4M point-level labels on 1.4M images, spanning 5,827 classes.
- Encompasses 61.4M image-level labels across a diverse set of 20,638 classes.
- Provides a unified platform for image classification, object detection, relationship detection, [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation), and multimodal image descriptions.

## Dataset Structure

Open Images V7 is structured in multiple components catering to varied computer vision challenges:

- **Images**: About 9 million images, often showcasing intricate scenes with an average of 8.3 objects per image.
- **Bounding Boxes**: Over 16 million boxes that demarcate objects across 600 categories.
- **Segmentation Masks**: These detail the exact boundary of 2.8M objects across 350 classes.
- **Visual Relationships**: 3.3M annotations indicating object relationships, properties, and actions.
- **Localized Narratives**: 675k descriptions combining voice, text, and mouse traces.
- **Point-Level Labels**: 66.4M labels across 1.4M images, suitable for zero/few-shot [semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation).

## Applications

Open Images V7 is a cornerstone for training and evaluating state-of-the-art models in various computer vision tasks. The dataset's broad scope and high-quality annotations make it indispensable for researchers and developers specializing in computer vision.

## Dataset YAML

Typically, datasets come with a YAML (Yet Another Markup Language) file that delineates the dataset's configuration. For the case of Open Images V7, a hypothetical `OpenImagesV7.yaml` might exist. For accurate paths and configurations, one should refer to the dataset's official repository or documentation.

!!! example "OpenImagesV7.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/open-images-v7.yaml"
    ```

## Usage

To train a YOLO11n model on the Open Images V7 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! warning

    The complete Open Images V7 dataset comprises 1,743,042 training images and 41,620 validation images, requiring approximately **561 GB of storage space** upon download.

    Executing the commands provided below will trigger an automatic download of the full dataset if it's not already present locally. Before running the below example it's crucial to:

    - Verify that your device has enough storage capacity.
    - Ensure a robust and speedy internet connection.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLO11n model
        model = YOLO("yolo11n.pt")

        # Train the model on the Open Images V7 dataset
        results = model.train(data="open-images-v7.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Train a COCO-pretrained YOLO11n model on the Open Images V7 dataset
        yolo detect train data=open-images-v7.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

Illustrations of the dataset help provide insights into its richness:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/oidv7-all-in-one-example-ab.avif)

- **Open Images V7**: This image exemplifies the depth and detail of annotations available, including bounding boxes, relationships, and segmentation masks.

Researchers can gain invaluable insights into the array of computer vision challenges that the dataset addresses, from basic object detection to intricate relationship identification.

## Citations and Acknowledgments

For those employing Open Images V7 in their work, it's prudent to cite the relevant papers and acknowledge the creators:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{OpenImages,
          author = {Alina Kuznetsova and Hassan Rom and Neil Alldrin and Jasper Uijlings and Ivan Krasin and Jordi Pont-Tuset and Shahab Kamali and Stefan Popov and Matteo Malloci and Alexander Kolesnikov and Tom Duerig and Vittorio Ferrari},
          title = {The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale},
          year = {2020},
          journal = {IJCV}
        }
        ```

A heartfelt acknowledgment goes out to the Google AI team for creating and maintaining the Open Images V7 dataset. For a deep dive into the dataset and its offerings, navigate to the [official Open Images V7 website](https://storage.googleapis.com/openimages/web/index.html).

## FAQ

### What is the Open Images V7 dataset?

Open Images V7 is an extensive and versatile dataset created by Google, designed to advance research in computer vision. It includes image-level labels, object bounding boxes, object segmentation masks, visual relationships, and localized narratives, making it ideal for various computer vision tasks such as object detection, segmentation, and relationship detection.

### How do I train a YOLO11 model on the Open Images V7 dataset?

To train a YOLO11 model on the Open Images V7 dataset, you can use both Python and CLI commands. Here's an example of training the YOLO11n model for 100 epochs with an image size of 640:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLO11n model
        model = YOLO("yolo11n.pt")

        # Train the model on the Open Images V7 dataset
        results = model.train(data="open-images-v7.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # Train a COCO-pretrained YOLO11n model on the Open Images V7 dataset
        yolo detect train data=open-images-v7.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

For more details on arguments and settings, refer to the [Training](../../modes/train.md) page.

### What are some key features of the Open Images V7 dataset?

The Open Images V7 dataset includes approximately 9 million images with various annotations:

- **Bounding Boxes**: 16 million bounding boxes across 600 object classes.
- **Segmentation Masks**: Masks for 2.8 million objects across 350 classes.
- **Visual Relationships**: 3.3 million annotations indicating relationships, properties, and actions.
- **Localized Narratives**: 675,000 descriptions combining voice, text, and mouse traces.
- **Point-Level Labels**: 66.4 million labels across 1.4 million images.
- **Image-Level Labels**: 61.4 million labels across 20,638 classes.

### What pretrained models are available for the Open Images V7 dataset?

Ultralytics provides several YOLOv8 pretrained models for the Open Images V7 dataset, each with different sizes and performance metrics:

| Model                                                                                     | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ----------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-oiv7.pt) | 640                   | 18.4                 | 142.4                          | 1.21                                | 3.5                | 10.5              |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-oiv7.pt) | 640                   | 27.7                 | 183.1                          | 1.40                                | 11.4               | 29.7              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-oiv7.pt) | 640                   | 33.6                 | 408.5                          | 2.26                                | 26.2               | 80.6              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-oiv7.pt) | 640                   | 34.9                 | 596.9                          | 2.43                                | 44.1               | 167.4             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-oiv7.pt) | 640                   | 36.3                 | 860.6                          | 3.56                                | 68.7               | 260.6             |

### What applications can the Open Images V7 dataset be used for?

The Open Images V7 dataset supports a variety of computer vision tasks including:

- **[Image Classification](https://www.ultralytics.com/glossary/image-classification)**
- **Object Detection**
- **Instance Segmentation**
- **Visual Relationship Detection**
- **Multimodal Image Descriptions**

Its comprehensive annotations and broad scope make it suitable for training and evaluating advanced [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models, as highlighted in practical use cases detailed in our [applications](#applications) section.
