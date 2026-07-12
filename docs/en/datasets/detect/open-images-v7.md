---
title: Open Images V7 Detection Dataset
comments: true
description: Open Images V7 is a Google object detection dataset with 1,743,042 training and 41,620 validation images across 601 classes. Train Ultralytics YOLO on it.
keywords: Open Images V7, Google dataset, object detection, 601 classes, YOLOv8 oiv7, pretrained models, computer vision, image segmentation, visual relationships, Ultralytics
---

# Open Images V7 Dataset

[Open Images V7](https://storage.googleapis.com/openimages/web/index.html) is a large-scale [object detection](../../tasks/detect.md) dataset created by Google, with 1,743,042 training images and 41,620 validation images across 601 object classes in the Ultralytics configuration. The upstream release spans roughly 9 million images annotated with image-level labels, bounding boxes, segmentation masks, visual relationships, and localized narratives, making it one of the broadest annotation resources in [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv).

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/u3pLlgzUeV8"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Object Detection Using an Open Images V7 Pretrained Model
</p>

## Open Images V7 Pretrained Models

Ultralytics publishes five [YOLOv8](../../models/yolov8.md) models pretrained on Open Images V7, so you can detect its 601 classes without downloading the dataset:

| Model                                                                                     | size<br><sup>(pixels)</sup> | mAP<sup>val<br>50-95</sup> | Speed<br><sup>CPU ONNX<br>(ms)</sup> | Speed<br><sup>A100 TensorRT<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ----------------------------------------------------------------------------------------- | --------------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- | ------------------------ | ----------------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-oiv7.pt) | 640                         | 18.4                       | 142.4                                | 1.21                                      | 3.5                      | 10.5                    |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-oiv7.pt) | 640                         | 27.7                       | 183.1                                | 1.40                                      | 11.4                     | 29.7                    |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m-oiv7.pt) | 640                         | 33.6                       | 408.5                                | 2.26                                      | 26.2                     | 80.6                    |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8l-oiv7.pt) | 640                         | 34.9                       | 596.9                                | 2.43                                      | 44.1                     | 167.4                   |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8x-oiv7.pt) | 640                         | 36.3                       | 860.6                                | 3.56                                      | 68.7                     | 260.6                   |

The visualization below shows the range of object classes these models can detect:

![Open Images V7 classes visual](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/open-images-v7-classes-visual.avif)

You can run [prediction](../../modes/predict.md) or start fine-tuning from these checkpoints as follows.

!!! example "Pretrained Model Usage Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load an Open Images V7 pretrained YOLOv8n model
        model = YOLO("yolov8n-oiv7.pt")

        # Run prediction
        results = model.predict(source="image.jpg")

        # Start training from the pretrained checkpoint
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Predict using an Open Images V7 pretrained model
        yolo detect predict source=image.jpg model=yolov8n-oiv7.pt

        # Start training from an Open Images V7 pretrained checkpoint
        yolo detect train data=coco8.yaml model=yolov8n-oiv7.pt epochs=100 imgsz=640
        ```

## Key Features

- Open Images V7 contains about 9 million images annotated for multiple computer vision tasks, averaging 8.3 objects per image.
- Roughly 90% of its 16 million bounding boxes on 1.9 million images were drawn manually by professional annotators, ensuring high [precision](https://www.ultralytics.com/glossary/precision).
- Annotations extend beyond boxes: 3.3 million visual-relationship annotations covering 1,466 triplets, segmentation masks for 2.8 million objects across 350 classes (added in V5), 675k localized narratives (V6), and 66.4 million point-level labels on 1.4 million images spanning 5,827 classes (V7).
- 61.4 million image-level labels across 20,638 classes make the dataset suitable for [image classification](https://www.ultralytics.com/glossary/image-classification) and multimodal research alongside detection and [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation).

## Dataset Structure

The Ultralytics `open-images-v7.yaml` configuration downloads the detection subset of Open Images V7:

| Split      | Images    | Description                                     |
| ---------- | --------- | ----------------------------------------------- |
| Train      | 1,743,042 | Box-annotated images for model training         |
| Validation | 41,620    | Held-out images for evaluation and benchmarking |

The configuration defines 601 object classes, indexed 0–600 from `Accordion` to `Zucchini`. Google's documentation rounds this figure down to 600, but both the official boxable class list and the Ultralytics YAML contain 601 entries. The `test:` key in the configuration is left empty.

## Applications

Open Images V7 supports training and evaluating models across a range of computer vision tasks:

- **Large-vocabulary object detection**: With 601 classes, detectors trained on Open Images V7 recognize far more categories than [COCO](coco.md)-trained ones.
- **Visual relationship detection**: 3.3 million relationship annotations support models that understand interactions between objects.
- **Instance segmentation**: Masks for 2.8 million objects enable pixel-level scene analysis.
- **Multimodal learning**: Localized narratives that combine voice, text, and mouse traces pair visual data with rich descriptions.
- **Zero-shot evaluation**: The extensive class coverage helps assess how models handle objects not seen during training.

To label your own images, train, and manage large-scale datasets in your browser, run the full workflow with [Ultralytics Platform](https://platform.ultralytics.com/).

## Dataset YAML

The `open-images-v7.yaml` file defines the dataset configuration — the dataset paths, class names, and other metadata. It is maintained in the Ultralytics repository at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/open-images-v7.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/open-images-v7.yaml).

!!! example "ultralytics/cfg/datasets/open-images-v7.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/open-images-v7.yaml"
    ```

## Usage

!!! warning "561 GB download"

    Open Images V7 downloads automatically on first use and requires about 561 GB of free disk space for its 1,743,042 training and 41,620 validation images. The download script installs the `fiftyone` package to fetch the images and converts the annotations to YOLO format, which can take a long time depending on your connection and hardware.

To train a YOLO26n model on the Open Images V7 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLO26n model
        model = YOLO("yolo26n.pt")

        # Train the model on the Open Images V7 dataset
        results = model.train(data="open-images-v7.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Train a COCO-pretrained YOLO26n model on the Open Images V7 dataset
        yolo detect train data=open-images-v7.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

The sample below shows the bounding-box, relationship, and mask annotations that Open Images V7 layers on a single image:

![Open Images V7 dataset sample with bounding box annotations](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/oidv7-all-in-one-example-ab.avif)

## Citations and Acknowledgments

If you use the Open Images V7 dataset in your research or development work, please cite the following paper:

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

We would like to acknowledge the Google AI team for creating and maintaining the Open Images V7 dataset. For more information about the dataset, visit the [official Open Images V7 website](https://storage.googleapis.com/openimages/web/index.html).

## FAQ

### What is the Open Images V7 dataset used for?

The [Open Images V7 dataset](https://storage.googleapis.com/openimages/web/index.html) is used to train and evaluate object detection models in [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and computer vision. The Ultralytics configuration provides 1,743,042 training and 41,620 validation images across 601 object classes, while the upstream release also carries segmentation masks, visual relationships, localized narratives, and point-level labels for broader research.

### How many images and classes are in the Open Images V7 dataset?

The Ultralytics `open-images-v7.yaml` configuration covers 601 object classes with 1,743,042 training images and 41,620 validation images, and no test split. The full upstream release spans about 9 million images; Google's documentation rounds the boxable class count down to 600, but the official class list contains 601 entries.

### How big is the Open Images V7 dataset download?

Open Images V7 requires about 561 GB of free disk space and downloads automatically the first time you train with `data="open-images-v7.yaml"`. The download script installs the `fiftyone` package and converts the annotations to YOLO format. If you need something smaller, browse the [detection datasets overview](index.md).

### How do I train a YOLO26 model on the Open Images V7 dataset?

To train a YOLO26 model on the Open Images V7 dataset, you can use both Python and CLI commands. Here's an example of training the YOLO26n model for 100 epochs with an image size of 640:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLO26n model
        model = YOLO("yolo26n.pt")

        # Train the model on the Open Images V7 dataset
        results = model.train(data="open-images-v7.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Train a COCO-pretrained YOLO26n model on the Open Images V7 dataset
        yolo detect train data=open-images-v7.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

For more details on arguments and settings, refer to the [Training](../../modes/train.md) page.

### Can I use the Open Images V7 pretrained models without downloading the dataset?

Yes. The five [Open Images V7 pretrained models](#open-images-v7-pretrained-models) (`yolov8n-oiv7.pt` through `yolov8x-oiv7.pt`, ranging from 18.4 to 36.3 mAP) detect all 601 classes out of the box, so you can run [prediction](../../modes/predict.md) on your own images with no 561 GB download.

### How does Open Images V7 compare to the COCO dataset?

Open Images V7 is larger and broader: 601 object classes and 1,743,042 training images versus 80 classes and 118,287 training images in [COCO](coco.md). COCO remains the standard benchmark for comparing detectors, while Open Images V7 is better suited to large-vocabulary detection and pretraining models that need wide category coverage.
