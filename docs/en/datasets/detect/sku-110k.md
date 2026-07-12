---
title: SKU-110K Detection Dataset
comments: true
description: SKU-110K is a single-class retail-shelf object detection dataset of 11,743 densely packed images (8,219 train / 588 val / 2,936 test) for training YOLO models.
keywords: SKU-110K, dataset, object detection, retail shelf images, densely packed objects, single class, deep learning, computer vision, YOLO26, Ultralytics
---

# SKU-110K Dataset

The SKU-110K dataset is a single-class [object detection](../../tasks/detect.md) dataset of 11,743 densely packed retail-shelf images, split into 8,219 training, 588 validation, and 2,936 test images. Every product is annotated with one bounding box under a single class, `object` — the name refers to the more than 110,000 unique store-keeping units (SKUs) pictured across the scenes, not to 110,000 detection classes. Created by Eran Goldman et al. for the CVPR 2019 paper [Precise Detection in Densely Packed Scenes](https://github.com/eg4000/SKU110K_CVPR19), it carries over 1.7 million annotated products — an average of roughly 147 per image — making it a demanding benchmark for [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models in crowded retail environments.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/_gRqR-miFPE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train YOLOv10 on SKU-110k Dataset using Ultralytics | Retail Dataset
</p>

![SKU-110K dataset densely packed retail shelf detection](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/densely-packed-retail-shelf.avif)

## Key Features

- **Single-class detection**: Every product is labeled with one bounding box under a single class, `object` (`names: {0: object}`) — the annotations carry no per-SKU category labels.
- **Extreme object density**: Store-shelf images from around the world average about 147 tightly packed products each, with objects that often look similar or even identical positioned in close proximity.
- **Large scale**: More than 110,000 unique SKUs and over 1.7 million annotated bounding boxes across 11,743 images challenge state-of-the-art object detectors.

## Dataset Structure

The SKU-110K dataset is split into three subsets, all sharing the single `object` class:

| Split      | Images | Description                                                          |
| ---------- | ------ | -------------------------------------------------------------------- |
| Train      | 8,219  | Images and annotations for model training                            |
| Validation | 588    | Held-out images for [evaluation](../../modes/val.md) during training |
| Test       | 2,936  | Images for final evaluation of the trained model                     |

## Applications

The SKU-110K dataset is widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in object detection tasks, especially in densely packed scenes such as retail shelf displays. Its applications include:

- Retail inventory management and automation
- Product recognition in e-commerce platforms
- Planogram compliance verification
- Self-checkout systems in stores
- Robotic picking and sorting in warehouses

To annotate your own shelf images, train, and manage retail-detection datasets in your browser, run the full workflow with [Ultralytics Platform](https://platform.ultralytics.com/).

## Dataset YAML

The `SKU-110K.yaml` file defines the dataset configuration — the dataset paths, class names, and other metadata. It is maintained in the Ultralytics repository at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/SKU-110K.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/SKU-110K.yaml).

!!! example "ultralytics/cfg/datasets/SKU-110K.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/SKU-110K.yaml"
    ```

## Usage

!!! warning "13.6 GB download"

    SKU-110K downloads automatically the first time you train and requires about 13.6 GB of free disk space for its 11,743 images. The download script also fetches the original annotations and converts them to YOLO format, which can take a few minutes.

To train a YOLO26n model on the SKU-110K dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="SKU-110K.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=SKU-110K.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

SKU-110K images capture densely packed products on real store shelves, where dozens of near-identical items sit side by side. Here is an example image with its annotations:

![SKU-110K retail product detection on store shelves](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/densely-packed-retail-shelf-1.avif)

- **Densely packed retail shelf image**: This image demonstrates an example of densely packed objects in a retail shelf setting. Objects are annotated with bounding boxes under the single `object` class.

The dense arrangement of products makes SKU-110K particularly valuable for developing robust retail-focused computer vision solutions, as the high object count per image pushes detectors well beyond typical benchmarks.

## Citations and Acknowledgments

If you use the SKU-110K dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{goldman2019dense,
          author    = {Eran Goldman and Roei Herzig and Aviv Eisenschtat and Jacob Goldberger and Tal Hassner},
          title     = {Precise Detection in Densely Packed Scenes},
          booktitle = {Proc. Conf. Comput. Vision Pattern Recognition (CVPR)},
          year      = {2019}
        }
        ```

We would like to acknowledge Eran Goldman et al. for creating and maintaining the SKU-110K dataset as a valuable resource for the computer vision research community. For more information about the SKU-110K dataset and its creators, visit the [SKU-110K dataset GitHub repository](https://github.com/eg4000/SKU110K_CVPR19).

## FAQ

### What is the SKU-110K dataset used for?

The SKU-110K dataset is a single-class object detection dataset of 11,743 densely packed retail-shelf images, created by Eran Goldman et al. for their CVPR 2019 paper. Every product is labeled with one `object` bounding box, and the imagery spans more than 110,000 unique store-keeping units (SKUs), making it a strong benchmark for detecting objects in crowded scenes and for building retail [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) systems.

### Does the SKU-110K dataset have 110,000 classes?

No. SKU-110K is single-class: every product is annotated with one bounding box under the class `object` (`names: {0: object}`). The "110K" in the name refers to the number of unique store-keeping units (SKUs) pictured across the images, not to the number of detection classes.

### How many images and classes are in the SKU-110K dataset?

The SKU-110K dataset contains 11,743 images — 8,219 for training, 588 for validation, and 2,936 for testing — and a single detection class, `object`. See the [Dataset Structure](#dataset-structure) section and the [`SKU-110K.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/SKU-110K.yaml) configuration for details.

### How big is the SKU-110K dataset download?

SKU-110K is about 13.6 GB and downloads automatically the first time you train with `data="SKU-110K.yaml"` — no manual download is required. To browse smaller options, see the [detection datasets overview](index.md).

### How do I train a YOLO26 model using the SKU-110K dataset?

Training a YOLO26 model on the SKU-110K dataset is straightforward. Here's an example to train a YOLO26n model for 100 epochs with an image size of 640:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="SKU-110K.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=SKU-110K.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page and [model training tips](../../guides/model-training-tips.md).
