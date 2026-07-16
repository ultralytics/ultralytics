---
title: PASCAL VOC Detection Dataset
comments: true
creator:
    name: PASCAL Visual Object Classes
    url: http://host.robots.ox.ac.uk/pascal/VOC/
license:
    name: Other
    url: http://host.robots.ox.ac.uk/pascal/VOC/
description: Train YOLO26 on the PASCAL VOC detection dataset - 16,551 training and 4,952 validation images across 20 object classes with automatic download.
keywords: PASCAL VOC, VOC dataset, VOC2007, VOC2012, object detection dataset, YOLO26, download PASCAL VOC, computer vision benchmark
---

# PASCAL VOC Dataset

The [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) (Visual Object Classes) dataset is a classic [object detection](../../tasks/detect.md) benchmark with 20 everyday object classes. The Ultralytics `VOC.yaml` configuration combines the VOC2007 and VOC2012 trainval splits into a 16,551-image training set, validates on the 4,952 publicly annotated VOC2007 test images, and downloads everything automatically (2.8 GB) on first use.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yrHzL8RyY6g"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train Ultralytics YOLO on the Pascal VOC Dataset | Object Detection | Computer Vision 🚀
</p>

The PASCAL VOC challenges ran from 2005 to 2012 and shaped how [object detection](https://www.ultralytics.com/glossary/object-detection) models are evaluated: the benchmark spans image classification, detection, and segmentation tasks, and popularized [mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP) as the standard detection metric. The Ultralytics `VOC.yaml` configuration uses the detection annotations, converting the original XML bounding boxes to YOLO format during download.

## Key Features

- **20 everyday object classes**: person; six animals (bird, cat, cow, dog, horse, sheep); seven vehicles (aeroplane, bicycle, boat, bus, car, motorbike, train); and six indoor objects (bottle, chair, diningtable, pottedplant, sofa, tvmonitor).
- **Two challenge generations combined**: training merges VOC2007 trainval (5,011 images) with VOC2012 trainval (11,540 images).
- **Standardized evaluation**: decades of published VOC baselines make it a convenient reference point for comparing detection models.
- **YOLO-ready**: the download script fetches the archives and converts the annotations automatically — no manual preparation.

## Dataset Structure

The Ultralytics `VOC.yaml` configuration defines the following splits:

| Split      | Images | Source                                                                       |
| ---------- | ------ | ---------------------------------------------------------------------------- |
| Train      | 16,551 | VOC2007 trainval (5,011) + VOC2012 trainval (11,540)                         |
| Validation | 4,952  | VOC2007 test, used for [evaluation](../../modes/val.md) during training      |
| Test       | 4,952  | The same VOC2007 test images — the config defines no separate held-out split |

VOC2007 test annotations were released publicly after that year's challenge, which is what allows this split to serve as a labeled validation set. VOC2012 test annotations remain withheld — results on them can only be scored through the official PASCAL evaluation server — so they are not part of this configuration.

!!! note "Difficult objects excluded"

    The automatic converter skips objects flagged `difficult` in the original VOC XML annotations, so per-class instance counts differ slightly from official VOC statistics.

Explore [VOC on Ultralytics Platform](https://platform.ultralytics.com/ultralytics/datasets/voc) to browse the images with their annotation overlays, view the class distribution and bounding-box heatmaps in the **Charts** tab, and clone it to train your own model in the cloud.

## Applications

PASCAL VOC was the primary benchmark for object detection research in the years before the larger [COCO dataset](coco.md): detectors such as [Faster R-CNN](https://arxiv.org/abs/1506.01497) and [SSD](https://arxiv.org/abs/1512.02325) reported their original results on it, and [Ultralytics YOLO](../../models/yolo26.md) models train on it out of the box. Today it remains popular for:

- Benchmarking new detection architectures against a long history of published baselines
- Fast experiments and coursework — at 16,551 training images it trains far quicker than COCO
- [Transfer learning](https://www.ultralytics.com/glossary/transfer-learning) studies on a compact, well-understood set of everyday classes

## Dataset YAML

The `VOC.yaml` file defines the dataset configuration — the dataset paths, the 20 class names, and the automatic download-and-convert script. It is maintained in the Ultralytics repository at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VOC.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VOC.yaml).

!!! example "ultralytics/cfg/datasets/VOC.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/VOC.yaml"
    ```

## Usage

!!! note "2.8 GB download"

    VOC downloads automatically the first time you train — three archives totaling 2.8 GB — and needs roughly 6 GB of free disk space during extraction and conversion.

To train a YOLO26n model on the VOC dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model - dataset will auto-download on first run
        results = model.train(data="VOC.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        # Dataset will auto-download and convert on first run
        yolo detect train data=VOC.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

The image below shows a mosaiced training batch from the VOC dataset. Mosaicing combines multiple images into a single training sample, increasing the variety of objects, scales, and scene contexts the model sees in each batch — see the [YOLO data augmentation guide](../../guides/yolo-data-augmentation.md) for details.

![Pascal VOC dataset mosaic training batch](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-voc-dataset-sample.avif)

## Citations and Acknowledgments

If you use the VOC dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{everingham2010pascal,
          author={Everingham, Mark and Van Gool, Luc and Williams, Christopher K. I. and Winn, John and Zisserman, Andrew},
          journal={International Journal of Computer Vision},
          title={The Pascal Visual Object Classes (VOC) Challenge},
          year={2010},
          volume={88},
          number={2},
          pages={303-338},
          doi={10.1007/s11263-009-0275-4}}
        ```

We would like to acknowledge the PASCAL VOC Consortium for creating and maintaining this valuable resource for the [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) community. For more information about the VOC dataset and its creators, visit the [PASCAL VOC dataset website](http://host.robots.ox.ac.uk/pascal/VOC/).

## FAQ

### What is the PASCAL VOC dataset used for?

PASCAL VOC is used to train and benchmark object detection models on 20 everyday object classes such as person, car, dog, and chair. Because it is compact, fully labeled, and backed by years of published baselines, it is a common choice for validating new architectures, running coursework experiments, and quick [transfer learning](https://www.ultralytics.com/glossary/transfer-learning) studies.

### How many images are in the PASCAL VOC dataset?

The Ultralytics VOC configuration contains 21,503 images: 16,551 for training (VOC2007 trainval + VOC2012 trainval) and 4,952 for validation (the VOC2007 test set). All splits share the same 20 classes. See [Dataset Structure](#dataset-structure) for the full breakdown.

### How do I download the PASCAL VOC dataset?

VOC downloads automatically the first time you train with `data="VOC.yaml"` — no manual steps are required. The script fetches three archives (2.8 GB) from Ultralytics GitHub release assets and converts the XML annotations to YOLO format.

### How do I train a YOLO26 model on the VOC dataset?

Train a YOLO26n model on VOC for 100 epochs at an image size of 640:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="VOC.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=VOC.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

For detailed configurations, see the [Training](../../modes/train.md) page and [model training tips](../../guides/model-training-tips.md).

### What is the difference between VOC2007 and VOC2012?

Both challenges share the same 20 classes but contribute different images. VOC2007 provides 5,011 trainval images plus a 4,952-image test set whose annotations are public; VOC2012 provides 11,540 trainval images, while its test annotations are withheld and scored only by the official evaluation server. The Ultralytics `VOC.yaml` merges both trainval sets for training and validates on VOC2007 test.

### How does PASCAL VOC compare to the COCO dataset?

VOC is smaller and simpler: 20 classes and 21,503 images versus COCO's 80 classes and 330K images. VOC results are traditionally reported as mAP at 0.5 IoU, while COCO averages mAP over IoU thresholds from 0.5 to 0.95. VOC trains much faster and suits quick experiments; the [COCO dataset](coco.md) is the standard for production-scale benchmarking.

### Can I train segmentation models with VOC.yaml?

No — `VOC.yaml` is a detection-only configuration: its converter extracts bounding boxes from the VOC XML annotations, and the segmentation masks included in the original benchmark are not converted. To train an [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) model, use a dataset with polygon labels such as [COCO-Seg](../segment/coco.md) with a `yolo26n-seg.pt` model.
