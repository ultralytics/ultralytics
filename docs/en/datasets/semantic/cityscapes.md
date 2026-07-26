---
title: Cityscapes Semantic Segmentation Dataset
comments: true
creator:
    name: Cityscapes Dataset
    url: https://www.cityscapes-dataset.com/
license:
    name: Research-Only
    url: https://www.cityscapes-dataset.com/license/
description: Train Ultralytics YOLO on Cityscapes — 2,975 training and 500 validation images across 19 urban classes for semantic segmentation of street scenes.
keywords: Cityscapes dataset, semantic segmentation, Ultralytics YOLO, YOLO26, autonomous driving, urban scene understanding, computer vision, deep learning
---

# Cityscapes Dataset

The [Cityscapes](https://www.cityscapes-dataset.com/) dataset is a large-scale [semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation) benchmark of urban street scenes captured across 50 European cities, with 2,975 finely annotated training images and 500 validation images across 19 classes. It is one of the most widely used datasets for autonomous driving research and urban scene understanding with [Ultralytics YOLO](../../models/index.md) models.

## Key Features

- Cityscapes fine annotations include 2,975 training images and 500 validation images across 19 classes; the archive also ships 1,525 test images, but their released masks only label the ego-vehicle and image border — real class annotations are withheld, and official test-set scores require submitting predictions to the [Cityscapes evaluation server](https://www.cityscapes-dataset.com/benchmarks/).
- The dataset covers 19 evaluation classes spanning flat, human, vehicle, construction, object, nature, and sky categories.
- Cityscapes provides standardized evaluation metrics like [mean Intersection over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (mIoU) for semantic segmentation, enabling effective comparison of model performance.
- Before committing to the ~11 GB manual download, sanity-check your training pipeline against the 8-image [Cityscapes8](cityscapes8.md) subset.

## Dataset Structure

The Ultralytics configuration expects the following layout after preparation:

```text
cityscapes/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── masks/
    ├── train/
    ├── val/
    └── test/
```

!!! warning "Manual Download Required"

    Cityscapes has no automatic archive download. Create an account on the [Cityscapes website](https://www.cityscapes-dataset.com/), then download the `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` archives (~11 GB combined) and extract both into the `cityscapes` dataset root. Ultralytics automatically reorganizes them into the `images/` and `masks/` layout above the first time you train.

The semantic masks are single-channel PNG files. The original Cityscapes label IDs are mapped to the standard 19 train IDs via the `label_mapping` section, and ignored or void labels are mapped to `255` so they are excluded from training and evaluation.

!!! note

    The publicly released `gtFine/test` masks only label the ego-vehicle and image border regions — all other classes are void. Compute mIoU on the `val` split for local evaluation; official test-set scores require submitting predictions to the [Cityscapes evaluation server](https://www.cityscapes-dataset.com/benchmarks/).

## Applications

Cityscapes is widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in semantic segmentation, particularly for [autonomous driving](https://www.ultralytics.com/glossary/autonomous-vehicles), advanced driver-assistance systems (ADAS), and urban robotics.

Its high-resolution images and detailed annotations also make it valuable for research on real-time scene parsing, lane and obstacle understanding, and any task that requires dense pixel-level understanding of complex urban environments. Pretrained YOLO26 semantic segmentation models reach up to 83.6 mIoU on the Cityscapes validation set — see the [semantic segmentation models](../../tasks/semantic.md) page for the full benchmark table. Cityscapes annotations are also available on [Ultralytics Platform](https://platform.ultralytics.com/ultralytics/datasets/cityscapes) for browsing and dataset management.

## Dataset YAML

A dataset YAML file defines the Cityscapes paths, classes, mask directory, and label mapping. The `cityscapes.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/cityscapes.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/cityscapes.yaml).

!!! example "ultralytics/cfg/datasets/cityscapes.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/cityscapes.yaml"
    ```

## Usage

To train a YOLO26n-sem model on the Cityscapes dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 1024, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-sem.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="cityscapes.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo semantic train data=cityscapes.yaml model=yolo26n-sem.pt epochs=100 imgsz=1024
        ```

## Citations, License and Acknowledgments

Cityscapes is released under a [custom non-commercial license](https://www.cityscapes-dataset.com/license/) — free for academic research and evaluation, but commercial use, licensing, or redistribution of the data requires separate permission from the Cityscapes team.

If you use the Cityscapes dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{Cordts2016Cityscapes,
          title={The Cityscapes Dataset for Semantic Urban Scene Understanding},
          author={Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
          booktitle={Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
          year={2016}
        }
        ```

We would like to acknowledge the Cityscapes team for creating and maintaining this valuable resource for the autonomous driving and computer vision communities. For more information about the Cityscapes dataset and its creators, visit the [Cityscapes dataset website](https://www.cityscapes-dataset.com/).

## FAQ

### What is the Cityscapes dataset and why is it important for computer vision?

The [Cityscapes](https://www.cityscapes-dataset.com/) dataset is a large-scale [semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation) benchmark of urban street scenes across 50 European cities, widely used as a standard reference for autonomous driving and ADAS research. Its 19 finely annotated evaluation classes, high-resolution imagery, and standardized mean Intersection over Union (mIoU) metric make it one of the most cited benchmarks for dense scene-understanding models.

### How can I train a YOLO model using the Cityscapes dataset?

To train a YOLO26n-sem model on the Cityscapes dataset for 100 epochs with an image size of 1024, you can use the following code snippets. For a detailed list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-sem.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="cityscapes.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo semantic train data=cityscapes.yaml model=yolo26n-sem.pt epochs=100 imgsz=1024
        ```

### How is the Cityscapes dataset structured?

After preparation, the dataset is organized into `images/{train,val,test}/` and `masks/{train,val,test}/` directories, with each image paired with a single-channel PNG mask. The Ultralytics YAML file pairs each image with its mask via the `masks_dir: masks` field, and uses `label_mapping` to convert original Cityscapes label IDs into the standard 19 contiguous train IDs, mapping ignored and void labels to `255`. The `test` split's masks only label ego-vehicle and border regions, so use `val` for local mIoU checks.

### Do I need to download Cityscapes manually?

Yes. Create an account on the [Cityscapes website](https://www.cityscapes-dataset.com/) and download the `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` archives (~11 GB combined). Extract both into the `cityscapes` dataset root — Ultralytics automatically reorganizes them into the expected `images/` and `masks/` layout the first time you train.

### Why does Cityscapes use `label_mapping`?

Cityscapes source masks store original label IDs that differ from the 19 train IDs used for evaluation. The `label_mapping` section converts valid labels to contiguous class IDs `0`–`18`, and assigns `255` to ignored and void labels so they are excluded from the loss and metrics during training and validation.

### Is the Cityscapes dataset free for commercial use?

No. Cityscapes is released under a [non-commercial license](https://www.cityscapes-dataset.com/license/) that permits academic research, teaching, and evaluation, but prohibits commercial use, licensing, or selling the dataset or derivative works. Contact the Cityscapes team directly for commercial licensing options.
