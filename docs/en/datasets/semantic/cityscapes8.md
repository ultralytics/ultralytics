---
title: Cityscapes8 Semantic Segmentation Dataset
comments: true
description: Explore the Ultralytics Cityscapes8 dataset, a compact set of 8 urban-scene images for testing semantic segmentation models and training pipelines.
keywords: Cityscapes8, Ultralytics, dataset, semantic segmentation, YOLO26, semantic, training, validation, urban scenes, computer vision
---

# Cityscapes8 Dataset

## Introduction

The [Ultralytics](https://www.ultralytics.com/) Cityscapes8 dataset is a compact [semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation) dataset with 8 images sampled from the [Cityscapes](cityscapes.md) dataset: 4 for training and 4 for validation. It is designed for rapid testing, debugging, and experimentation with [YOLO](../../models/yolo26.md) semantic segmentation models and training pipelines. Its urban-scene content provides a useful pipeline check before scaling to the full Cityscapes dataset.

Cityscapes8 uses the same 19 evaluation classes and the same `label_mapping` behavior as the full Cityscapes dataset, and is fully compatible with [YOLO26](../../models/yolo26.md) semantic segmentation workflows.

!!! note

    Cityscapes8 is for pipeline testing only, not benchmarking — its 8 images are too few for a meaningful mIoU comparison. Use the full [Cityscapes](cityscapes.md) validation set for representative results.

## Dataset Structure

Cityscapes8 mirrors the full dataset's layout, without a `test` split:

```text
cityscapes8/
├── images/
│   ├── train/  # 4 images
│   └── val/    # 4 images
└── masks/
    ├── train/  # 4 single-channel PNG masks
    └── val/    # 4 single-channel PNG masks
```

Masks are paired with images via the `masks_dir: masks` field, and `label_mapping` converts source Cityscapes label IDs into the 19 contiguous train IDs described in the [full Cityscapes dataset structure](cityscapes.md#dataset-structure).

Explore [Cityscapes8 on Ultralytics Platform](https://platform.ultralytics.com/ultralytics/datasets/cityscapes8) to browse every image with its segmentation masks and clone it to train in the cloud.

## Dataset YAML

The Cityscapes8 dataset configuration is defined in a dataset YAML file, which specifies dataset paths, class names, and other essential metadata. You can review the official `cityscapes8.yaml` file in the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/cityscapes8.yaml). The YAML includes a download URL for the small packaged subset.

!!! example "ultralytics/cfg/datasets/cityscapes8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/cityscapes8.yaml"
    ```

## Usage

To train a YOLO26n-sem model on the Cityscapes8 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 1024, use the following examples. For a full list of training options, see the [YOLO Training documentation](../../modes/train.md).

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO26n-sem model
        model = YOLO("yolo26n-sem.pt")

        # Train the model on Cityscapes8
        results = model.train(data="cityscapes8.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        # Train YOLO26n-sem on Cityscapes8 using the command line
        yolo semantic train data=cityscapes8.yaml model=yolo26n-sem.pt epochs=100 imgsz=1024
        ```

## Citations, License and Acknowledgments

Cityscapes8 is sampled from Cityscapes, which is released under a [custom non-commercial license](https://www.cityscapes-dataset.com/license/) — see the full [Cityscapes dataset page](cityscapes.md#citations-license-and-acknowledgments) for details.

If you use the Cityscapes dataset in your research or development, please cite the following paper:

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

Special thanks to the [Cityscapes team](https://www.cityscapes-dataset.com/) for their ongoing contributions to the autonomous driving and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) communities.

## FAQ

### What is the Ultralytics Cityscapes8 dataset used for?

The Ultralytics Cityscapes8 dataset is designed for rapid testing and debugging of [semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation) models. With only 8 images (4 for training, 4 for validation), it is ideal for verifying [YOLO](../../models/yolo26.md) semantic segmentation pipelines, including mask loading, augmentations, validation, and export paths, before scaling to the full [Cityscapes](cityscapes.md) dataset. Explore the [Cityscapes8 YAML configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/cityscapes8.yaml) for more details.

### How does Cityscapes8 differ from the full Cityscapes dataset?

Cityscapes8 samples 8 images (4 train, 4 val) from Cityscapes' 2,975-training/500-validation split, using the same 19 classes and `label_mapping`, so a pipeline that runs on Cityscapes8 runs unmodified on the full dataset — just point `data=` at `cityscapes.yaml` instead of `cityscapes8.yaml`. Unlike the full dataset, Cityscapes8 has no manual-download step and no `test` split.

### How do I train a YOLO26 model using the Cityscapes8 dataset?

You can train a YOLO26 semantic segmentation model on Cityscapes8 using either Python or the CLI:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO26n-sem model
        model = YOLO("yolo26n-sem.pt")

        # Train the model on Cityscapes8
        results = model.train(data="cityscapes8.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        # Train YOLO26n-sem on Cityscapes8 using the command line
        yolo semantic train data=cityscapes8.yaml model=yolo26n-sem.pt epochs=100 imgsz=1024
        ```

For additional training options, refer to the [YOLO Training documentation](../../modes/train.md).

### Should I use Cityscapes8 for benchmarking?

No. Cityscapes8 is too small for meaningful model comparison and is intended for training and evaluation pipeline checks. Use the full [Cityscapes](cityscapes.md) validation set when you need representative benchmark results for semantic segmentation.
