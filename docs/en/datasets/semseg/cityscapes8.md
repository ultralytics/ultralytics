---
comments: true
description: Explore the Ultralytics Cityscapes8 dataset, a compact set of 8 urban-scene images perfect for testing semantic segmentation models and training pipelines.
keywords: Cityscapes8, Ultralytics, dataset, semantic segmentation, YOLO26, semseg, training, validation, urban scenes, computer vision
---

# Cityscapes8 Dataset

## Introduction

The [Ultralytics](https://www.ultralytics.com/) Cityscapes8 dataset is a compact yet powerful [semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation) dataset, consisting of 8 images sampled from the [Cityscapes](cityscapes.md) dataset—4 for training and 4 for validation. This dataset is specifically designed for rapid testing, debugging, and experimentation with [YOLO](https://docs.ultralytics.com/models/yolo26/) semseg models and training pipelines. Its small size makes it highly manageable, while its urban-scene content ensures it serves as an effective sanity check before scaling up to the full Cityscapes dataset.

Cityscapes8 uses the same 19 evaluation classes and the same `label_mapping` behavior as the full Cityscapes dataset, and is fully compatible with [YOLO26](../../models/yolo26.md) for seamless integration into your computer vision workflows.

## Dataset YAML

The Cityscapes8 dataset configuration is defined in a YAML (Yet Another Markup Language) file, which specifies dataset paths, class names, and other essential metadata. You can review the official `cityscapes8.yaml` file in the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/cityscapes8.yaml). The YAML includes a download URL for the small packaged subset.

!!! example "ultralytics/cfg/datasets/cityscapes8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/cityscapes8.yaml"
    ```

## Usage

To train a YOLO26n-semseg model on the Cityscapes8 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 1024, use the following examples. For a full list of training options, see the [YOLO Training documentation](../../modes/train.md).

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO26n-semseg model
        model = YOLO("yolo26n-semseg.pt")

        # Train the model on Cityscapes8
        results = model.train(data="cityscapes8.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        # Train YOLO26n-semseg on Cityscapes8 using the command line
        yolo semseg train data=cityscapes8.yaml model=yolo26n-semseg.pt epochs=100 imgsz=1024
        ```

## Citations and Acknowledgments

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

### What Is the Ultralytics Cityscapes8 Dataset Used For?

The Ultralytics Cityscapes8 dataset is designed for rapid testing and debugging of [semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation) models. With only 8 images (4 for training, 4 for validation), it is ideal for verifying your [YOLO](https://docs.ultralytics.com/models/yolo26/) semseg training pipelines—mask loading, augmentations, validation, and export paths—before scaling to the full [Cityscapes](cityscapes.md) dataset. Explore the [Cityscapes8 YAML configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/cityscapes8.yaml) for more details.

### How Do I Train a YOLO26 Model Using the Cityscapes8 Dataset?

You can train a YOLO26 semseg model on Cityscapes8 using either Python or the CLI:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO26n-semseg model
        model = YOLO("yolo26n-semseg.pt")

        # Train the model on Cityscapes8
        results = model.train(data="cityscapes8.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        yolo semseg train data=cityscapes8.yaml model=yolo26n-semseg.pt epochs=100 imgsz=1024
        ```

For additional training options, refer to the [YOLO Training documentation](../../modes/train.md).

### Should I Use Cityscapes8 for Benchmarking?

No. Cityscapes8 is too small for meaningful model comparison and is intended only as a quick sanity check for your training and evaluation pipeline. Use the full [Cityscapes](cityscapes.md) validation set when you need representative benchmark results for semantic segmentation.
