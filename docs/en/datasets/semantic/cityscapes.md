---
comments: true
description: Explore the Cityscapes semantic segmentation dataset with Ultralytics YOLO. Learn about its 19 urban classes, dataset structure, YAML configuration, and training examples.
keywords: Cityscapes dataset, semantic segmentation, Ultralytics YOLO, YOLO26, autonomous driving, urban scene understanding, computer vision, deep learning
---

# Cityscapes Dataset

The [Cityscapes](https://www.cityscapes-dataset.com/) dataset is a large-scale [semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation) benchmark focused on urban street scenes captured across 50 European cities. It provides high-quality pixel-level annotations and is one of the most widely used datasets for autonomous driving research and urban scene understanding with [Ultralytics YOLO](https://docs.ultralytics.com/models) models.

## Key Features

- Cityscapes fine annotations include 2,975 training images, 500 validation images, and 1,525 test images.
- The dataset covers 19 evaluation classes spanning road, vehicle, human, construction, object, nature, and sky categories.
- Cityscapes provides standardized evaluation metrics like [mean Intersection over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (mIoU) for semantic segmentation, enabling effective comparison of model performance.

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

The semantic masks are single-channel PNG files. The original Cityscapes label IDs are mapped to the standard 19 train IDs via the `label_mapping` section, and ignored or void labels are mapped to `255` so they are excluded from training and evaluation. Download the official `leftImg8bit` and `gtFine` archives from the Cityscapes website and extract them into the dataset root; the preparation block in `cityscapes.yaml` then organizes images and masks into this layout.

## Applications

Cityscapes is widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in semantic segmentation, particularly for [autonomous driving](https://www.ultralytics.com/glossary/autonomous-vehicles), advanced driver-assistance systems (ADAS), and urban robotics.

Its high-resolution images and detailed annotations also make it valuable for research on real-time scene parsing, lane and obstacle understanding, and any task that requires dense pixel-level understanding of complex urban environments.

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

## Citations and Acknowledgments

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

The [Cityscapes](https://www.cityscapes-dataset.com/) dataset is a large-scale [semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation) benchmark focused on urban street scenes captured across 50 European cities. It contains 5,000 finely annotated images across 19 evaluation classes, making it a foundational resource for autonomous driving and urban scene understanding research. Its high-resolution images, dense annotations, and standardized mean Intersection over Union (mIoU) metric make it ideal for benchmarking dense prediction models.

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

After preparation, the dataset is organized into `images/{train,val,test}/` and `masks/{train,val,test}/` directories, with each image paired with a single-channel PNG mask. The Ultralytics YAML file pairs each image with its mask via the `masks_dir: masks` field, and uses `label_mapping` to convert original Cityscapes label IDs into the standard 19 contiguous train IDs, mapping ignored and void labels to `255`.

### Do I need to download Cityscapes manually?

Yes. Cityscapes requires accepting the dataset terms on the official website. Download and extract `leftImg8bit` and `gtFine` into the `cityscapes` dataset root before using the preparation block in `cityscapes.yaml` to create the expected `images/` and `masks/` layout.

### Why does Cityscapes use `label_mapping`?

Cityscapes source masks store original label IDs that differ from the 19 train IDs used for evaluation. The `label_mapping` section converts valid labels to contiguous class IDs `0`–`18`, and assigns `255` to ignored and void labels so they are excluded from the loss and metrics during training and validation.
