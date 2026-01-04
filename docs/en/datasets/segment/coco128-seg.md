---
comments: true
description: Discover the COCO128-Seg dataset by Ultralytics, a compact yet diverse segmentation dataset ideal for testing and training YOLO11 models.
keywords: COCO128-Seg, Ultralytics, segmentation dataset, YOLO11, COCO 2017, model training, computer vision, dataset configuration
---

# COCO128-Seg Dataset

## Introduction

[Ultralytics](https://www.ultralytics.com/) COCO128-Seg is a small but versatile [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) dataset composed of the first 128 images of the COCO train 2017 set. This dataset is ideal for testing and debugging segmentation models, or for experimenting with new detection approaches. With 128 images, it is small enough to be easily manageable, yet diverse enough to test training pipelines for errors and act as a sanity check before training larger datasets.

## Dataset Structure

- **Images**: 128 total. The default YAML reuses the same directory for train and val so you can quickly iterate, but you can duplicate or customize the split if desired.
- **Classes**: Same 80 object categories as COCO.
- **Labels**: YOLO-format polygons saved beside each image inside `labels/{train,val}`.

This dataset is intended for use with Ultralytics [HUB](https://hub.ultralytics.com/) and [YOLO11](https://github.com/ultralytics/ultralytics).

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the COCO128-Seg dataset, the `coco128-seg.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128-seg.yaml).

!!! example "ultralytics/cfg/datasets/coco128-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco128-seg.yaml"
    ```

## Usage

To train a YOLO11n-seg model on the COCO128-Seg dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco128-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo segment train data=coco128-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

Here are some examples of images from the COCO128-Seg dataset, along with their corresponding annotations:

<img src="https://github.com/ultralytics/docs/releases/download/0/mosaiced-training-batch-2.avif" alt="Dataset sample image" width="800">

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the COCO128-Seg dataset and the benefits of using mosaicing during the training process.

## Citations and Acknowledgments

If you use the COCO dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{lin2015microsoft,
              title={Microsoft COCO: Common Objects in Context},
              author={Tsung-Yi Lin and Michael Maire and Serge Belongie and Lubomir Bourdev and Ross Girshick and James Hays and Pietro Perona and Deva Ramanan and C. Lawrence Zitnick and Piotr Dollár},
              year={2015},
              eprint={1405.0312},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

We would like to acknowledge the COCO Consortium for creating and maintaining this valuable resource for the [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) community. For more information about the COCO dataset and its creators, visit the [COCO dataset website](https://cocodataset.org/#home).

## FAQ

### What is the COCO128-Seg dataset, and how is it used in Ultralytics YOLO11?

The **COCO128-Seg dataset** is a compact instance segmentation dataset by Ultralytics, consisting of the first 128 images from the COCO train 2017 set. This dataset is tailored for testing and debugging segmentation models or experimenting with new detection methods. It is particularly useful with Ultralytics [YOLO11](https://github.com/ultralytics/ultralytics) and [HUB](https://hub.ultralytics.com/) for rapid iteration and pipeline error-checking before scaling to larger datasets. For detailed usage, refer to the model [Training](../../modes/train.md) page.

### How can I train a YOLO11n-seg model using the COCO128-Seg dataset?

To train a **YOLO11n-seg** model on the COCO128-Seg dataset for 100 epochs with an image size of 640, you can use Python or CLI commands. Here's a quick example:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-seg.pt")  # Load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco128-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo segment train data=coco128-seg.yaml model=yolo11n-seg.pt epochs=100 imgsz=640
        ```

For a thorough explanation of available arguments and configuration options, you can check the [Training](../../modes/train.md) documentation.

### Why is the COCO128-Seg dataset important for model development and debugging?

The **COCO128-Seg dataset** offers a balanced combination of manageability and diversity with 128 images, making it perfect for quickly testing and debugging segmentation models or experimenting with new detection techniques. Its moderate size allows for fast training iterations while providing enough diversity to validate training pipelines before scaling to larger datasets. Learn more about supported dataset formats in the [Ultralytics segmentation dataset guide](https://docs.ultralytics.com/datasets/segment/).

### Where can I find the YAML configuration file for the COCO128-Seg dataset?

The YAML configuration file for the **COCO128-Seg dataset** is available in the Ultralytics repository. You can access the file directly at <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128-seg.yaml>. The YAML file includes essential information about dataset paths, classes, and configuration settings required for model training and validation.

### What are some benefits of using mosaicing during training with the COCO128-Seg dataset?

Using **mosaicing** during training helps increase the diversity and variety of objects and scenes in each training batch. This technique combines multiple images into a single composite image, enhancing the model's ability to generalize to different object sizes, aspect ratios, and contexts within the scene. Mosaicing is beneficial for improving a model's robustness and [accuracy](https://www.ultralytics.com/glossary/accuracy), especially when working with moderately-sized datasets like COCO128-Seg. For an example of mosaiced images, see the [Sample Images and Annotations](#sample-images-and-annotations) section.
