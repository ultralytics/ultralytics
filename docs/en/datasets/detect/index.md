---
comments: true
description: Learn about dataset formats compatible with Ultralytics YOLO for robust object detection. Explore supported datasets and learn how to convert formats.
keywords: Ultralytics, YOLO, object detection datasets, dataset formats, COCO, dataset conversion, training datasets
---

# Object Detection Datasets Overview

Training a robust and accurate [object detection](https://www.ultralytics.com/glossary/object-detection) model requires a comprehensive dataset. This guide introduces various formats of datasets that are compatible with the Ultralytics YOLO model and provides insights into their structure, usage, and how to convert between different formats.

## Supported Dataset Formats

### Ultralytics YOLO format

The Ultralytics YOLO format is a dataset configuration format that allows you to define the dataset root directory, the relative paths to training/validation/testing image directories or `*.txt` files containing image paths, and a dictionary of class names. Here is an example:

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco8 # dataset root dir
train: images/train # train images (relative to 'path') 4 images
val: images/val # val images (relative to 'path') 4 images
test: # test images (optional)

# Classes (80 COCO classes)
names:
    0: person
    1: bicycle
    2: car
    # ...
    77: teddy bear
    78: hair drier
    79: toothbrush
```

Labels for this format should be exported to YOLO format with one `*.txt` file per image. If there are no objects in an image, no `*.txt` file is required. The `*.txt` file should be formatted with one row per object in `class x_center y_center width height` format. Box coordinates must be in **normalized xywh** format (from 0 to 1). If your boxes are in pixels, you should divide `x_center` and `width` by image width, and `y_center` and `height` by image height. Class numbers should be zero-indexed (start with 0).

<p align="center"><img width="750" src="https://github.com/ultralytics/docs/releases/download/0/two-persons-tie.avif" alt="Example labelled image"></p>

The label file corresponding to the above image contains 2 persons (class `0`) and a tie (class `27`):

<p align="center"><img width="428" src="https://github.com/ultralytics/docs/releases/download/0/two-persons-tie-1.avif" alt="Example label file"></p>

When using the Ultralytics YOLO format, organize your training and validation images and labels as shown in the [COCO8 dataset](coco8.md) example below.

<p align="center"><img width="800" src="https://github.com/ultralytics/docs/releases/download/0/two-persons-tie-2.avif" alt="Example dataset directory structure"></p>

## Usage

Here's how you can use these formats to train your model:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## Supported Datasets

Here is a list of the supported datasets and a brief description for each:

- [Argoverse](argoverse.md): A dataset containing 3D tracking and motion forecasting data from urban environments with rich annotations.
- [COCO](coco.md): Common Objects in Context (COCO) is a large-scale object detection, segmentation, and captioning dataset with 80 object categories.
- [LVIS](lvis.md): A large-scale object detection, segmentation, and captioning dataset with 1203 object categories.
- [COCO8](coco8.md): A smaller subset of the first 4 images from COCO train and COCO val, suitable for quick tests.
- [COCO128](coco.md): A smaller subset of the first 128 images from COCO train and COCO val, suitable for tests.
- [Global Wheat 2020](globalwheat2020.md): A dataset containing images of wheat heads for the Global Wheat Challenge 2020.
- [Objects365](objects365.md): A high-quality, large-scale dataset for object detection with 365 object categories and over 600K annotated images.
- [OpenImagesV7](open-images-v7.md): A comprehensive dataset by Google with 1.7M train images and 42k validation images.
- [SKU-110K](sku-110k.md): A dataset featuring dense object detection in retail environments with over 11K images and 1.7 million bounding boxes.
- [VisDrone](visdrone.md): A dataset containing object detection and multi-object tracking data from drone-captured imagery with over 10K images and video sequences.
- [VOC](voc.md): The Pascal Visual Object Classes (VOC) dataset for object detection and segmentation with 20 object classes and over 11K images.
- [xView](xview.md): A dataset for object detection in overhead imagery with 60 object categories and over 1 million annotated objects.
- [Roboflow 100](roboflow-100.md): A diverse object detection benchmark with 100 datasets spanning seven imagery domains for comprehensive model evaluation.
- [Brain-tumor](brain-tumor.md): A dataset for detecting brain tumors includes MRI or CT scan images with details on tumor presence, location, and characteristics.
- [African-wildlife](african-wildlife.md): A dataset featuring images of African wildlife, including buffalo, elephant, rhino, and zebras.
- [Signature](signature.md): A dataset featuring images of various documents with annotated signatures, supporting document verification and fraud detection research.

### Adding your own dataset

If you have your own dataset and would like to use it for training detection models with Ultralytics YOLO format, ensure that it follows the format specified above under "Ultralytics YOLO format". Convert your annotations to the required format and specify the paths, number of classes, and class names in the YAML configuration file.

## Port or Convert Label Formats

### COCO Dataset Format to YOLO Format

You can easily convert labels from the popular COCO dataset format to the YOLO format using the following code snippet:

!!! example

    === "Python"

        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco(labels_dir="path/to/coco/annotations/")
        ```

This conversion tool can be used to convert the COCO dataset or any dataset in the COCO format to the Ultralytics YOLO format.

Remember to double-check if the dataset you want to use is compatible with your model and follows the necessary format conventions. Properly formatted datasets are crucial for training successful object detection models.

## FAQ

### What is the Ultralytics YOLO dataset format and how to structure it?

The Ultralytics YOLO format is a structured configuration for defining datasets in your training projects. It involves setting paths to your training, validation, and testing images and corresponding labels. For example:

```yaml
path: ../datasets/coco8 # dataset root directory
train: images/train # training images (relative to 'path')
val: images/val # validation images (relative to 'path')
test: # optional test images
names:
    0: person
    1: bicycle
    2: car
    # ...
```

Labels are saved in `*.txt` files with one file per image, formatted as `class x_center y_center width height` with normalized coordinates. For a detailed guide, see the [COCO8 dataset example](coco8.md).

### How do I convert a COCO dataset to the YOLO format?

You can convert a COCO dataset to the YOLO format using the Ultralytics conversion tools. Here's a quick method:

```python
from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="path/to/coco/annotations/")
```

This code will convert your COCO annotations to YOLO format, enabling seamless integration with Ultralytics YOLO models. For additional details, visit the [Port or Convert Label Formats](#port-or-convert-label-formats) section.

### Which datasets are supported by Ultralytics YOLO for object detection?

Ultralytics YOLO supports a wide range of datasets, including:

- [Argoverse](argoverse.md)
- [COCO](coco.md)
- [LVIS](lvis.md)
- [COCO8](coco8.md)
- [Global Wheat 2020](globalwheat2020.md)
- [Objects365](objects365.md)
- [OpenImagesV7](open-images-v7.md)

Each dataset page provides detailed information on the structure and usage tailored for efficient YOLO11 training. Explore the full list in the [Supported Datasets](#supported-datasets) section.

### How do I start training a YOLO11 model using my dataset?

To start training a YOLO11 model, ensure your dataset is formatted correctly and the paths are defined in a YAML file. Use the following script to begin training:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")  # Load a pretrained model
        results = model.train(data="path/to/your_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=path/to/your_dataset.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

Refer to the [Usage](#usage) section for more details on utilizing different modes, including CLI commands.

### Where can I find practical examples of using Ultralytics YOLO for object detection?

Ultralytics provides numerous examples and practical guides for using YOLO11 in diverse applications. For a comprehensive overview, visit the [Ultralytics Blog](https://www.ultralytics.com/blog) where you can find case studies, detailed tutorials, and community stories showcasing object detection, segmentation, and more with YOLO11. For specific examples, check the [Usage](../../modes/predict.md) section in the documentation.
