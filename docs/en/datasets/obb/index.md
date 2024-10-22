---
comments: true
description: Discover OBB dataset formats for Ultralytics YOLO models. Learn about their structure, application, and format conversions to enhance your object detection training.
keywords: Oriented Bounding Box, OBB Datasets, YOLO, Ultralytics, Object Detection, Dataset Formats
---

# Oriented Bounding Box (OBB) Datasets Overview

Training a precise [object detection](https://www.ultralytics.com/glossary/object-detection) model with oriented bounding boxes (OBB) requires a thorough dataset. This guide explains the various OBB dataset formats compatible with Ultralytics YOLO models, offering insights into their structure, application, and methods for format conversions.

## Supported OBB Dataset Formats

### YOLO OBB Format

The YOLO OBB format designates bounding boxes by their four corner points with coordinates normalized between 0 and 1. It follows this format:

```bash
class_index x1 y1 x2 y2 x3 y3 x4 y4
```

Internally, YOLO processes losses and outputs in the `xywhr` format, which represents the [bounding box](https://www.ultralytics.com/glossary/bounding-box)'s center point (xy), width, height, and rotation.

<p align="center"><img width="800" src="https://github.com/ultralytics/docs/releases/download/0/obb-format-examples.avif" alt="OBB format examples"></p>

An example of a `*.txt` label file for the above image, which contains an object of class `0` in OBB format, could look like:

```bash
0 0.780811 0.743961 0.782371 0.74686 0.777691 0.752174 0.776131 0.749758
```

## Usage

To train a model using these OBB formats:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Create a new YOLO11n-OBB model from scratch
        model = YOLO("yolo11n-obb.yaml")

        # Train the model on the DOTAv1 dataset
        results = model.train(data="DOTAv1.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        # Train a new YOLO11n-OBB model on the DOTAv1 dataset
        yolo obb train data=DOTAv1.yaml model=yolo11n-obb.pt epochs=100 imgsz=1024
        ```

## Supported Datasets

Currently, the following datasets with Oriented Bounding Boxes are supported:

- [DOTA-v1](dota-v2.md): The first version of the DOTA dataset, providing a comprehensive set of aerial images with oriented bounding boxes for object detection.
- [DOTA-v1.5](dota-v2.md): An intermediate version of the DOTA dataset, offering additional annotations and improvements over DOTA-v1 for enhanced object detection tasks.
- [DOTA-v2](dota-v2.md): DOTA (A Large-scale Dataset for Object Detection in Aerial Images) version 2, emphasizes detection from aerial perspectives and contains oriented bounding boxes with 1.7 million instances and 11,268 images.
- [DOTA8](dota8.md): A small, 8-image subset of the full DOTA dataset suitable for testing workflows and Continuous Integration (CI) checks of OBB training in the `ultralytics` repository.

### Incorporating your own OBB dataset

For those looking to introduce their own datasets with oriented bounding boxes, ensure compatibility with the "YOLO OBB format" mentioned above. Convert your annotations to this required format and detail the paths, classes, and class names in a corresponding YAML configuration file.

## Convert Label Formats

### DOTA Dataset Format to YOLO OBB Format

Transitioning labels from the DOTA dataset format to the YOLO OBB format can be achieved with this script:

!!! example

    === "Python"

        ```python
        from ultralytics.data.converter import convert_dota_to_yolo_obb

        convert_dota_to_yolo_obb("path/to/DOTA")
        ```

This conversion mechanism is instrumental for datasets in the DOTA format, ensuring alignment with the Ultralytics YOLO OBB format.

It's imperative to validate the compatibility of the dataset with your model and adhere to the necessary format conventions. Properly structured datasets are pivotal for training efficient object detection models with oriented bounding boxes.

## FAQ

### What are Oriented Bounding Boxes (OBB) and how are they used in Ultralytics YOLO models?

Oriented Bounding Boxes (OBB) are a type of bounding box annotation where the box can be rotated to align more closely with the object being detected, rather than just being axis-aligned. This is particularly useful in aerial or satellite imagery where objects might not be aligned with the image axes. In Ultralytics YOLO models, OBBs are represented by their four corner points in the YOLO OBB format. This allows for more accurate object detection since the bounding boxes can rotate to fit the objects better.

### How do I convert my existing DOTA dataset labels to YOLO OBB format for use with Ultralytics YOLO11?

You can convert DOTA dataset labels to YOLO OBB format using the `convert_dota_to_yolo_obb` function from Ultralytics. This conversion ensures compatibility with the Ultralytics YOLO models, enabling you to leverage the OBB capabilities for enhanced object detection. Here's a quick example:

```python
from ultralytics.data.converter import convert_dota_to_yolo_obb

convert_dota_to_yolo_obb("path/to/DOTA")
```

This script will reformat your DOTA annotations into a YOLO-compatible format.

### How do I train a YOLO11 model with oriented bounding boxes (OBB) on my dataset?

Training a YOLO11 model with OBBs involves ensuring your dataset is in the YOLO OBB format and then using the Ultralytics API to train the model. Here's an example in both Python and CLI:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Create a new YOLO11n-OBB model from scratch
        model = YOLO("yolo11n-obb.yaml")

        # Train the model on the custom dataset
        results = model.train(data="your_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Train a new YOLO11n-OBB model on the custom dataset
        yolo obb train data=your_dataset.yaml model=yolo11n-obb.yaml epochs=100 imgsz=640
        ```

This ensures your model leverages the detailed OBB annotations for improved detection [accuracy](https://www.ultralytics.com/glossary/accuracy).

### What datasets are currently supported for OBB training in Ultralytics YOLO models?

Currently, Ultralytics supports the following datasets for OBB training:

- [DOTA-v1](dota-v2.md): The first version of the DOTA dataset, providing a comprehensive set of aerial images with oriented bounding boxes for object detection.
- [DOTA-v1.5](dota-v2.md): An intermediate version of the DOTA dataset, offering additional annotations and improvements over DOTA-v1 for enhanced object detection tasks.
- [DOTA-v2](dota-v2.md): This dataset includes 1.7 million instances with oriented bounding boxes and 11,268 images, primarily focusing on aerial object detection.
- [DOTA8](dota8.md): A smaller, 8-image subset of the DOTA dataset used for testing and continuous integration (CI) checks.

These datasets are tailored for scenarios where OBBs offer a significant advantage, such as aerial and satellite image analysis.

### Can I use my own dataset with oriented bounding boxes for YOLO11 training, and if so, how?

Yes, you can use your own dataset with oriented bounding boxes for YOLO11 training. Ensure your dataset annotations are converted to the YOLO OBB format, which involves defining bounding boxes by their four corner points. You can then create a YAML configuration file specifying the dataset paths, classes, and other necessary details. For more information on creating and configuring your datasets, refer to the [Supported Datasets](#supported-datasets) section.
