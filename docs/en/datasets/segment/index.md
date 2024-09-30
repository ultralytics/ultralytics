---
comments: true
description: Explore the supported dataset formats for Ultralytics YOLO and learn how to prepare and use datasets for training object segmentation models.
keywords: Ultralytics, YOLO, instance segmentation, dataset formats, auto-annotation, COCO, segmentation models, training data
---

# Instance Segmentation Datasets Overview

## Supported Dataset Formats

### Ultralytics YOLO format

The dataset label format used for training YOLO segmentation models is as follows:

1. One text file per image: Each image in the dataset has a corresponding text file with the same name as the image file and the ".txt" extension.
2. One row per object: Each row in the text file corresponds to one object instance in the image.
3. Object information per row: Each row contains the following information about the object instance:
    - Object class index: An integer representing the class of the object (e.g., 0 for person, 1 for car, etc.).
    - Object bounding coordinates: The bounding coordinates around the mask area, normalized to be between 0 and 1.

The format for a single row in the segmentation dataset file is as follows:

```
<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

In this format, `<class-index>` is the index of the class for the object, and `<x1> <y1> <x2> <y2> ... <xn> <yn>` are the bounding coordinates of the object's segmentation mask. The coordinates are separated by spaces.

Here is an example of the YOLO dataset format for a single image with two objects made up of a 3-point segment and a 5-point segment.

```
0 0.681 0.485 0.670 0.487 0.676 0.487
1 0.504 0.000 0.501 0.004 0.498 0.004 0.493 0.010 0.492 0.0104
```

!!! tip

      - The length of each row does **not** have to be equal.
      - Each segmentation label must have a **minimum of 3 xy points**: `<class-index> <x1> <y1> <x2> <y2> <x3> <y3>`

### Dataset YAML format

The Ultralytics framework uses a YAML file format to define the dataset and model configuration for training Detection Models. Here is an example of the YAML format used for defining a detection dataset:

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco8-seg # dataset root dir
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

The `train` and `val` fields specify the paths to the directories containing the training and validation images, respectively.

`names` is a dictionary of class names. The order of the names should match the order of the object class indices in the YOLO dataset files.

## Usage

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo segment train data=coco8-seg.yaml model=yolov8n-seg.pt epochs=100 imgsz=640
        ```

## Supported Datasets

## Supported Datasets

- [COCO](coco.md): A comprehensive dataset for [object detection](https://www.ultralytics.com/glossary/object-detection), segmentation, and captioning, featuring over 200K labeled images across a wide range of categories.
- [COCO8-seg](coco8-seg.md): A compact, 8-image subset of COCO designed for quick testing of segmentation model training, ideal for CI checks and workflow validation in the `ultralytics` repository.
- [COCO128-seg](coco.md): A smaller dataset for [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) tasks, containing a subset of 128 COCO images with segmentation annotations.
- [Carparts-seg](carparts-seg.md): A specialized dataset focused on the segmentation of car parts, ideal for automotive applications. It includes a variety of vehicles with detailed annotations of individual car components.
- [Crack-seg](crack-seg.md): A dataset tailored for the segmentation of cracks in various surfaces. Essential for infrastructure maintenance and quality control, it provides detailed imagery for training models to identify structural weaknesses.
- [Package-seg](package-seg.md): A dataset dedicated to the segmentation of different types of packaging materials and shapes. It's particularly useful for logistics and warehouse automation, aiding in the development of systems for package handling and sorting.

### Adding your own dataset

If you have your own dataset and would like to use it for training segmentation models with Ultralytics YOLO format, ensure that it follows the format specified above under "Ultralytics YOLO format". Convert your annotations to the required format and specify the paths, number of classes, and class names in the YAML configuration file.

## Port or Convert Label Formats

### COCO Dataset Format to YOLO Format

You can easily convert labels from the popular COCO dataset format to the YOLO format using the following code snippet:

!!! example

    === "Python"

        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco(labels_dir="path/to/coco/annotations/", use_segments=True)
        ```

This conversion tool can be used to convert the COCO dataset or any dataset in the COCO format to the Ultralytics YOLO format.

Remember to double-check if the dataset you want to use is compatible with your model and follows the necessary format conventions. Properly formatted datasets are crucial for training successful object detection models.

## Auto-Annotation

Auto-annotation is an essential feature that allows you to generate a segmentation dataset using a pre-trained detection model. It enables you to quickly and accurately annotate a large number of images without the need for manual labeling, saving time and effort.

### Generate Segmentation Dataset Using a Detection Model

To auto-annotate your dataset using the Ultralytics framework, you can use the `auto_annotate` function as shown below:

!!! example

    === "Python"

        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data="path/to/images", det_model="yolov8x.pt", sam_model="sam_b.pt")
        ```

| Argument     | Type                    | Description                                                                                                 | Default        |
| ------------ | ----------------------- | ----------------------------------------------------------------------------------------------------------- | -------------- |
| `data`       | `str`                   | Path to a folder containing images to be annotated.                                                         | `None`         |
| `det_model`  | `str, optional`         | Pre-trained YOLO detection model. Defaults to `'yolov8x.pt'`.                                               | `'yolov8x.pt'` |
| `sam_model`  | `str, optional`         | Pre-trained SAM segmentation model. Defaults to `'sam_b.pt'`.                                               | `'sam_b.pt'`   |
| `device`     | `str, optional`         | Device to run the models on. Defaults to an empty string (CPU or GPU, if available).                        | `''`           |
| `output_dir` | `str or None, optional` | Directory to save the annotated results. Defaults to a `'labels'` folder in the same directory as `'data'`. | `None`         |

The `auto_annotate` function takes the path to your images, along with optional arguments for specifying the pre-trained detection and [SAM segmentation models](../../models/sam.md), the device to run the models on, and the output directory for saving the annotated results.

By leveraging the power of pre-trained models, auto-annotation can significantly reduce the time and effort required for creating high-quality segmentation datasets. This feature is particularly useful for researchers and developers working with large image collections, as it allows them to focus on model development and evaluation rather than manual annotation.

## FAQ

### What dataset formats does Ultralytics YOLO support for instance segmentation?

Ultralytics YOLO supports several dataset formats for instance segmentation, with the primary format being its own Ultralytics YOLO format. Each image in your dataset needs a corresponding text file with object information segmented into multiple rows (one row per object), listing the class index and normalized bounding coordinates. For more detailed instructions on the YOLO dataset format, visit the [Instance Segmentation Datasets Overview](#instance-segmentation-datasets-overview).

### How can I convert COCO dataset annotations to the YOLO format?

Converting COCO format annotations to YOLO format is straightforward using Ultralytics tools. You can use the `convert_coco` function from the `ultralytics.data.converter` module:

```python
from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="path/to/coco/annotations/", use_segments=True)
```

This script converts your COCO dataset annotations to the required YOLO format, making it suitable for training your YOLO models. For more details, refer to [Port or Convert Label Formats](#coco-dataset-format-to-yolo-format).

### How do I prepare a YAML file for training Ultralytics YOLO models?

To prepare a YAML file for training YOLO models with Ultralytics, you need to define the dataset paths and class names. Here's an example YAML configuration:

```yaml
path: ../datasets/coco8-seg # dataset root dir
train: images/train # train images (relative to 'path')
val: images/val # val images (relative to 'path')

names:
    0: person
    1: bicycle
    2: car
    # ...
```

Ensure you update the paths and class names according to your dataset. For more information, check the [Dataset YAML Format](#dataset-yaml-format) section.

### What is the auto-annotation feature in Ultralytics YOLO?

Auto-annotation in Ultralytics YOLO allows you to generate segmentation annotations for your dataset using a pre-trained detection model. This significantly reduces the need for manual labeling. You can use the `auto_annotate` function as follows:

```python
from ultralytics.data.annotator import auto_annotate

auto_annotate(data="path/to/images", det_model="yolov8x.pt", sam_model="sam_b.pt")
```

This function automates the annotation process, making it faster and more efficient. For more details, explore the [Auto-Annotation](#auto-annotation) section.
