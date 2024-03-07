---
comments: true
description: Navigate through supported dataset formats, methods to utilize them and how to add your own datasets. Get insights on porting or converting label formats.
keywords: Ultralytics, YOLO, datasets, object detection, dataset formats, label formats, data conversion
---

# Object Detection Datasets Overview

Training a robust and accurate object detection model requires a comprehensive dataset. This guide introduces various formats of datasets that are compatible with the Ultralytics YOLO model and provides insights into their structure, usage, and how to convert between different formats.

## Supported Dataset Formats

### Ultralytics YOLO format

The Ultralytics YOLO format is a dataset configuration format that allows you to define the dataset root directory, the relative paths to training/validation/testing image directories or `*.txt` files containing image paths, and a dictionary of class names. Here is an example:

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco8  # dataset root dir
train: images/train  # train images (relative to 'path') 4 images
val: images/val  # val images (relative to 'path') 4 images
test:  # test images (optional)

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

<p align="center"><img width="750" src="https://user-images.githubusercontent.com/26833433/91506361-c7965000-e886-11ea-8291-c72b98c25eec.jpg" alt="Example labelled image"></p>

The label file corresponding to the above image contains 2 persons (class `0`) and a tie (class `27`):

<p align="center"><img width="428" src="https://user-images.githubusercontent.com/26833433/112467037-d2568c00-8d66-11eb-8796-55402ac0d62f.png" alt="Example label file"></p>

When using the Ultralytics YOLO format, organize your training and validation images and labels as shown in the example below.

<p align="center"><img width="700" src="https://user-images.githubusercontent.com/26833433/134436012-65111ad1-9541-4853-81a6-f19a3468b75f.png" alt="Example dataset directory structure"></p>

## Usage

Here's how you can use these formats to train your model:

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=coco8.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Supported Datasets

Here is a list of the supported datasets and a brief description for each:

- [**Argoverse**](argoverse.md): A collection of sensor data collected from autonomous vehicles. It contains 3D tracking annotations for car objects.
- [**COCO**](coco.md): Common Objects in Context (COCO) is a large-scale object detection, segmentation, and captioning dataset with 80 object categories.
- [**COCO8**](coco8.md): A smaller subset of the COCO dataset, COCO8 is more lightweight and faster to train.
- [**GlobalWheat2020**](globalwheat2020.md): A dataset containing images of wheat heads for the Global Wheat Challenge 2020.
- [**Objects365**](objects365.md): A large-scale object detection dataset with 365 object categories and 600k images, aimed at advancing object detection research.
- [**OpenImagesV7**](open-images-v7.md): A comprehensive dataset by Google with 1.7M train images and 42k validation images.
- [**SKU-110K**](sku-110k.md): A dataset containing images of densely packed retail products, intended for retail environment object detection.
- [**VisDrone**](visdrone.md): A dataset focusing on drone-based images, containing various object categories like cars, pedestrians, and cyclists.
- [**VOC**](voc.md): PASCAL VOC is a popular object detection dataset with 20 object categories including vehicles, animals, and furniture.
- [**xView**](xview.md): A dataset containing high-resolution satellite imagery, designed for the detection of various object classes in overhead views.

### Adding your own dataset

If you have your own dataset and would like to use it for training detection models with Ultralytics YOLO format, ensure that it follows the format specified above under "Ultralytics YOLO format". Convert your annotations to the required format and specify the paths, number of classes, and class names in the YAML configuration file.

## Port or Convert Label Formats

### COCO Dataset Format to YOLO Format

You can easily convert labels from the popular COCO dataset format to the YOLO format using the following code snippet:

!!! Example

    === "Python"

        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco(labels_dir='path/to/coco/annotations/')
        ```

This conversion tool can be used to convert the COCO dataset or any dataset in the COCO format to the Ultralytics YOLO format.

Remember to double-check if the dataset you want to use is compatible with your model and follows the necessary format conventions. Properly formatted datasets are crucial for training successful object detection models.
