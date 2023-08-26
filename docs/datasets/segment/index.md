---
comments: true
description: Learn how Ultralytics YOLO supports various dataset formats for instance segmentation. This guide includes information on data conversions, auto-annotations, and dataset usage.
keywords: Ultralytics, YOLO, Instance Segmentation, Dataset, YAML, COCO, Auto-Annotation, Image Segmentation
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

!!! tip "Tip"

      - The length of each row does **not** have to be equal.
      - Each segmentation label must have a **minimum of 3 xy points**: `<class-index> <x1> <y1> <x2> <y2> <x3> <y3>`

### Dataset YAML format

The Ultralytics framework uses a YAML file format to define the dataset and model configuration for training Detection Models. Here is an example of the YAML format used for defining a detection dataset:

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco8-seg  # dataset root dir
train: images/train  # train images (relative to 'path') 4 images
val: images/val  # val images (relative to 'path') 4 images
test:  # test images (optional)

# Classes (80 COCO classes)
names:
  0: person
  1: bicycle
  2: car
  ...
  77: teddy bear
  78: hair drier
  79: toothbrush
```

The `train` and `val` fields specify the paths to the directories containing the training and validation images, respectively.

`names` is a dictionary of class names. The order of the names should match the order of the object class indices in the YOLO dataset files.

## Usage

!!! example ""

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data='coco128-seg.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=coco128-seg.yaml model=yolov8n-seg.pt epochs=100 imgsz=640
        ```

## Supported Datasets

* [COCO](coco.md): A large-scale dataset designed for object detection, segmentation, and captioning tasks with over 200K labeled images.
* [COCO8-seg](coco8-seg.md): A smaller dataset for instance segmentation tasks, containing a subset of 8 COCO images with segmentation annotations.

### Adding your own dataset

If you have your own dataset and would like to use it for training segmentation models with Ultralytics YOLO format, ensure that it follows the format specified above under "Ultralytics YOLO format". Convert your annotations to the required format and specify the paths, number of classes, and class names in the YAML configuration file.

## Port or Convert Label Formats

### COCO Dataset Format to YOLO Format

You can easily convert labels from the popular COCO dataset format to the YOLO format using the following code snippet:

!!! example ""

    === "Python"

        ```python
        from ultralytics.data.converter import convert_coco
        
        convert_coco(labels_dir='path/to/coco/annotations/', use_segments=True)
        ```

This conversion tool can be used to convert the COCO dataset or any dataset in the COCO format to the Ultralytics YOLO format.

Remember to double-check if the dataset you want to use is compatible with your model and follows the necessary format conventions. Properly formatted datasets are crucial for training successful object detection models.

## Auto-Annotation

Auto-annotation is an essential feature that allows you to generate a segmentation dataset using a pre-trained detection model. It enables you to quickly and accurately annotate a large number of images without the need for manual labeling, saving time and effort.

### Generate Segmentation Dataset Using a Detection Model

To auto-annotate your dataset using the Ultralytics framework, you can use the `auto_annotate` function as shown below:

!!! example ""

    === "Python"

        ```python
        from ultralytics.data.annotator import auto_annotate
         
        auto_annotate(data="path/to/images", det_model="yolov8x.pt", sam_model='sam_b.pt')
        ```

Certainly, here is the table updated with code snippets:

| Argument     | Type                    | Description                                                                                                 | Default        |
|--------------|-------------------------|-------------------------------------------------------------------------------------------------------------|----------------|
| `data`       | `str`                   | Path to a folder containing images to be annotated.                                                         | `None`         |
| `det_model`  | `str, optional`         | Pre-trained YOLO detection model. Defaults to `'yolov8x.pt'`.                                               | `'yolov8x.pt'` |
| `sam_model`  | `str, optional`         | Pre-trained SAM segmentation model. Defaults to `'sam_b.pt'`.                                               | `'sam_b.pt'`   |
| `device`     | `str, optional`         | Device to run the models on. Defaults to an empty string (CPU or GPU, if available).                        | `''`           |
| `output_dir` | `str or None, optional` | Directory to save the annotated results. Defaults to a `'labels'` folder in the same directory as `'data'`. | `None`         |

The `auto_annotate` function takes the path to your images, along with optional arguments for specifying the pre-trained detection and [SAM segmentation models](https://docs.ultralytics.com/models/sam), the device to run the models on, and the output directory for saving the annotated results.

By leveraging the power of pre-trained models, auto-annotation can significantly reduce the time and effort required for creating high-quality segmentation datasets. This feature is particularly useful for researchers and developers working with large image collections, as it allows them to focus on model development and evaluation rather than manual annotation.
