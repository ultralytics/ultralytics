---
comments: true
description: Understand the YOLO multitask dataset format and learn to use Ultralytics datasets to train your multitask models effectively.
keywords: Ultralytics, YOLO, Instance Segmentation, pose estimation, datasets, training, YAML, keypoints, COCO8-multitask, data conversion, Image Segmentation, Multitask
---

# Multitask Datasets Overview

## Supported Dataset Formats

### Ultralytics YOLO format

The dataset label format used for training YOLO multitask models is as follows:

1. One text file per image: Each image in the dataset has a corresponding text file with the same name as the image file and the ".txt" extension.
2. One row per object: Each row in the text file corresponds to one object instance in the image.
3. Object information per row: Each row contains the following information about the object instance:
    - Object class index: An integer representing the class of the object (e.g., 0 for person, 1 for car, etc.).
    - Object keypoint coordinates: The keypoints of the object, normalized to be between 0 and 1.
    - Object bounding coordinates: The bounding coordinates around the mask area, normalized to be between 0 and 1.

Here is an example of the label format for the multitask task:

Format with Dim = 3

```
 <class-index> <px1> <py1> <p1-visibility> ... <pxn> <pyn> <pn-visibility> <x1> <y1> ... <xn> <yn>
```

In this format, `<class-index>` is the index of the class for the object, `<px1> <py1> ... <pxn> <pyn>` are the pixel coordinates of the keypoints and `<x1> <y1> <x2> <y2> ... <xn> <yn>` are the bounding coordinates of the object's segmentation mask. The coordinates are separated by spaces.

Here is an example of the YOLO dataset format for a single image with two objects made up of a (2-point pose, 3-point segment) and a 5-point segment.

```
0 0.675 0.486 2 0.679 0.487 2 0.681 0.485 0.670 0.487 0.676 0.487
1 0.504 0.000 0.501 0.004 0.498 0.004 0.493 0.010 0.492 0.0104
```

!!! Tip "Tip"

      - The length of each row does **not** have to be equal.
      - Not each label contains keypoint labels.
      - For objects which have less than the in `kpt_shape` maximum defined number of keypoints, all missing keypoints need to be labeled `0 0 0`.
      - Each segmentation label must have a **minimum of 3 xy points**: `<class-index> ..<points>.. <x1> <y1> <x2> <y2> <x3> <y3>`
### Dataset YAML format

The Ultralytics framework uses a YAML file format to define the dataset and model configuration for training Detection Models. Here is an example of the YAML format used for defining a detection dataset:

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco8-multitask  # dataset root dir
train: images/train  # train images (relative to 'path') 4 images
val: images/val  # val images (relative to 'path') 4 images
test:  # test images (optional)

# Keypoints
kpt_shape: [17, 3]  # maximum number of keypoints, number of dims (x,y,visible)
flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Classes (80 COCO classes)
names:
  0: person
  1: bicycle
  2: car
  # ...
  77: teddy bear
  78: hair drier
  79: toothbrush

# Classes for keypoint detection
kpt_names:
  0: person

# Download script/URL (optional)
#download: https://ultralytics.com/assets/coco8-multitask.zip
```

The `train` and `val` fields specify the paths to the directories containing the training and validation images, respectively.

`names` is a dictionary of class names. The order of the names should match the order of the object class indices in the YOLO dataset files.

`kpt_names` is a subset of `names` containing class names for which keypoint detection is enabled. The order of the names should match the order in `names`.
 

(Optional if only 1 class in `kpt_names`) if the points are symmetric then need flip_idx, like left-right side of human or face. For example if we assume five keypoints of facial landmark: [left eye, right eye, nose, left mouth, right mouth], and the original index is [0, 1, 2, 3, 4], then flip_idx is [1, 0, 2, 4, 3] (just exchange the left-right index, i.e. 0-1 and 3-4, and do not modify others like nose in this example).
If keypoints for multiple classes get predicted, `flip_idx` *must* be in the natural order.

## Usage

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-multitask.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco8-multitask.yaml", epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo multitask train data=coco8-multitask.yaml model=yolov8n-multitask.pt epochs=100 imgsz=640
        ```

## Supported Datasets

This section outlines the datasets that are compatible with Ultralytics YOLO format and can be used for training multitask models:

### COCO8-Multitask

- **Description**: [Ultralytics](https://ultralytics.com) COCO8-multitask is a small, but versatile multitask dataset composed of 8 images of the COCO train 2017 set, 4 for training and 4 for validation.
- **Label Format**: Same as Ultralytics YOLO format as described above, with keypoints for human poses and segments for instance segmentation
- **Number of Classes**: 80.
- **Number of Classes with Keypoints** 1 (Human).
- **Keypoints**: 17 keypoints.
- **Usage**: Suitable for testing and debugging object detection models, or for experimenting with new detection approaches.
- **Additional Notes**: COCO8-multitask is ideal for sanity checks and CI checks.
- [Read more about COCO8-multitask](coco8-multitask.md)

### Adding your own dataset

If you have your own dataset and would like to use it for training multitask models with Ultralytics YOLO format, ensure that it follows the format specified above under "Ultralytics YOLO format". Convert your annotations to the required format and specify the paths, number of classes, and class names in the YAML configuration file.

### Conversion Tool
Conversion of a Dataset is a two-step process, first convert the dataset in COCO format to seperate keypoint and segmentation datasets.
Then merge the two datasets to a the YOLO multitask format.

Ultralytics provides a convenient conversion tool to convert labels from the popular COCO dataset format to YOLO format:

!!! Example

    === "Python"

        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco(labels_dir="path/to/coco/annotations/", use_keypoints=True)
        convert_coco(labels_dir="path/to/coco/annotations/", use_segments=True)
        ```

        After converting the datasets, merge them using the [merge_kpt_seg.py](https://github.com/stedavkle/ultralytics/tree/multitask/ultralytics/data/scripts/merge_kpt_seg.py) script:
        ```bash
        python merge_kpt_seg.py -kpt path/to/keypoint_dataset -seg path/to/segmentation_dataset -o path/to/output_dataset
        ```

This conversion tool can be used to convert the COCO dataset or any dataset in the COCO format to the Ultralytics YOLO format.
Remember to double-check if the dataset you want to use is compatible with your model and follows the necessary format conventions. Properly formatted datasets are crucial for training successful object detection models.
