---
comments: true
description: Learn how to format your dataset for training YOLO models with Ultralytics YOLO format using our concise tutorial and example YAML files.
keywords: pose estimation, datasets, supported formats, YAML file, object class index, keypoints, ultralytics YOLO format
---

# Pose Estimation Datasets Overview

## Supported Dataset Formats

### Ultralytics YOLO format

** Label Format **

The dataset format used for training YOLO pose models is as follows:

1. One text file per image: Each image in the dataset has a corresponding text file with the same name as the image file and the ".txt" extension.
2. One row per object: Each row in the text file corresponds to one object instance in the image.
3. Object information per row: Each row contains the following information about the object instance:
    - Object class index: An integer representing the class of the object (e.g., 0 for person, 1 for car, etc.).
    - Object center coordinates: The x and y coordinates of the center of the object, normalized to be between 0 and 1.
    - Object width and height: The width and height of the object, normalized to be between 0 and 1.
    - Object keypoint coordinates: The keypoints of the object, normalized to be between 0 and 1.

Here is an example of the label format for pose estimation task:

Format with Dim = 2

```
<class-index> <x> <y> <width> <height> <px1> <py1> <px2> <py2> ... <pxn> <pyn>
```

Format with Dim = 3

```
<class-index> <x> <y> <width> <height> <px1> <py1> <p1-visibility> <px2> <py2> <p2-visibility> <pxn> <pyn> <p2-visibility>
```

In this format, `<class-index>` is the index of the class for the object,`<x> <y> <width> <height>` are coordinates of boudning box, and `<px1> <py1> <px2> <py2> ... <pxn> <pyn>` are the pixel coordinates of the keypoints. The coordinates are separated by spaces.

** Dataset file format **

The Ultralytics framework uses a YAML file format to define the dataset and model configuration for training Detection Models. Here is an example of the YAML format used for defining a detection dataset:

```yaml
train: <path-to-training-images>
val: <path-to-validation-images>

nc: <number-of-classes>
names: [<class-1>, <class-2>, ..., <class-n>]

# Keypoints
kpt_shape: [num_kpts, dim]  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
flip_idx: [n1, n2 ... , n(num_kpts)]
```

The `train` and `val` fields specify the paths to the directories containing the training and validation images, respectively.

The `nc` field specifies the number of object classes in the dataset.

The `names` field is a list of the names of the object classes. The order of the names should match the order of the object class indices in the YOLO dataset files.

NOTE: Either `nc` or `names` must be defined. Defining both are not mandatory

Alternatively, you can directly define class names like this:

```yaml
names:
  0: person
  1: bicycle
```

(Optional) if the points are symmetric then need flip_idx, like left-right side of human or face.
For example let's say there're five keypoints of facial landmark: [left eye, right eye, nose, left point of mouth, right point of mouse], and the original index is [0, 1, 2, 3, 4], then flip_idx is [1, 0, 2, 4, 3].(just exchange the left-right index, i.e 0-1 and 3-4, and do not modify others like nose in this example)

** Example **

```yaml
train: data/train/
val: data/val/

nc: 2
names: ['person', 'car']

# Keypoints
kpt_shape: [17, 3]  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
```

## Usage

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)

        # Train the model
        model.train(data='coco128-pose.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"
    
        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=coco128-pose.yaml model=yolov8n-pose.pt epochs=100 imgsz=640
        ```

## Supported Datasets

TODO

## Port or Convert label formats

### COCO dataset format to YOLO format

```python
from ultralytics.yolo.data.converter import convert_coco

convert_coco(labels_dir='../coco/annotations/', use_keypoints=True)
```
