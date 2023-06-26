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

### Dataset YAML format

The Ultralytics framework uses a YAML file format to define the dataset and model configuration for training Detection Models. Here is an example of the YAML format used for defining a detection dataset:

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco8-pose  # dataset root dir
train: images/train  # train images (relative to 'path') 4 images
val: images/val  # val images (relative to 'path') 4 images
test:  # test images (optional)

# Keypoints
kpt_shape: [17, 3]  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# Classes dictionary
names:
  0: person
```

The `train` and `val` fields specify the paths to the directories containing the training and validation images, respectively.

`names` is a dictionary of class names. The order of the names should match the order of the object class indices in the YOLO dataset files.

(Optional) if the points are symmetric then need flip_idx, like left-right side of human or face.
For example if we assume five keypoints of facial landmark: [left eye, right eye, nose, left mouth, right mouth], and the original index is [0, 1, 2, 3, 4], then flip_idx is [1, 0, 2, 4, 3] (just exchange the left-right index, i.e 0-1 and 3-4, and do not modify others like nose in this example).

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

This section outlines the datasets that are compatible with Ultralytics YOLO format and can be used for training pose estimation models:

### COCO-Pose

- **Description**: COCO-Pose is a large-scale object detection, segmentation, and pose estimation dataset. It is a subset of the popular COCO dataset and focuses on human pose estimation. COCO-Pose includes multiple keypoints for each human instance.
- **Label Format**: Same as Ultralytics YOLO format as described above, with keypoints for human poses.
- **Number of Classes**: 1 (Human).
- **Keypoints**: 17 keypoints including nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles.
- **Usage**: Suitable for training human pose estimation models.
- **Additional Notes**: The dataset is rich and diverse, containing over 200k labeled images.
- [Read more about COCO-Pose](./coco.md)

### COCO8-Pose

- **Description**: [Ultralytics](https://ultralytics.com) COCO8-Pose is a small, but versatile pose detection dataset composed of the first 8 images of the COCO train 2017 set, 4 for training and 4 for validation.
- **Label Format**: Same as Ultralytics YOLO format as described above, with keypoints for human poses.
- **Number of Classes**: 1 (Human).
- **Keypoints**: 17 keypoints including nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles.
- **Usage**: Suitable for testing and debugging object detection models, or for experimenting with new detection approaches.
- **Additional Notes**: COCO8-Pose is ideal for sanity checks and CI checks.
- [Read more about COCO8-Pose](./coco8-pose.md)

### Adding your own dataset

If you have your own dataset and would like to use it for training pose estimation models with Ultralytics YOLO format, ensure that it follows the format specified above under "Ultralytics YOLO format". Convert your annotations to the required format and specify the paths, number of classes, and class names in the YAML configuration file.

### Conversion Tool

Ultralytics provides a convenient conversion tool to convert labels from the popular COCO dataset format to YOLO format:

```python
from ultralytics.yolo.data.converter import convert_coco

convert_coco(labels_dir='../coco/annotations/', use_keypoints=True)
```

This conversion tool can be used to convert the COCO dataset or any dataset in the COCO format to the Ultralytics YOLO format. The `use_keypoints` parameter specifies whether to include keypoints (for pose estimation) in the converted labels.
