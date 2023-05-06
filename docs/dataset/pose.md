comments: true
---

## Supported Dataset Formats

### Ultralytics YOLO format

** Label Format **

Here is an example of the label format for pose estimation task:

Format with Dim = 2

```
<class-index> <x> <y> <width> <height> <px1> <py1> <px2> <py2>  <pxn> <pyn>
```
Format with Dim = 3

```
<class-index> <x> <y> <width> <height> <px1> <py1> <p1-visibility> <px2> <py2> <p2-visibility> <pxn> <pyn> <p2-visibility>
```

In this format, `<class-index>` is the index of the class for the object,`<x> <y> <width> <height>` are coordinates of boudning box, and `<px1> <py1> <px2> <py2>  <pxn> <pyn>` are the pixel coordinates of the keypoints. The coordinates are separated by commas, and multiple coordinates are separated by spaces. 


** Dataset file format **

The Ultralytics framework uses a YAML file format to define the dataset and model configuration for training Detection Models. Here is an example of the YAML format used for defining a detection dataset:

```
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
```
names:
  0: person
  1: bicycle
```

(Optional) if the points are symmetric then need flip_idx, like left-right side of human or face.
For example let's say there're five keypoints of facial landmark: [left eye, right eye, nose, left point of mouth, right point of mouse], and the original index is [0, 1, 2, 3, 4], then flip_idx is [1, 0, 2, 4, 3].(just exchange the left-right index, i.e 0-1 and 3-4, and do not modify others like nose in this example) 

** Example **

```
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

```
from ultralytics.yolo.data.converter import convert_coco

convert_coco(labels_dir='../coco/annotations/', use_keypoints=True)
```
