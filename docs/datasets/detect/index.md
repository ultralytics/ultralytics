---
comments: true
---

# Object Detection Datasets Overview

## Supported Dataset Formats

### Ultralytics YOLO format

** Label Format **

The dataset format used for training YOLO detection models is as follows:

1. One text file per image: Each image in the dataset has a corresponding text file with the same name as the image file and the ".txt" extension.
2. One row per object: Each row in the text file corresponds to one object instance in the image.
3. Object information per row: Each row contains the following information about the object instance:
   - Object class index: An integer representing the class of the object (e.g., 0 for person, 1 for car, etc.).
   - Object center coordinates: The x and y coordinates of the center of the object, normalized to be between 0 and 1.
   - Object width and height: The width and height of the object, normalized to be between 0 and 1.
   
The format for a single row in the detection dataset file is as follows:
```
<object-class> <x> <y> <width> <height>
```

Here is an example of the YOLO dataset format for a single image with two object instances:

```
0 0.5 0.4 0.3 0.6
1 0.3 0.7 0.4 0.2
```

In this example, the first object is of class 0 (person), with its center at (0.5, 0.4), width of 0.3, and height of 0.6. The second object is of class 1 (car), with its center at (0.3, 0.7), width of 0.4, and height of 0.2.

** Dataset file format **

The Ultralytics framework uses a YAML file format to define the dataset and model configuration for training Detection Models. Here is an example of the YAML format used for defining a detection dataset:

```
train: <path-to-training-images>
val: <path-to-validation-images>

nc: <number-of-classes>
names: [<class-1>, <class-2>, ..., <class-n>]

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

** Example **

```yaml
train: data/train/
val: data/val/

nc: 2
names: ['person', 'car']
```

## Usage
!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

        # Train the model
        model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"
    
        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Supported Datasets
TODO

## Port or Convert label formats

### COCO dataset format to YOLO format

```
from ultralytics.yolo.data.converter import convert_coco

convert_coco(labels_dir='../coco/annotations/')
```
