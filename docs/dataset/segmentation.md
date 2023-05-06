comments: true
---

## Supported Dataset Formats

### Ultralytics YOLO format

** Label Format **

The label format for segmentation tasks is a bit different from the label format for object detection. In segmentation tasks, we need to create pixel-level labels for each object in the image.

Here is an example of the label format for segmentation tasks in YOLOv5:

```
<class-index> <x1>,<y1>,<x2>,<y2> ... <xn>,<yn>
```

In this format, `<class-index>` is the index of the class for the object, and `<x1>,<y1>,<x2>,<y2> ... <xn>,<yn>` are the pixel coordinates of the object's segmentation mask. The coordinates are separated by commas, and multiple coordinates are separated by spaces. 

For example, suppose we have a segmentation mask for an object of class "person" in an image. The mask has a value of 1 for all pixels that belong to the person and a value of 0 for all other pixels. We can create a label file for this object with the following format:

```
0 100,150,200,300 250,100,350,300
```

In this label file, the class index for "person" is 0, and the object has two bounding boxes with coordinates (100,150,200,300) and (250,100,350,300), respectively.

Note that for segmentation tasks in YOLOv5, we use pixel-level labels instead of bounding boxes for each object. Therefore, we need to create segmentation masks for each object in the image and then convert these masks into the label format described above.

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
```
names:
  0: person
  1: bicycle
```

** Example **

```
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
        model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

        # Train the model
        model.train(data='coco128-seg.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"
    
        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=coco128-seg.yaml model=yolov8n-seg.pt epochs=100 imgsz=640
        ```

## Supported Datasets

## Port or Convert label formats

### COCO dataset format to YOLO format

```
from ultralytics.yolo.data.converter import convert_coco

convert_coco(labels_dir='../coco/annotations/', use_segments=True)
```