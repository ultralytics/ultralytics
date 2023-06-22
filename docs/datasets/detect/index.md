---
comments: true
description: Learn about supported dataset formats for training YOLO detection models, including Ultralytics YOLO and COCO, in this Object Detection Datasets Overview.
keywords: object detection, datasets, formats, Ultralytics YOLO, label format, dataset file format, dataset definition, YOLO dataset, model configuration
---

# Object Detection Datasets Overview

## Supported Dataset Formats

### Ultralytics YOLO format

The dataset config file that defines 1) the dataset root directory `path` and relative paths to `train` / `val` / `test` image directories (or *.txt files with image paths) and 2) a class `names` dictionary:

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco128  # dataset root dir
train: images/train2017  # train images (relative to 'path') 128 images
val: images/train2017  # val images (relative to 'path') 128 images
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

### 1.2 Create Labels

After using an annotation tool to label your images, export your labels to **YOLO format**, with one `*.txt` file per image (if no objects in image, no `*.txt` file is required). The `*.txt` file specifications are:

- One row per object
- Each row is `class x_center y_center width height` format.
- Box coordinates must be in **normalized xywh** format (from 0 - 1). If your boxes are in pixels, divide `x_center` and `width` by image width, and `y_center` and `height` by image height.
- Class numbers are zero-indexed (start from 0).

<p align="center"><img width="750" src="https://user-images.githubusercontent.com/26833433/91506361-c7965000-e886-11ea-8291-c72b98c25eec.jpg"></p>

The label file corresponding to the above image contains 2 persons (class `0`) and a tie (class `27`):

<p align="center"><img width="428" src="https://user-images.githubusercontent.com/26833433/112467037-d2568c00-8d66-11eb-8796-55402ac0d62f.png"></p>

### 1.3 Organize Directories

Organize your train and val images and labels according to the example below. YOLOv5 assumes  `/coco128` is inside a `/datasets` directory **next to** the `/yolov5` directory. **YOLOv5 locates labels automatically for each image** by replacing the last instance of `/images/` in each image path with `/labels/`. For example:

```bash
../datasets/coco128/images/im0.jpg  # image
../datasets/coco128/labels/im0.txt  # label
```

<p align="center"><img width="700" src="https://user-images.githubusercontent.com/26833433/134436012-65111ad1-9541-4853-81a6-f19a3468b75f.png"></p>

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

- Argoverse
- COCO
- COCO8
- GlobalWheat2020
- Objects365
- SKU-110K
- VisDrone
- VOC
- xView

## Port or Convert label formats

### COCO dataset format to YOLO format

```python
from ultralytics.yolo.data.converter import convert_coco

convert_coco(labels_dir='../coco/annotations/')
```