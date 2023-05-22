---
comments: true
description: Learn how torchvision organizes classification image datasets. Use this code to create and train models. CLI and Python code shown.
---

# Image Classification Datasets Overview

## Dataset format

The folder structure for classification datasets in torchvision typically follows a standard format:

```
root/
|-- class1/
|   |-- img1.jpg
|   |-- img2.jpg
|   |-- ...
|
|-- class2/
|   |-- img1.jpg
|   |-- img2.jpg
|   |-- ...
|
|-- class3/
|   |-- img1.jpg
|   |-- img2.jpg
|   |-- ...
|
|-- ...
```

In this folder structure, the `root` directory contains one subdirectory for each class in the dataset. Each subdirectory is named after the corresponding class and contains all the images for that class. Each image file is named uniquely and is typically in a common image file format such as JPEG or PNG.

** Example **

For example, in the CIFAR10 dataset, the folder structure would look like this:

```
cifar-10-/
|
|-- train/
|   |-- airplane/
|   |   |-- 10008_airplane.png
|   |   |-- 10009_airplane.png
|   |   |-- ...
|   |
|   |-- automobile/
|   |   |-- 1000_automobile.png
|   |   |-- 1001_automobile.png
|   |   |-- ...
|   |
|   |-- bird/
|   |   |-- 10014_bird.png
|   |   |-- 10015_bird.png
|   |   |-- ...
|   |
|   |-- ...
|
|-- test/
|   |-- airplane/
|   |   |-- 10_airplane.png
|   |   |-- 11_airplane.png
|   |   |-- ...
|   |
|   |-- automobile/
|   |   |-- 100_automobile.png
|   |   |-- 101_automobile.png
|   |   |-- ...
|   |
|   |-- bird/
|   |   |-- 1000_bird.png
|   |   |-- 1001_bird.png
|   |   |-- ...
|   |
|   |-- ...
```

In this example, the `train` directory contains subdirectories for each class in the dataset, and each class subdirectory contains all the images for that class. The `test` directory has a similar structure. The `root` directory also contains other files that are part of the CIFAR10 dataset.

## Usage

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

        # Train the model
        model.train(data='path/to/dataset', epochs=100, imgsz=640)
        ```
    === "CLI"
    
        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=path/to/data model=yolov8n-seg.pt epochs=100 imgsz=640
        ```

## Supported Datasets

TODO