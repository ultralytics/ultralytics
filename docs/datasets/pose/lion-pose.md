---
comments: true
description: Discover the versatile Lion-Pose dataset, perfect for testing and debugging pose detection models. Learn how to get started with YOLOv8-pose model training.
keywords: Ultralytics, YOLOv8, pose detection, COCO8-Pose dataset, dataset, model training, YAML
---

# Lion-Pose Dataset

## Introduction

[Ultralytics](https://ultralytics.com) Lion-Pose is a small, but versatile pose detection dataset composed of 
the 
250 images collected from a [YouTube Video](https://www.youtube.com/watch?v=MIBAT6BGE6U&pp=ygUbVGlnZXIgd2Fsa2luZyByZWZlcmVuY2UubXA0), 210 for training and 53 for validation. This dataset is ideal for testing and debugging lion detection and pose estimation, or for experimenting with new detection approaches. With 210 images, it is small 
to be easily manageable, yet diverse enough to test training pipelines for errors and act as a sanity check before training larger datasets.

This dataset is intended for use with Ultralytics [HUB](https://hub.ultralytics.com)
and [YOLOv8](https://github.com/ultralytics/ultralytics).

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the Lion-Pose dataset, the `lion-pose.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lion-pose.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lion-pose.yaml).

!!! example "ultralytics/cfg/datasets/lion-pose.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/lion-pose.yaml"
    ```

## Usage

To train a YOLOv8n-pose model on the Lion-Pose dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data='lion-pose.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=lion-pose.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

Here are some examples of images from the Lion-Pose dataset, along with their corresponding annotations:

<img src="https://github.com/RizwanMunawar/ultralytics/assets/62513924/c963d2bf-505f-4a15-abd7-259de302cffa" alt="Dataset sample image" width="800">

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the Lion-Pose dataset and the benefits of using mosaicing during the training process.

## Citations and Acknowledgments

If you use the Lion-Pose dataset in your research or development work, please cite the following paper:

!!! note ""

    === "BibTeX"

        ```bibtex
        @misc{lin2015microsoft,
              title={Microsoft COCO: Common Objects in Context},
              author={Tsung-Yi Lin and Michael Maire and Serge Belongie and Lubomir Bourdev and Ross Girshick and James Hays and Pietro Perona and Deva Ramanan and C. Lawrence Zitnick and Piotr Dollár},
              year={2015},
              eprint={1405.0312},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```
