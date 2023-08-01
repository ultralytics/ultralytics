---
comments: true
description: 'Discover the COCO8-Seg: a compact but versatile instance segmentation dataset ideal for testing Ultralytics YOLOv8 detection approaches. Complete usage guide included.'
keywords: COCO8-Seg dataset, Ultralytics, YOLOv8, instance segmentation, dataset configuration, YAML, YOLOv8n-seg model, mosaiced dataset images
---

# COCO8-Seg Dataset

## Introduction

[Ultralytics](https://ultralytics.com) COCO8-Seg is a small, but versatile instance segmentation dataset composed of the
first 8 images of the COCO train 2017 set, 4 for training and 4 for validation. This dataset is ideal for testing and
debugging segmentation models, or for experimenting with new detection approaches. With 8 images, it is small enough to
be easily manageable, yet diverse enough to test training pipelines for errors and act as a sanity check before training
larger datasets.

This dataset is intended for use with Ultralytics [HUB](https://hub.ultralytics.com)
and [YOLOv8](https://github.com/ultralytics/ultralytics).

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the COCO8-Seg dataset, the `coco8-seg.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-seg.yaml).

!!! example "ultralytics/cfg/datasets/coco8-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8-seg.yaml"
    ```

## Usage

To train a YOLOv8n-seg model on the COCO8-Seg dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

        # Train the model
        model.train(data='coco8-seg.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=coco8-seg.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

Here are some examples of images from the COCO8-Seg dataset, along with their corresponding annotations:

<img src="https://user-images.githubusercontent.com/26833433/236818387-f7bde7df-caaa-46d1-8341-1f7504cd11a1.jpg" alt="Dataset sample image" width="800">

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the COCO8-Seg dataset and the benefits of using mosaicing during the training process.

## Citations and Acknowledgments

If you use the COCO dataset in your research or development work, please cite the following paper:

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

We would like to acknowledge the COCO Consortium for creating and maintaining this valuable resource for the computer vision community. For more information about the COCO dataset and its creators, visit the [COCO dataset website](https://cocodataset.org/#home).
