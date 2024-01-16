---
comments: true
description: Explore xView, a large-scale, high resolution satellite imagery dataset for object detection. Dive into dataset structure, usage examples & its potential applications.
keywords: Ultralytics, YOLO, computer vision, xView dataset, satellite imagery, object detection, overhead imagery, training, deep learning, dataset YAML
---

# xView Dataset

The [xView](http://xviewdataset.org/) dataset is one of the largest publicly available datasets of overhead imagery, containing images from complex scenes around the world annotated using bounding boxes. The goal of the xView dataset is to accelerate progress in four computer vision frontiers:

1. Reduce minimum resolution for detection.
2. Improve learning efficiency.
3. Enable discovery of more object classes.
4. Improve detection of fine-grained classes.

xView builds on the success of challenges like Common Objects in Context (COCO) and aims to leverage computer vision to analyze the growing amount of available imagery from space in order to understand the visual world in new ways and address a range of important applications.

## Key Features

- xView contains over 1 million object instances across 60 classes.
- The dataset has a resolution of 0.3 meters, providing higher resolution imagery than most public satellite imagery datasets.
- xView features a diverse collection of small, rare, fine-grained, and multi-type objects with bounding box annotation.
- Comes with a pre-trained baseline model using the TensorFlow object detection API and an example for PyTorch.

## Dataset Structure

The xView dataset is composed of satellite images collected from WorldView-3 satellites at a 0.3m ground sample distance. It contains over 1 million objects across 60 classes in over 1,400 km² of imagery.

## Applications

The xView dataset is widely used for training and evaluating deep learning models for object detection in overhead imagery. The dataset's diverse set of object classes and high-resolution imagery make it a valuable resource for researchers and practitioners in the field of computer vision, especially for satellite imagery analysis.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the xView dataset, the `xView.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/xView.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/xView.yaml).

!!! Example "ultralytics/cfg/datasets/xView.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/xView.yaml"
    ```

## Usage

To train a model on the xView dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! Example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data='xView.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=xView.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

The xView dataset contains high-resolution satellite images with a diverse set of objects annotated using bounding boxes. Here are some examples of data from the dataset, along with their corresponding annotations:

![Dataset sample image](https://user-images.githubusercontent.com/26833433/277141257-ae6ba4de-5dcb-4c76-bc05-bc1e386361ba.jpg)

- **Overhead Imagery**: This image demonstrates an example of object detection in overhead imagery, where objects are annotated with bounding boxes. The dataset provides high-resolution satellite images to facilitate the development of models for this task.

The example showcases the variety and complexity of the data in the xView dataset and highlights the importance of high-quality satellite imagery for object detection tasks.

## Citations and Acknowledgments

If you use the xView dataset in your research or development work, please cite the following paper:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{lam2018xview,
              title={xView: Objects in Context in Overhead Imagery},
              author={Darius Lam and Richard Kuzma and Kevin McGee and Samuel Dooley and Michael Laielli and Matthew Klaric and Yaroslav Bulatov and Brendan McCord},
              year={2018},
              eprint={1802.07856},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

We would like to acknowledge the [Defense Innovation Unit](https://www.diu.mil/) (DIU) and the creators of the xView dataset for their valuable contribution to the computer vision research community. For more information about the xView dataset and its creators, visit the [xView dataset website](http://xviewdataset.org/).
