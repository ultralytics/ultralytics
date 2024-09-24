---
comments: true
description: Explore the xView dataset, a rich resource of 1M+ object instances in high-resolution satellite imagery. Enhance detection, learning efficiency, and more.
keywords: xView dataset, overhead imagery, satellite images, object detection, high resolution, bounding boxes, computer vision, TensorFlow, PyTorch, dataset structure
---

# xView Dataset

The [xView](http://xviewdataset.org/) dataset is one of the largest publicly available datasets of overhead imagery, containing images from complex scenes around the world annotated using bounding boxes. The goal of the xView dataset is to accelerate progress in four [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) frontiers:

1. Reduce minimum resolution for detection.
2. Improve learning efficiency.
3. Enable discovery of more object classes.
4. Improve detection of fine-grained classes.

xView builds on the success of challenges like Common Objects in Context (COCO) and aims to leverage computer vision to analyze the growing amount of available imagery from space in order to understand the visual world in new ways and address a range of important applications.

## Key Features

- xView contains over 1 million object instances across 60 classes.
- The dataset has a resolution of 0.3 meters, providing higher resolution imagery than most public satellite imagery datasets.
- xView features a diverse collection of small, rare, fine-grained, and multi-type objects with [bounding box](https://www.ultralytics.com/glossary/bounding-box) annotation.
- Comes with a pre-trained baseline model using the TensorFlow object detection API and an example for [PyTorch](https://www.ultralytics.com/glossary/pytorch).

## Dataset Structure

The xView dataset is composed of satellite images collected from WorldView-3 satellites at a 0.3m ground sample distance. It contains over 1 million objects across 60 classes in over 1,400 km² of imagery.

## Applications

The xView dataset is widely used for training and evaluating deep learning models for object detection in overhead imagery. The dataset's diverse set of object classes and high-resolution imagery make it a valuable resource for researchers and practitioners in the field of computer vision, especially for satellite imagery analysis.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the xView dataset, the `xView.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/xView.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/xView.yaml).

!!! example "ultralytics/cfg/datasets/xView.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/xView.yaml"
    ```

## Usage

To train a model on the xView dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="xView.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=xView.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

The xView dataset contains high-resolution satellite images with a diverse set of objects annotated using bounding boxes. Here are some examples of data from the dataset, along with their corresponding annotations:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/overhead-imagery-object-detection.avif)

- **Overhead Imagery**: This image demonstrates an example of [object detection](https://www.ultralytics.com/glossary/object-detection) in overhead imagery, where objects are annotated with bounding boxes. The dataset provides high-resolution satellite images to facilitate the development of models for this task.

The example showcases the variety and complexity of the data in the xView dataset and highlights the importance of high-quality satellite imagery for object detection tasks.

## Citations and Acknowledgments

If you use the xView dataset in your research or development work, please cite the following paper:

!!! quote ""

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

## FAQ

### What is the xView dataset and how does it benefit computer vision research?

The [xView](http://xviewdataset.org/) dataset is one of the largest publicly available collections of high-resolution overhead imagery, containing over 1 million object instances across 60 classes. It is designed to enhance various facets of computer vision research such as reducing the minimum resolution for detection, improving learning efficiency, discovering more object classes, and advancing fine-grained object detection.

### How can I use Ultralytics YOLO to train a model on the xView dataset?

To train a model on the xView dataset using Ultralytics YOLO, follow these steps:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="xView.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=xView.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

For detailed arguments and settings, refer to the model [Training](../../modes/train.md) page.

### What are the key features of the xView dataset?

The xView dataset stands out due to its comprehensive set of features:

- Over 1 million object instances across 60 distinct classes.
- High-resolution imagery at 0.3 meters.
- Diverse object types including small, rare, and fine-grained objects, all annotated with bounding boxes.
- Availability of a pre-trained baseline model and examples in [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) and PyTorch.

### What is the dataset structure of xView, and how is it annotated?

The xView dataset comprises high-resolution satellite images collected from WorldView-3 satellites at a 0.3m ground sample distance. It encompasses over 1 million objects across 60 classes in approximately 1,400 km² of imagery. Each object within the dataset is annotated with bounding boxes, making it ideal for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models for object detection in overhead imagery. For a detailed overview, you can look at the dataset structure section [here](#dataset-structure).

### How do I cite the xView dataset in my research?

If you utilize the xView dataset in your research, please cite the following paper:

!!! quote ""

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

For more information about the xView dataset, visit the official [xView dataset website](http://xviewdataset.org/).
