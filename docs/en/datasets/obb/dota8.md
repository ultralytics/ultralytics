---
comments: true
description: Explore the DOTA8 dataset - a small, versatile oriented object detection dataset ideal for testing and debugging object detection models using Ultralytics YOLO11.
keywords: DOTA8 dataset, Ultralytics, YOLO11, object detection, debugging, training models, oriented object detection, dataset YAML
---

# DOTA8 Dataset

## Introduction

[Ultralytics](https://www.ultralytics.com/) DOTA8 is a small but versatile oriented [object detection](https://www.ultralytics.com/glossary/object-detection) dataset composed of the first 8 images of the split DOTAv1 set, 4 for training and 4 for validation. This dataset is ideal for testing and debugging object detection models, or for experimenting with new detection approaches. With 8 images, it is small enough to be easily manageable, yet diverse enough to test training pipelines for errors and act as a sanity check before training larger datasets.

## Dataset Structure

- **Images**: 8 aerial tiles (4 train, 4 val) sourced from DOTAv1.
- **Classes**: Inherits the 15 DOTAv1 categories such as plane, ship, and large vehicle.
- **Labels**: YOLO-format oriented bounding boxes saved as `.txt` files beside each image.
- **Recommended layout**:

    ```
    datasets/dota8/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    ```

This dataset is intended for use with Ultralytics [HUB](https://hub.ultralytics.com/) and [YOLO11](https://github.com/ultralytics/ultralytics).

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the DOTA8 dataset, the `dota8.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dota8.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dota8.yaml).

!!! example "ultralytics/cfg/datasets/dota8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/dota8.yaml"
    ```

## Usage

To train a YOLO11n-obb model on the DOTA8 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-obb.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="dota8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo obb train data=dota8.yaml model=yolo11n-obb.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

Here are some examples of images from the DOTA8 dataset, along with their corresponding annotations:

<img src="https://github.com/ultralytics/docs/releases/download/0/mosaiced-training-batch.avif" alt="Dataset sample image" width="800">

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the DOTA8 dataset and the benefits of using mosaicing during the training process.

## Citations and Acknowledgments

If you use the DOTA dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{9560031,
          author={Ding, Jian and Xue, Nan and Xia, Gui-Song and Bai, Xiang and Yang, Wen and Yang, Michael and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          title={Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges},
          year={2021},
          volume={},
          number={},
          pages={1-1},
          doi={10.1109/TPAMI.2021.3117983}
        }
        ```

A special note of gratitude to the team behind the DOTA datasets for their commendable effort in curating this dataset. For an exhaustive understanding of the dataset and its nuances, please visit the [official DOTA website](https://captain-whu.github.io/DOTA/index.html).

## FAQ

### What is the DOTA8 dataset and how can it be used?

The DOTA8 dataset is a small, versatile oriented object detection dataset made up of the first 8 images from the DOTAv1 split set, with 4 images designated for training and 4 for validation. It's ideal for testing and debugging object detection models like Ultralytics YOLO11. Due to its manageable size and diversity, it helps in identifying pipeline errors and running sanity checks before deploying larger datasets. Learn more about object detection with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics).

### How do I train a YOLO11 model using the DOTA8 dataset?

To train a YOLO11n-obb model on the DOTA8 dataset for 100 epochs with an image size of 640, you can use the following code snippets. For comprehensive argument options, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-obb.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="dota8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo obb train data=dota8.yaml model=yolo11n-obb.pt epochs=100 imgsz=640
        ```

### What are the key features of the DOTA dataset and where can I access the YAML file?

The DOTA dataset is known for its large-scale benchmark and the challenges it presents for object detection in aerial images. The DOTA8 subset is a smaller, manageable dataset ideal for initial tests. You can access the `dota8.yaml` file, which contains paths, classes, and configuration details, at this [GitHub link](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dota8.yaml).

### How does mosaicing enhance model training with the DOTA8 dataset?

Mosaicing combines multiple images into one during training, increasing the variety of objects and contexts within each batch. This improves a model's ability to generalize to different object sizes, aspect ratios, and scenes. This technique can be visually demonstrated through a training batch composed of mosaiced DOTA8 dataset images, helping in robust model development. Explore more about mosaicing and training techniques on our [Training](../../modes/train.md) page.

### Why should I use Ultralytics YOLO11 for object detection tasks?

Ultralytics YOLO11 provides state-of-the-art real-time object detection capabilities, including features like oriented bounding boxes (OBB), [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation), and a highly versatile training pipeline. It's suitable for various applications and offers pretrained models for efficient fine-tuning. Explore further about the advantages and usage in the [Ultralytics YOLO11 documentation](https://github.com/ultralytics/ultralytics).
