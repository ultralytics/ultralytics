---
title: DOTA8 OBB Dataset
comments: true
creator:
    name: Ultralytics
    url: https://www.ultralytics.com
license:
    name: Research-Only
    url: https://captain-whu.github.io/DOTA/dataset.html
description: DOTA8 is an 8-image DOTAv1 subset (4 train, 4 val) for oriented bounding box (OBB) detection, ideal for testing and debugging Ultralytics YOLO26 OBB pipelines.
keywords: DOTA8 dataset, Ultralytics, YOLO26, object detection, debugging, training models, oriented object detection, dataset YAML
---

# DOTA8 Dataset

## Introduction

[Ultralytics](https://www.ultralytics.com) DOTA8 is a small oriented [object detection](https://www.ultralytics.com/glossary/object-detection) dataset composed of the first 8 images of the split DOTAv1 set, 4 for training and 4 for validation. This dataset is ideal for testing and debugging object detection models, or for experimenting with new detection approaches. With 8 images, it is small enough to be easily manageable, yet diverse enough to test training pipelines for errors and act as a sanity check before training larger datasets.

## Dataset Structure

- **Images**: 8 aerial tiles (4 train, 4 val) sourced from DOTAv1.
- **Classes**: Inherits the 15 DOTAv1 categories such as plane, ship, and large vehicle.
- **Labels**: YOLO-format oriented bounding boxes saved as `.txt` files beside each image.
- **Download**: 1 MB, fetched automatically from Ultralytics GitHub assets the first time you train.
- **Recommended layout**:

    ```text
    datasets/dota8/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    ```

Explore [DOTA8 on Ultralytics Platform](https://platform.ultralytics.com/ultralytics/datasets/dota8) to browse the aerial tiles with their oriented-box overlays, view the class distribution and annotation heatmaps in the **Charts** tab, and clone it to train your own model in the cloud.

## Dataset YAML

A YAML file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the DOTA8 dataset, the `dota8.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dota8.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dota8.yaml).

!!! example "ultralytics/cfg/datasets/dota8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/dota8.yaml"
    ```

## Usage

To train a [YOLO26n-obb](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-obb) model on the DOTA8 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-obb.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="dota8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo obb train data=dota8.yaml model=yolo26n-obb.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

Here are some examples of images from the DOTA8 dataset, along with their corresponding annotations:

<img src="https://cdn.ul.run/i/abcdbb24150fbc2a192327b59185ddbb.avif" alt="DOTA8 oriented bounding box dataset training mosaic" width="800">

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the DOTA8 dataset and the benefits of using mosaicing during the training process.

## Citations and Acknowledgments

If you use the DOTA dataset in your research or development work, please cite the following paper:

!!! quote "Citation"

    === "BibTeX"

        ```bibtex
        @article{9560031,
          author={Ding, Jian and Xue, Nan and Xia, Gui-Song and Bai, Xiang and Yang, Wen and Yang, Michael and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          title={Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges},
          year={2022},
          volume={44},
          number={11},
          pages={7778-7796},
          doi={10.1109/TPAMI.2021.3117983}
        }
        ```

We thank the team behind the DOTA datasets for curating them. For full details on the dataset, visit the [official DOTA website](https://captain-whu.github.io/DOTA/index.html).

## FAQ

### What is the DOTA8 dataset and how can it be used?

The DOTA8 dataset is a small oriented object detection dataset made up of the first 8 images from the DOTAv1 split set, with 4 images designated for training and 4 for validation. It's ideal for testing and debugging object detection models like Ultralytics YOLO26. Due to its manageable size and diversity, it helps in identifying pipeline errors and running sanity checks before training on larger datasets. Learn more about [OBB detection](../../tasks/obb.md) with Ultralytics YOLO26.

### How do I train a YOLO26 model using the DOTA8 dataset?

To train a YOLO26n-obb model on the DOTA8 dataset for 100 epochs with an image size of 640, you can use the following code snippets. For comprehensive argument options, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-obb.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="dota8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo obb train data=dota8.yaml model=yolo26n-obb.pt epochs=100 imgsz=640
        ```

### What are the key features of the DOTA dataset and where can I access the YAML file?

The DOTA dataset is known for its large-scale benchmark and the challenges it presents for object detection in aerial images. The DOTA8 subset is a smaller, manageable dataset ideal for initial tests. The [`dota8.yaml` configuration file](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dota8.yaml), which contains paths, classes, and configuration details, is maintained in the Ultralytics GitHub repository.

### How does mosaicing enhance model training with the DOTA8 dataset?

Mosaicing combines multiple images into one during training, increasing the variety of objects and contexts within each batch. This improves a model's ability to generalize to different object sizes, aspect ratios, and scenes. The [Sample Images and Annotations](#sample-images-and-annotations) section shows a mosaiced DOTA8 training batch. Learn more about mosaicing and other augmentations in the [YOLO data augmentation guide](../../guides/yolo-data-augmentation.md).

### Why should I use Ultralytics YOLO26 for object detection tasks?

Ultralytics YOLO26 supports real-time object detection, including oriented bounding boxes (OBB), alongside other tasks such as [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation), all through one training pipeline. Its pretrained models make [fine-tuning](https://www.ultralytics.com/glossary/fine-tuning) efficient. Learn more in the [Ultralytics YOLO26 documentation](../../models/yolo26.md).
