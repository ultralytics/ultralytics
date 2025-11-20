---
comments: true
description: Explore the COCO dataset for object detection and segmentation. Learn about its structure, usage, pretrained models, and key features.
keywords: COCO dataset, object detection, segmentation, benchmarking, computer vision, pose estimation, YOLO models, COCO annotations
---

# COCO Dataset

The [COCO](https://cocodataset.org/#home) (Common Objects in Context) dataset is a large-scale object detection, segmentation, and captioning dataset. It is designed to encourage research on a wide variety of object categories and is commonly used for benchmarking [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models. It is an essential dataset for researchers and developers working on object detection, segmentation, and pose estimation tasks.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/uDrn9QZJ2lk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics COCO Dataset Overview
</p>

## COCO Pretrained Models

{% include "macros/yolo-det-perf.md" %}

## Key Features

- COCO contains 330K images, with 200K images having annotations for object detection, segmentation, and captioning tasks.
- The dataset comprises 80 object categories, including common objects like cars, bicycles, and animals, as well as more specific categories such as umbrellas, handbags, and sports equipment.
- Annotations include object bounding boxes, segmentation masks, and captions for each image.
- COCO provides standardized evaluation metrics like [mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP) for object detection, and mean Average [Recall](https://www.ultralytics.com/glossary/recall) (mAR) for segmentation tasks, making it suitable for comparing model performance.

## Dataset Structure

The COCO dataset is split into three subsets:

1. **Train2017**: This subset contains 118K images for training object detection, segmentation, and captioning models.
2. **Val2017**: This subset has 5K images used for validation purposes during model training.
3. **Test2017**: This subset consists of 20K images used for testing and benchmarking the trained models. Ground truth annotations for this subset are not publicly available, and the results are submitted to the [COCO evaluation server](https://codalab.lisn.upsaclay.fr/competitions/7384) for performance evaluation.

## Applications

The COCO dataset is widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in object detection (such as [Ultralytics YOLO](../../models/yolo11.md), [Faster R-CNN](https://arxiv.org/abs/1506.01497), and [SSD](https://arxiv.org/abs/1512.02325)), [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) (such as [Mask R-CNN](https://arxiv.org/abs/1703.06870)), and keypoint detection (such as [OpenPose](https://arxiv.org/abs/1812.08008)). The dataset's diverse set of object categories, large number of annotated images, and standardized evaluation metrics make it an essential resource for computer vision researchers and practitioners.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the COCO dataset, the `coco.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

!!! example "ultralytics/cfg/datasets/coco.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco.yaml"
    ```

## Usage

To train a YOLO11n model on the COCO dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=coco.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

The COCO dataset contains a diverse set of images with various object categories and complex scenes. Here are some examples of images from the dataset, along with their corresponding annotations:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/mosaiced-coco-dataset-sample.avif)

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the COCO dataset and the benefits of using mosaicing during the training process.

## Citations and Acknowledgments

If you use the COCO dataset in your research or development work, please cite the following paper:

!!! quote ""

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

We would like to acknowledge the COCO Consortium for creating and maintaining this valuable resource for the computer vision community. For more information about the COCO dataset and its creators, visit the [COCO dataset website](https://cocodataset.org/#home).

## FAQ

### What is the COCO dataset and why is it important for computer vision?

The [COCO dataset](https://cocodataset.org/#home) (Common Objects in Context) is a large-scale dataset used for [object detection](https://www.ultralytics.com/glossary/object-detection), segmentation, and captioning. It contains 330K images with detailed annotations for 80 object categories, making it essential for benchmarking and training computer vision models. Researchers use COCO due to its diverse categories and standardized evaluation metrics like mean Average [Precision](https://www.ultralytics.com/glossary/precision) (mAP).

### How can I train a YOLO model using the COCO dataset?

To train a YOLO11 model using the COCO dataset, you can use the following code snippets:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=coco.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

Refer to the [Training page](../../modes/train.md) for more details on available arguments.

### What are the key features of the COCO dataset?

The COCO dataset includes:

- 330K images, with 200K annotated for object detection, segmentation, and captioning.
- 80 object categories ranging from common items like cars and animals to specific ones like handbags and sports equipment.
- Standardized evaluation metrics for object detection (mAP) and segmentation (mean Average Recall, mAR).
- **Mosaicing** technique in training batches to enhance model generalization across various object sizes and contexts.

### Where can I find pretrained YOLO11 models trained on the COCO dataset?

Pretrained YOLO11 models on the COCO dataset can be downloaded from the links provided in the documentation. Examples include:

- [YOLO11n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)
- [YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)
- [YOLO11m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt)
- [YOLO11l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt)
- [YOLO11x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt)

These models vary in size, mAP, and inference speed, providing options for different performance and resource requirements.

### How is the COCO dataset structured and how do I use it?

The COCO dataset is split into three subsets:

1. **Train2017**: 118K images for training.
2. **Val2017**: 5K images for validation during training.
3. **Test2017**: 20K images for benchmarking trained models. Results need to be submitted to the [COCO evaluation server](https://codalab.lisn.upsaclay.fr/competitions/7384) for performance evaluation.

The dataset's YAML configuration file is available at [coco.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), which defines paths, classes, and dataset details.
