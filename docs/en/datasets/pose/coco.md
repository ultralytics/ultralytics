---
comments: true
description: Explore the COCO-Pose dataset for advanced pose estimation. Learn about datasets, pretrained models, metrics, and applications for training with YOLO.
keywords: COCO-Pose, pose estimation, dataset, keypoints, COCO Keypoints 2017, YOLO, deep learning, computer vision
---

# COCO-Pose Dataset

The [COCO-Pose](https://cocodataset.org/#keypoints-2017) dataset is a specialized version of the COCO (Common Objects in Context) dataset, designed for pose estimation tasks. It leverages the COCO Keypoints 2017 images and labels to enable the training of models like YOLO for pose estimation tasks.

![Pose sample image](https://github.com/ultralytics/docs/releases/download/0/pose-sample-image.avif)

## COCO-Pose Pretrained Models

{% include "macros/yolo-pose-perf.md" %}

## Key Features

- COCO-Pose builds upon the COCO Keypoints 2017 dataset which contains 200K images labeled with keypoints for pose estimation tasks.
- The dataset supports 17 keypoints for human figures, facilitating detailed pose estimation.
- Like COCO, it provides standardized evaluation metrics, including Object Keypoint Similarity (OKS) for pose estimation tasks, making it suitable for comparing model performance.

## Dataset Structure

The COCO-Pose dataset is split into three subsets:

1. **Train2017**: This subset contains 56599 images from the COCO dataset, annotated for training pose estimation models.
2. **Val2017**: This subset has 2346 images used for validation purposes during model training.
3. **Test2017**: This subset consists of images used for testing and benchmarking the trained models. Ground truth annotations for this subset are not publicly available, and the results are submitted to the [COCO evaluation server](https://codalab.lisn.upsaclay.fr/competitions/7384) for performance evaluation.

## Applications

The COCO-Pose dataset is specifically used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in keypoint detection and pose estimation tasks, such as OpenPose. The dataset's large number of annotated images and standardized evaluation metrics make it an essential resource for [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) researchers and practitioners focused on pose estimation.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the COCO-Pose dataset, the `coco-pose.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml).

!!! example "ultralytics/cfg/datasets/coco-pose.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco-pose.yaml"
    ```

## Usage

To train a YOLO11n-pose model on the COCO-Pose dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo pose train data=coco-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

The COCO-Pose dataset contains a diverse set of images with human figures annotated with keypoints. Here are some examples of images from the dataset, along with their corresponding annotations:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/mosaiced-training-batch-6.avif)

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the COCO-Pose dataset and the benefits of using mosaicing during the training process.

## Citations and Acknowledgments

If you use the COCO-Pose dataset in your research or development work, please cite the following paper:

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

We would like to acknowledge the COCO Consortium for creating and maintaining this valuable resource for the computer vision community. For more information about the COCO-Pose dataset and its creators, visit the [COCO dataset website](https://cocodataset.org/#home).

## FAQ

### What is the COCO-Pose dataset and how is it used with Ultralytics YOLO for pose estimation?

The [COCO-Pose](https://cocodataset.org/#keypoints-2017) dataset is a specialized version of the COCO (Common Objects in Context) dataset designed for pose estimation tasks. It builds upon the COCO Keypoints 2017 images and annotations, allowing for the training of models like Ultralytics YOLO for detailed pose estimation. For instance, you can use the COCO-Pose dataset to train a YOLO11n-pose model by loading a pretrained model and training it with a YAML configuration. For training examples, refer to the [Training](../../modes/train.md) documentation.

### How can I train a YOLO11 model on the COCO-Pose dataset?

Training a YOLO11 model on the COCO-Pose dataset can be accomplished using either Python or CLI commands. For example, to train a YOLO11n-pose model for 100 epochs with an image size of 640, you can follow the steps below:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo pose train data=coco-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
        ```

For more details on the training process and available arguments, check the [training page](../../modes/train.md).

### What are the different metrics provided by the COCO-Pose dataset for evaluating model performance?

The COCO-Pose dataset provides several standardized evaluation metrics for pose estimation tasks, similar to the original COCO dataset. Key metrics include the Object Keypoint Similarity (OKS), which evaluates the [accuracy](https://www.ultralytics.com/glossary/accuracy) of predicted keypoints against ground truth annotations. These metrics allow for thorough performance comparisons between different models. For instance, the COCO-Pose pretrained models such as YOLO11n-pose, YOLO11s-pose, and others have specific performance metrics listed in the documentation, like mAP<sup>pose</sup>50-95 and mAP<sup>pose</sup>50.

### How is the dataset structured and split for the COCO-Pose dataset?

The COCO-Pose dataset is split into three subsets:

1. **Train2017**: Contains 56599 COCO images, annotated for training pose estimation models.
2. **Val2017**: 2346 images for validation purposes during model training.
3. **Test2017**: Images used for testing and benchmarking trained models. Ground truth annotations for this subset are not publicly available; results are submitted to the [COCO evaluation server](https://codalab.lisn.upsaclay.fr/competitions/7403) for performance evaluation.

These subsets help organize the training, validation, and testing phases effectively. For configuration details, explore the `coco-pose.yaml` file available on [GitHub](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml).

### What are the key features and applications of the COCO-Pose dataset?

The COCO-Pose dataset extends the COCO Keypoints 2017 annotations to include 17 keypoints for human figures, enabling detailed pose estimation. Standardized evaluation metrics (e.g., OKS) facilitate comparisons across different models. Applications of the COCO-Pose dataset span various domains, such as sports analytics, healthcare, and human-computer interaction, wherever detailed pose estimation of human figures is required. For practical use, leveraging pretrained models like those provided in the documentation (e.g., YOLO11n-pose) can significantly streamline the process ([Key Features](#key-features)).

If you use the COCO-Pose dataset in your research or development work, please cite the paper with the following [BibTeX entry](#citations-and-acknowledgments).
