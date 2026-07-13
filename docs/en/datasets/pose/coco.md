---
title: COCO-Pose Estimation Dataset
comments: true
description: "Explore the Ultralytics COCO-Pose dataset: 58,945 images with 156K+ annotated people and a 17-keypoint schema, for training YOLO26 pose estimation models."
keywords: COCO-Pose, pose estimation, dataset, keypoints, COCO Keypoints 2017, YOLO, deep learning, computer vision
---

# COCO-Pose Dataset

The [COCO-Pose](https://cocodataset.org/#keypoints-2017) dataset adapts COCO (Common Objects in Context) for [pose estimation](../../tasks/pose.md): 58,945 images from COCO Keypoints 2017, annotated with 156,165 people using a 17-keypoint schema. It is the standard set for training and benchmarking keypoint models such as [Ultralytics YOLO26](../../models/yolo26.md), and the 8-image [COCO8-Pose](coco8-pose.md) subset mirrors its format for quick sanity checks.

![COCO pose estimation with human keypoints](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/pose-sample-image.avif)

## COCO-Pose Pretrained Models

{% include "macros/yolo-pose-perf.md" %}

## Key Features

- COCO-Pose builds upon the [COCO Keypoints 2017](http://presentations.cocodataset.org/COCO17-Keypoints-Overview.pdf) challenge, which labels 1,710,498 individual keypoints across 156,165 annotated people.
- Each person annotation uses 17 keypoint types — nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles — stored as `(x, y, visibility)` triplets.
- Like COCO, it provides standardized evaluation metrics, including Object Keypoint Similarity (OKS) for pose estimation tasks, making it suitable for comparing model performance.
- **Download size**: ~27 GB on first use. The `coco-pose.yaml` header lists 20.1 GB (`train2017.zip` + `val2017.zip` only), but the download script also fetches the 7 GB `test2017.zip` unconditionally, even though that archive is needed only for the optional test-dev2017 submission split.

## Dataset Structure

For training and validation, COCO-Pose includes only COCO 2017 images with keypoint-annotated people, so its labeled splits are smaller than full COCO's. Its YAML defines three subsets:

1. **Train2017**: This subset contains 56,599 images from the COCO dataset, annotated for training pose estimation models.
2. **Val2017**: This subset has 2,346 images used for validation purposes during model training.
3. **Test-dev2017**: A 20,288-image subset of the full 40,670-image test2017 set with withheld ground truth. The dataset YAML links this split to the [COCO test-dev keypoints evaluation server](https://codalab.lisn.upsaclay.fr/competitions/7403).

Training at this scale is where [Ultralytics Platform](https://platform.ultralytics.com/) helps most — it manages the compute so you can launch and monitor runs without provisioning your own GPUs.

## Applications

The COCO-Pose dataset is specifically used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models on keypoint detection and [pose estimation](../../tasks/pose.md). The dataset's large number of annotated images and standardized evaluation metrics make it an essential resource for [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) researchers and practitioners working on human pose.

## Dataset YAML

A YAML file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the COCO-Pose dataset, the `coco-pose.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml).

!!! example "ultralytics/cfg/datasets/coco-pose.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco-pose.yaml"
    ```

## Usage

To train a YOLO26n-pose model on the COCO-Pose dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo pose train data=coco-pose.yaml model=yolo26n-pose.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

The COCO-Pose dataset contains a diverse set of images with human figures annotated with keypoints. Here are some examples of images from the dataset, along with their corresponding annotations:

![COCO pose estimation dataset mosaic training batch](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-training-batch-6.avif)

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

COCO-Pose supplies the COCO Keypoints 2017 images and annotations converted to YOLO keypoint format, using a 17-keypoint schema across 58,945 images. Point any Ultralytics YOLO pose model at it with `data=coco-pose.yaml`, and the [Training](../../modes/train.md) page documents every argument you can tune from there.

### How can I train a YOLO26 model on the COCO-Pose dataset?

Load `yolo26n-pose.pt` and call `model.train(data="coco-pose.yaml", epochs=100, imgsz=640)` — see the [Train Example](#usage) above for the full Python and CLI snippets, and the [training page](../../modes/train.md) for a comprehensive list of arguments.

### What are the different metrics provided by the COCO-Pose dataset for evaluating model performance?

The COCO-Pose dataset provides several standardized evaluation metrics for pose estimation tasks, similar to the original COCO dataset. Key metrics include the Object Keypoint Similarity (OKS), which evaluates the [accuracy](https://www.ultralytics.com/glossary/accuracy) of predicted keypoints against ground truth annotations. These metrics allow for thorough performance comparisons between different models. For instance, the COCO-Pose pretrained models such as YOLO26n-pose, YOLO26s-pose, and others have specific performance metrics listed in the documentation, like mAP<sup>pose</sup>50-95 and mAP<sup>pose</sup>50.

### How is the dataset structured and split for the COCO-Pose dataset?

COCO-Pose ships two labeled splits: 56,599 train2017 images and 2,346 val2017 images. A third split, test-dev2017 (20,288 of the full 40,670 test2017 images), keeps its ground truth private; the dataset YAML links it to the [COCO test-dev keypoints evaluation server](https://codalab.lisn.upsaclay.fr/competitions/7403). See the [Dataset Structure](#dataset-structure) section, or the `coco-pose.yaml` file on [GitHub](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) for the exact split paths.

### What are the key features and applications of the COCO-Pose dataset?

COCO-Pose uses 17 human keypoint types and inherits COCO's standardized metrics, including Object Keypoint Similarity (OKS), for comparing models. That combination suits human pose applications such as sports analytics, healthcare, and human-computer interaction. Pretrained YOLO26-pose weights are listed under [COCO-Pose Pretrained Models](#coco-pose-pretrained-models).

For more on keypoint models, see the [Pose Estimation](../../tasks/pose.md) task docs.
