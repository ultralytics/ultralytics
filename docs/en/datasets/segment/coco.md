---
title: COCO Segmentation Dataset
comments: true
description: Explore the COCO-Seg dataset, an extension of COCO with 123,287 segmentation-labeled images across 80 classes. Learn how to train YOLO models with COCO-Seg.
keywords: COCO-Seg, dataset, YOLO models, instance segmentation, object detection, COCO dataset, YOLO26, computer vision, Ultralytics, machine learning
---

# COCO-Seg Dataset

The [COCO-Seg](https://cocodataset.org/#home) dataset provides [COCO](https://cocodataset.org/#home) (Common Objects in Context) instance segmentation masks — 118,287 training and 5,000 validation images with polygon masks across 80 object categories — in the [Ultralytics YOLO](../../models/index.md) label format. It uses COCO's original images and native segmentation annotations, converted for YOLO training, making it a crucial resource for researchers and developers working on [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) tasks.

## COCO-Seg Pretrained Models

{% include "macros/yolo-seg-perf.md" %}

## Key Features

- COCO-Seg provides instance segmentation masks for 123,287 labeled COCO train2017/val2017 images (118,287 train + 5,000 val), out of COCO's broader ~330K-image release.
- The dataset consists of the same 80 object categories found in the original COCO dataset.
- Annotations provide instance segmentation masks in the YOLO polygon label format.
- COCO-Seg provides standardized mAP and mAR metrics for evaluating instance segmentation performance, enabling effective comparison of model performance.
- **Download size**: ~27 GB on first use. The `coco.yaml` header lists 20.1 GB (`train2017.zip` + `val2017.zip` only), but the download script also fetches the 7 GB `test2017.zip` unconditionally, even though that archive is needed only for the optional test-dev2017 submission split.

## Dataset Structure

The COCO-Seg dataset is partitioned into three subsets:

1. **Train2017**: 118,287 images for training instance segmentation models.
2. **Val2017**: 5,000 images used for validation during model development.
3. **Test-dev2017**: 20,288 of the 40,670 test2017 images, used for benchmarking. Ground-truth annotations for this subset are not publicly available, so predictions must be submitted to the [COCO evaluation server](https://cocodataset.org/#upload) for scoring.

For smaller experimentation needs, see the [COCO128-Seg](coco128-seg.md) (128 images) and [COCO8-Seg](coco8-seg.md) (8 images) subsets.

## Applications

COCO-Seg is widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models on [instance segmentation](../../tasks/segment.md), such as the YOLO models. The large number of annotated images, the diversity of object categories, and the standardized evaluation metrics make it an indispensable resource for [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) researchers and practitioners. Full COCO-Seg annotations can also be browsed and managed on [Ultralytics Platform](https://platform.ultralytics.com/).

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the COCO-Seg dataset, the `coco.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

!!! example "ultralytics/cfg/datasets/coco.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco.yaml"
    ```

## Usage

To train a YOLO26n-seg model on the COCO-Seg dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-seg.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo segment train data=coco.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

COCO-Seg contains the same diverse images, object categories, and complex scenes as COCO, with instance segmentation masks provided in the YOLO label format. Here are some examples of images from the dataset, along with their corresponding instance segmentation masks:

![COCO segmentation dataset mosaic training batch](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-training-batch-3.avif)

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. [Mosaicing](../../guides/hyperparameter-tuning.md) is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This aids the model's ability to generalize to different object sizes, aspect ratios, and contexts.

## Citations and Acknowledgments

If you use the COCO-Seg dataset in your research or development work, please cite the original COCO paper and acknowledge the extension to COCO-Seg:

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

We extend our thanks to the COCO Consortium for creating and maintaining this invaluable resource for the computer vision community. For more information about the COCO dataset and its creators, visit the [COCO dataset website](https://cocodataset.org/#home).

## FAQ

### What is the COCO-Seg dataset and how does it differ from the original COCO dataset?

[COCO-Seg](https://cocodataset.org/#home) is the [Ultralytics YOLO](../../models/index.md)-format packaging of COCO's (Common Objects in Context) native instance segmentation masks for the same 118,287 train2017 and 5,000 val2017 images. The original COCO annotations already include these polygon masks for all 80 object categories; COCO-Seg converts them to the YOLO label format used for [object instance segmentation](../../tasks/segment.md) training.

### How can I train a YOLO26 model using the COCO-Seg dataset?

To train a YOLO26n-seg model on the COCO-Seg dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a detailed list of available training arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-seg.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo segment train data=coco.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
        ```

### What are the key features of the COCO-Seg dataset?

The COCO-Seg dataset includes several key features:

- Provides instance segmentation masks for 123,287 labeled COCO train2017/val2017 images (118,287 train + 5,000 val).
- Annotates the same 80 object categories found in the original COCO.
- Provides instance segmentation masks in the YOLO polygon label format.
- Uses standardized evaluation metrics such as mean Average [Precision](https://www.ultralytics.com/glossary/precision) (mAP) and mean Average Recall (mAR) for [instance segmentation](../../tasks/segment.md) tasks.

### What pretrained models are available for COCO-Seg, and what are their performance metrics?

The COCO-Seg dataset supports multiple pretrained YOLO26 segmentation models with varying performance metrics. Here's a summary of the available models and their key metrics:

{% include "macros/yolo-seg-perf.md" %}

These models range from the lightweight YOLO26n-seg to the more powerful YOLO26x-seg, offering different trade-offs between speed and accuracy to suit various application requirements. For more information on model selection, visit the [Ultralytics models page](../../models/index.md).

### How is the COCO-Seg dataset structured and what subsets does it contain?

The COCO-Seg dataset is partitioned into three subsets for specific training and evaluation needs:

1. **Train2017**: Contains 118,287 images used primarily for training instance segmentation models.
2. **Val2017**: Comprises 5,000 images utilized for validation during the training process.
3. **Test-dev2017**: Encompasses 20,288 of the 40,670 test2017 images reserved for testing and benchmarking trained models. Note that ground truth annotations for this subset are not publicly available, and performance results are submitted to the [COCO evaluation server](https://cocodataset.org/#upload) for assessment.

For smaller experimentation needs, you might also consider the [COCO128-Seg dataset](coco128-seg.md) (128 images) or the [COCO8-Seg dataset](coco8-seg.md), a compact version containing just 8 images from the COCO train 2017 set.
