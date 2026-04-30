---
comments: true
description: Explore the COCO-Seg dataset, an extension of COCO, with detailed segmentation annotations. Learn how to train YOLO models with COCO-Seg.
keywords: COCO-Seg, dataset, YOLO models, instance segmentation, object detection, COCO dataset, YOLO26, computer vision, Ultralytics, machine learning
---

# COCO-Seg Dataset

The [COCO-Seg](https://cocodataset.org/#home) dataset, an extension of the COCO (Common Objects in Context) dataset, is specially designed to aid research in object [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation). It uses the same images as COCO but introduces more detailed segmentation annotations. This dataset is a crucial resource for researchers and developers working on instance segmentation tasks, especially for training [Ultralytics YOLO](https://docs.ultralytics.com/models/) models.

## COCO-Seg Pretrained Models

{% include "macros/yolo-seg-perf.md" %}

## Key Features

- COCO-Seg retains the original 330K images from COCO.
- The dataset consists of the same 80 object categories found in the original COCO dataset.
- Annotations now include more detailed instance segmentation masks for each object in the images.
- COCO-Seg provides standardized evaluation metrics like [mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP) for object detection, and mean Average [Recall](https://www.ultralytics.com/glossary/recall) (mAR) for instance segmentation tasks, enabling effective comparison of model performance.

## Dataset Structure

The COCO-Seg dataset is partitioned into three subsets:

1. **Train2017**: 118K images for training instance segmentation models.
2. **Val2017**: 5K images used for validation during model development.
3. **Test2017**: 20K images used for benchmarking. Ground-truth annotations for this subset are not publicly available, so predictions must be submitted to the [COCO evaluation server](https://codalab.lisn.upsaclay.fr/competitions/7383) for scoring.

## Applications

COCO-Seg is widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in instance segmentation, such as the YOLO models. The large number of annotated images, the diversity of object categories, and the standardized evaluation metrics make it an indispensable resource for [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) researchers and practitioners.

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

COCO-Seg, like its predecessor COCO, contains a diverse set of images with various object categories and complex scenes. However, COCO-Seg introduces more detailed instance segmentation masks for each object in the images. Here are some examples of images from the dataset, along with their corresponding instance segmentation masks:

![COCO segmentation dataset mosaic training batch](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-training-batch-3.avif)

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. [Mosaicing](https://docs.ultralytics.com/guides/hyperparameter-tuning/) is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This aids the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the COCO-Seg dataset and the benefits of using mosaicing during the training process.

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

The [COCO-Seg](https://cocodataset.org/#home) dataset is an extension of the original COCO (Common Objects in Context) dataset, specifically designed for instance segmentation tasks. While it uses the same images as the COCO dataset, COCO-Seg includes more detailed segmentation annotations, making it a powerful resource for researchers and developers focusing on [object instance segmentation](https://docs.ultralytics.com/tasks/segment/).

### How can I train a YOLO26 model using the COCO-Seg dataset?

To train a YOLO26n-seg model on the COCO-Seg dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a detailed list of available arguments, refer to the model [Training](../../modes/train.md) page.

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

- Retains the original 330K images from the COCO dataset.
- Annotates the same 80 object categories found in the original COCO.
- Provides more detailed instance segmentation masks for each object.
- Uses standardized evaluation metrics such as mean Average [Precision](https://www.ultralytics.com/glossary/precision) (mAP) for [object detection](https://www.ultralytics.com/glossary/object-detection) and mean Average Recall (mAR) for instance segmentation tasks.

### What pretrained models are available for COCO-Seg, and what are their performance metrics?

The COCO-Seg dataset supports multiple pretrained YOLO26 segmentation models with varying performance metrics. Here's a summary of the available models and their key metrics:

{% include "macros/yolo-seg-perf.md" %}

These models range from the lightweight YOLO26n-seg to the more powerful YOLO26x-seg, offering different trade-offs between speed and accuracy to suit various application requirements. For more information on model selection, visit the [Ultralytics models page](https://docs.ultralytics.com/models/).

### How is the COCO-Seg dataset structured and what subsets does it contain?

The COCO-Seg dataset is partitioned into three subsets for specific training and evaluation needs:

1. **Train2017**: Contains 118K images used primarily for training instance segmentation models.
2. **Val2017**: Comprises 5K images utilized for validation during the training process.
3. **Test2017**: Encompasses 20K images reserved for testing and benchmarking trained models. Note that ground truth annotations for this subset are not publicly available, and performance results are submitted to the [COCO evaluation server](https://codalab.lisn.upsaclay.fr/competitions/7383) for assessment.

For smaller experimentation needs, you might also consider using the [COCO8-seg dataset](https://docs.ultralytics.com/datasets/segment/coco8-seg/), which is a compact version containing just 8 images from the COCO train 2017 set.
