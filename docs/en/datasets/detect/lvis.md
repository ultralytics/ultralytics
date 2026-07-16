---
title: LVIS Detection Dataset
comments: true
creator:
    name: LVIS Consortium
    url: https://www.lvisdataset.org/
license:
    name: CC-BY-4.0
    url: https://www.lvisdataset.org/dataset
description: LVIS is a large-vocabulary object detection and instance segmentation dataset with 1,203 classes over ~160K COCO images. Train Ultralytics YOLO on LVIS.
keywords: LVIS dataset, object detection, instance segmentation, large vocabulary, Facebook AI Research, YOLO26, computer vision, model training, rare categories
---

# LVIS Dataset

The [LVIS dataset](https://www.lvisdataset.org/) is a large-scale [object detection](../../tasks/detect.md) and [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) benchmark with 1,203 object categories across roughly 160,000 images. Developed and released by Facebook AI Research (FAIR), it reuses the [COCO](./coco.md) images but adds a much larger, long-tailed vocabulary — from common objects like cars and bicycles to rarer ones such as handbags, umbrellas, and sports equipment — to push progress on recognizing infrequent categories.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/cfTKj96TjSE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> YOLO World training workflow with LVIS dataset
</p>

<p align="center">
    <img width="640" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/lvis-dataset-example-images.avif" alt="LVIS large vocabulary instance segmentation dataset">
</p>

## Key Features

- LVIS spans roughly 160,000 images with about 2 million instance annotations for object detection and instance segmentation.
- The dataset defines 1,203 object categories, including common objects like cars, bicycles, and animals as well as fine-grained categories such as umbrellas, handbags, and sports equipment.
- Annotations include object bounding boxes and segmentation masks, with a strong focus on rare, long-tailed categories.
- LVIS provides standardized evaluation metrics like [mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP) for detection and mean Average [Recall](https://www.ultralytics.com/glossary/recall) (mAR) for segmentation, making model comparison straightforward.
- LVIS uses the same images as the [COCO](./coco.md) dataset but with different splits and far more detailed annotations.

## Dataset Structure

The Ultralytics `lvis.yaml` configuration defines three splits:

| Split      | Images  | Description                                                |
| ---------- | ------- | ---------------------------------------------------------- |
| Train      | 100,170 | Images for training detection and segmentation models      |
| Validation | 19,809  | Held-out images for evaluation during training             |
| Minival    | 5,000   | Fast-validation subset identical to the COCO `val2017` set |

The upstream LVIS benchmark also includes a held-out test set of roughly 20,000 images whose ground-truth annotations are not public; results are submitted to the [LVIS evaluation server](https://eval.ai/web/challenges/challenge-page/675/overview) for scoring.

## Applications

The LVIS dataset is widely used to train and evaluate [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models for object detection (such as [YOLO](../../models/yolo26.md), [Faster R-CNN](https://arxiv.org/abs/1506.01497), and [SSD](https://arxiv.org/abs/1512.02325)) and instance segmentation (such as [Mask R-CNN](https://arxiv.org/abs/1703.06870)). Its large, long-tailed vocabulary, high annotation volume, and standardized evaluation metrics make it an essential benchmark for measuring how well [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models handle rare categories.

To label your own images, train, and manage large-vocabulary datasets like LVIS in your browser, run the full workflow with [Ultralytics Platform](https://platform.ultralytics.com/).

## Dataset YAML

The `lvis.yaml` file defines the dataset configuration — the dataset paths, class names, and other metadata. It is maintained in the Ultralytics repository at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml).

!!! example "ultralytics/cfg/datasets/lvis.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/lvis.yaml"
    ```

## Usage

To train a YOLO26n model on the LVIS dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. The dataset (20.7 GB) downloads automatically on first use. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="lvis.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=lvis.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

The LVIS dataset contains diverse images with many object categories in complex scenes. Below is an example of a mosaiced training batch:

![LVIS large vocabulary instance segmentation dataset mosaic](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/lvis-mosaiced-training-batch.avif)

- **Mosaiced Image**: This training batch combines multiple LVIS images into one through mosaicing, a technique that increases the variety of objects and scenes in each batch and helps the model generalize to different object sizes, aspect ratios, and contexts.

## Citations and Acknowledgments

If you use the LVIS dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{gupta2019lvis,
          title={LVIS: A Dataset for Large Vocabulary Instance Segmentation},
          author={Gupta, Agrim and Dollar, Piotr and Girshick, Ross},
          booktitle={Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition},
          year={2019}
        }
        ```

We would like to acknowledge the LVIS Consortium for creating and maintaining this valuable resource for the [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) community. For more information about the LVIS dataset and its creators, visit the [LVIS dataset website](https://www.lvisdataset.org/).

## FAQ

### What is the LVIS dataset used for?

The [LVIS dataset](https://www.lvisdataset.org/) is used to train and benchmark object detection and instance segmentation models on a large, long-tailed vocabulary. Developed by Facebook AI Research (FAIR), LVIS features 1,203 object categories and about 2 million instance annotations, which makes it an essential resource for measuring how well models like Ultralytics YOLO recognize both common and rare categories.

### How many images and classes are in the LVIS dataset?

The Ultralytics `lvis.yaml` configuration covers 1,203 object categories split into 100,170 training images, 19,809 validation images, and a 5,000-image minival subset that is identical to the COCO `val2017` set. The images are the same as those in COCO, but LVIS annotates them with a far larger vocabulary.

### How can I train a YOLO26n model using the LVIS dataset?

To train a YOLO26n model on the LVIS dataset for 100 epochs with an image size of 640, follow the example below.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="lvis.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=lvis.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

For detailed training configurations, refer to the [Training](../../modes/train.md) documentation.

### How does the LVIS dataset differ from the COCO dataset?

The images in the LVIS dataset are the same as those in the [COCO dataset](./coco.md), but the two differ in their splits and annotations. LVIS provides a larger and more detailed vocabulary with 1,203 object categories compared to COCO's 80 categories. LVIS also focuses on annotation completeness and diversity, aiming to push the limits of [object detection](https://www.ultralytics.com/glossary/object-detection) and instance segmentation models by offering more nuanced and comprehensive data.

### Does the LVIS dataset include a test set?

The Ultralytics `lvis.yaml` configuration provides train (100,170 images), validation (19,809 images), and minival (5,000 images) splits. The upstream LVIS benchmark also has a held-out test set of roughly 20,000 images whose ground-truth annotations are not public — results are submitted to the [LVIS evaluation server](https://eval.ai/web/challenges/challenge-page/675/overview) for scoring rather than evaluated locally.

### Why should I use Ultralytics YOLO for training on the LVIS dataset?

Ultralytics YOLO models, including the latest YOLO26, are optimized for real-time object detection with state-of-the-art [accuracy](https://www.ultralytics.com/glossary/accuracy) and speed. They support a wide range of annotations, such as the fine-grained ones provided by the LVIS dataset, making them ideal for advanced computer vision applications. Ultralytics also offers seamless integration with [training](../../modes/train.md), [validation](../../modes/val.md), and [prediction](../../modes/predict.md) modes, ensuring efficient model development and deployment.
