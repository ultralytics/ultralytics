---
comments: true
description: A complete guide to the PASCAL VOC dataset used for object detection, segmentation and classification tasks with relevance to YOLO model training.
keywords: Ultralytics, PASCAL VOC dataset, object detection, segmentation, image classification, YOLO, model training, VOC.yaml, deep learning
---

# VOC Dataset

The [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) (Visual Object Classes) dataset is a well-known object detection, segmentation, and classification dataset. It is designed to encourage research on a wide variety of object categories and is commonly used for benchmarking computer vision models. It is an essential dataset for researchers and developers working on object detection, segmentation, and classification tasks.

## Key Features

- VOC dataset includes two main challenges: VOC2007 and VOC2012.
- The dataset comprises 20 object categories, including common objects like cars, bicycles, and animals, as well as more specific categories such as boats, sofas, and dining tables.
- Annotations include object bounding boxes and class labels for object detection and classification tasks, and segmentation masks for the segmentation tasks.
- VOC provides standardized evaluation metrics like mean Average Precision (mAP) for object detection and classification, making it suitable for comparing model performance.

## Dataset Structure

The VOC dataset is split into three subsets:

1. **Train**: This subset contains images for training object detection, segmentation, and classification models.
2. **Validation**: This subset has images used for validation purposes during model training.
3. **Test**: This subset consists of images used for testing and benchmarking the trained models. Ground truth annotations for this subset are not publicly available, and the results are submitted to the [PASCAL VOC evaluation server](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php) for performance evaluation.

## Applications

The VOC dataset is widely used for training and evaluating deep learning models in object detection (such as YOLO, Faster R-CNN, and SSD), instance segmentation (such as Mask R-CNN), and image classification. The dataset's diverse set of object categories, large number of annotated images, and standardized evaluation metrics make it an essential resource for computer vision researchers and practitioners.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the VOC dataset, the `VOC.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VOC.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VOC.yaml).

!!! Example "ultralytics/cfg/datasets/VOC.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/VOC.yaml"
    ```

## Usage

To train a YOLOv8n model on the VOC dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! Example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data='VOC.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from
        a pretrained *.pt model
        yolo detect train data=VOC.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

The VOC dataset contains a diverse set of images with various object categories and complex scenes. Here are some examples of images from the dataset, along with their corresponding annotations:

![Dataset sample image](https://github.com/ultralytics/ultralytics/assets/26833433/7d4c18f4-774e-43f8-a5f3-9467cda7de4a)

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the VOC dataset and the benefits of using mosaicing during the training process.

## Citations and Acknowledgments

If you use the VOC dataset in your research or development work, please cite the following paper:

!!! Note ""

    === "BibTeX"

        ```bibtex
        @misc{everingham2010pascal,
              title={The PASCAL Visual Object Classes (VOC) Challenge},
              author={Mark Everingham and Luc Van Gool and Christopher K. I. Williams and John Winn and Andrew Zisserman},
              year={2010},
              eprint={0909.5206},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

We would like to acknowledge the PASCAL VOC Consortium for creating and maintaining this valuable resource for the computer vision community. For more information about the VOC dataset and its creators, visit the [PASCAL VOC dataset website](http://host.robots.ox.ac.uk/pascal/VOC/).
