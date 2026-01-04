---
comments: true
description: Discover the PASCAL VOC dataset, essential for object detection, segmentation, and classification. Learn key features, applications, and usage tips.
keywords: PASCAL VOC, VOC dataset, object detection, segmentation, classification, YOLO, Faster R-CNN, Mask R-CNN, image annotations, computer vision
---

# VOC Dataset

The [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) (Visual Object Classes) dataset is a well-known object detection, segmentation, and classification dataset. It is designed to encourage research on a wide variety of object categories and is commonly used for benchmarking computer vision models. It is an essential dataset for researchers and developers working on object detection, segmentation, and classification tasks.

## Key Features

- VOC dataset includes two main challenges: VOC2007 and VOC2012.
- The dataset comprises 20 object categories, including common objects like cars, bicycles, and animals, as well as more specific categories such as boats, sofas, and dining tables.
- Annotations include object bounding boxes and class labels for object detection and classification tasks, and segmentation masks for the segmentation tasks.
- VOC provides standardized evaluation metrics like [mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP) for object detection and classification, making it suitable for comparing model performance.

## Dataset Structure

The VOC dataset is split into three subsets:

1. **Train**: This subset contains images for training object detection, segmentation, and classification models.
2. **Validation**: This subset has images used for validation purposes during model training.
3. **Test**: This subset consists of images used for testing and benchmarking the trained models. Ground truth annotations for this subset are not publicly available, and the results are submitted to the [PASCAL VOC evaluation server](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php) for performance evaluation.

## Applications

The VOC dataset is widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in object detection (such as [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo11/), [Faster R-CNN](https://arxiv.org/abs/1506.01497), and [SSD](https://arxiv.org/abs/1512.02325)), [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) (such as [Mask R-CNN](https://arxiv.org/abs/1703.06870)), and [image classification](https://www.ultralytics.com/glossary/image-classification). The dataset's diverse set of object categories, large number of annotated images, and standardized evaluation metrics make it an essential resource for [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) researchers and practitioners.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the VOC dataset, the `VOC.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VOC.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VOC.yaml).

!!! example "ultralytics/cfg/datasets/VOC.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/VOC.yaml"
    ```

## Usage

To train a YOLO11n model on the VOC dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="VOC.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=VOC.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

The VOC dataset contains a diverse set of images with various object categories and complex scenes. Here are some examples of images from the dataset, along with their corresponding annotations:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/mosaiced-voc-dataset-sample.avif)

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the VOC dataset and the benefits of using mosaicing during the training process.

## Citations and Acknowledgments

If you use the VOC dataset in your research or development work, please cite the following paper:

!!! quote ""

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

We would like to acknowledge the PASCAL VOC Consortium for creating and maintaining this valuable resource for the [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) community. For more information about the VOC dataset and its creators, visit the [PASCAL VOC dataset website](http://host.robots.ox.ac.uk/pascal/VOC/).

## FAQ

### What is the PASCAL VOC dataset and why is it important for computer vision tasks?

The [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) (Visual Object Classes) dataset is a renowned benchmark for [object detection](https://www.ultralytics.com/glossary/object-detection), segmentation, and classification in computer vision. It includes comprehensive annotations like bounding boxes, class labels, and segmentation masks across 20 different object categories. Researchers use it widely to evaluate the performance of models like Faster R-CNN, YOLO, and Mask R-CNN due to its standardized evaluation metrics such as mean Average Precision (mAP).

### How do I train a YOLO11 model using the VOC dataset?

To train a YOLO11 model with the VOC dataset, you need the dataset configuration in a YAML file. Here's an example to start training a YOLO11n model for 100 epochs with an image size of 640:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="VOC.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=VOC.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

### What are the primary challenges included in the VOC dataset?

The VOC dataset includes two main challenges: VOC2007 and VOC2012. These challenges test object detection, segmentation, and classification across 20 diverse object categories. Each image is meticulously annotated with bounding boxes, class labels, and segmentation masks. The challenges provide standardized metrics like mAP, facilitating the comparison and benchmarking of different computer vision models.

### How does the PASCAL VOC dataset enhance model benchmarking and evaluation?

The PASCAL VOC dataset enhances model benchmarking and evaluation through its detailed annotations and standardized metrics like mean Average [Precision](https://www.ultralytics.com/glossary/precision) (mAP). These metrics are crucial for assessing the performance of object detection and classification models. The dataset's diverse and complex images ensure comprehensive model evaluation across various real-world scenarios.

### How do I use the VOC dataset for [semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation) in YOLO models?

To use the VOC dataset for semantic segmentation tasks with YOLO models, you need to configure the dataset properly in a YAML file. The YAML file defines paths and classes needed for training segmentation models. Check the VOC dataset YAML configuration file at [VOC.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VOC.yaml) for detailed setups. For segmentation tasks, you would use a segmentation-specific model like `yolo11n-seg.pt` instead of the detection model.
