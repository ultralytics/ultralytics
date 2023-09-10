---
comments: true
description: Explore the SKU-110k dataset of densely packed retail shelf images for object detection research. Learn how to use it with Ultralytics.
keywords: SKU-110k dataset, object detection, retail shelf images, Ultralytics, YOLO, computer vision, deep learning models
---

# SKU-110k Dataset

The [SKU-110k](https://github.com/eg4000/SKU110K_CVPR19) dataset is a collection of densely packed retail shelf images, designed to support research in object detection tasks. Developed by Eran Goldman et al., the dataset contains over 110,000 unique store keeping unit (SKU) categories with densely packed objects, often looking similar or even identical, positioned in close proximity.

![Dataset sample image](https://github.com/eg4000/SKU110K_CVPR19/raw/master/figures/benchmarks_comparison.jpg)

## Key Features

- SKU-110k contains images of store shelves from around the world, featuring densely packed objects that pose challenges for state-of-the-art object detectors.
- The dataset includes over 110,000 unique SKU categories, providing a diverse range of object appearances.
- Annotations include bounding boxes for objects and SKU category labels.

## Dataset Structure

The SKU-110k dataset is organized into three main subsets:

1. **Training set**: This subset contains images and annotations used for training object detection models.
2. **Validation set**: This subset consists of images and annotations used for model validation during training.
3. **Test set**: This subset is designed for the final evaluation of trained object detection models.

## Applications

The SKU-110k dataset is widely used for training and evaluating deep learning models in object detection tasks, especially in densely packed scenes such as retail shelf displays. The dataset's diverse set of SKU categories and densely packed object arrangements make it a valuable resource for researchers and practitioners in the field of computer vision.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. For the case of the SKU-110K dataset, the `SKU-110K.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/SKU-110K.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/SKU-110K.yaml).

!!! example "ultralytics/cfg/datasets/SKU-110K.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/SKU-110K.yaml"
    ```

## Usage

To train a YOLOv8n model on the SKU-110K dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data='SKU-110K.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=SKU-110K.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

The SKU-110k dataset contains a diverse set of retail shelf images with densely packed objects, providing rich context for object detection tasks. Here are some examples of data from the dataset, along with their corresponding annotations:

![Dataset sample image](https://user-images.githubusercontent.com/26833433/238215979-1ab791c4-15d9-46f6-a5d6-0092c05dff7a.jpg)

- **Densely packed retail shelf image**: This image demonstrates an example of densely packed objects in a retail shelf setting. Objects are annotated with bounding boxes and SKU category labels.

The example showcases the variety and complexity of the data in the SKU-110k dataset and highlights the importance of high-quality data for object detection tasks.

## Citations and Acknowledgments

If you use the SKU-110k dataset in your research or development work, please cite the following paper:

!!! note ""

    === "BibTeX"

        ```bibtex
        @inproceedings{goldman2019dense,
         author    = {Eran Goldman and Roei Herzig and Aviv Eisenschtat and Jacob Goldberger and Tal Hassner},
         title     = {Precise Detection in Densely Packed Scenes},
         booktitle = {Proc. Conf. Comput. Vision Pattern Recognition (CVPR)},
         year      = {2019}
        }
        ```

We would like to acknowledge Eran Goldman et al. for creating and maintaining the SKU-110k dataset as a valuable resource for the computer vision research community. For more information about the SKU-110k dataset and its creators, visit the [SKU-110k dataset GitHub repository](https://github.com/eg4000/SKU110K_CVPR19).
