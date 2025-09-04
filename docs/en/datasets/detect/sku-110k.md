---
comments: true
description: Explore the SKU-110k dataset of densely packed retail shelf images, perfect for training and evaluating deep learning models in object detection tasks.
keywords: SKU-110k, dataset, object detection, retail shelf images, deep learning, computer vision, model training
---

# SKU-110k Dataset

The [SKU-110k](https://github.com/eg4000/SKU110K_CVPR19) dataset is a collection of densely packed retail shelf images, designed to support research in [object detection](https://www.ultralytics.com/glossary/object-detection) tasks. Developed by Eran Goldman et al., the dataset contains over 110,000 unique store keeping unit (SKU) categories with densely packed objects, often looking similar or even identical, positioned in proximity.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/_gRqR-miFPE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train YOLOv10 on SKU-110k Dataset using Ultralytics | Retail Dataset
</p>

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/densely-packed-retail-shelf.avif)

## Key Features

- SKU-110k contains images of store shelves from around the world, featuring densely packed objects that pose challenges for state-of-the-art object detectors.
- The dataset includes over 110,000 unique SKU categories, providing a diverse range of object appearances.
- Annotations include bounding boxes for objects and SKU category labels.

## Dataset Structure

The SKU-110k dataset is organized into three main subsets:

1. **Training set**: This subset contains 8,219 images and annotations used for training object detection models.
2. **Validation set**: This subset consists of 588 images and annotations used for model validation during training.
3. **Test set**: This subset includes 2,936 images designed for the final evaluation of trained object detection models.

## Applications

The SKU-110k dataset is widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in object detection tasks, especially in densely packed scenes such as retail shelf displays. Its applications include:

- Retail inventory management and automation
- Product recognition in e-commerce platforms
- Planogram compliance verification
- Self-checkout systems in stores
- Robotic picking and sorting in warehouses

The dataset's diverse set of SKU categories and densely packed object arrangements make it a valuable resource for researchers and practitioners in the field of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv).

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. For the case of the SKU-110K dataset, the `SKU-110K.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/SKU-110K.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/SKU-110K.yaml).

!!! example "ultralytics/cfg/datasets/SKU-110K.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/SKU-110K.yaml"
    ```

## Usage

To train a YOLO11n model on the SKU-110K dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="SKU-110K.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=SKU-110K.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

The SKU-110k dataset contains a diverse set of retail shelf images with densely packed objects, providing rich context for object detection tasks. Here are some examples of data from the dataset, along with their corresponding annotations:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/densely-packed-retail-shelf-1.avif)

- **Densely packed retail shelf image**: This image demonstrates an example of densely packed objects in a retail shelf setting. Objects are annotated with bounding boxes and SKU category labels.

The example showcases the variety and complexity of the data in the SKU-110k dataset and highlights the importance of high-quality data for object detection tasks. The dense arrangement of products presents unique challenges for detection algorithms, making this dataset particularly valuable for developing robust retail-focused computer vision solutions.

## Citations and Acknowledgments

If you use the SKU-110k dataset in your research or development work, please cite the following paper:

!!! quote ""

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

## FAQ

### What is the SKU-110k dataset and why is it important for object detection?

The SKU-110k dataset consists of densely packed retail shelf images designed to aid research in object detection tasks. Developed by Eran Goldman et al., it includes over 110,000 unique SKU categories. Its importance lies in its ability to challenge state-of-the-art object detectors with diverse object appearances and proximity, making it an invaluable resource for researchers and practitioners in computer vision. Learn more about the dataset's structure and applications in our [SKU-110k Dataset](#sku-110k-dataset) section.

### How do I train a YOLO11 model using the SKU-110k dataset?

Training a YOLO11 model on the SKU-110k dataset is straightforward. Here's an example to train a YOLO11n model for 100 epochs with an image size of 640:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="SKU-110K.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=SKU-110K.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

### What are the main subsets of the SKU-110k dataset?

The SKU-110k dataset is organized into three main subsets:

1. **Training set**: Contains 8,219 images and annotations used for training object detection models.
2. **Validation set**: Consists of 588 images and annotations used for model validation during training.
3. **Test set**: Includes 2,936 images designed for the final evaluation of trained object detection models.

Refer to the [Dataset Structure](#dataset-structure) section for more details.

### How do I configure the SKU-110k dataset for training?

The SKU-110k dataset configuration is defined in a YAML file, which includes details about the dataset's paths, classes, and other relevant information. The `SKU-110K.yaml` file is maintained at [SKU-110K.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/SKU-110K.yaml). For example, you can train a model using this configuration as shown in our [Usage](#usage) section.

### What are the key features of the SKU-110k dataset in the context of [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl)?

The SKU-110k dataset features images of store shelves from around the world, showcasing densely packed objects that pose significant challenges for object detectors:

- Over 110,000 unique SKU categories
- Diverse object appearances
- Annotations include bounding boxes and SKU category labels

These features make the SKU-110k dataset particularly valuable for training and evaluating deep learning models in object detection tasks. For more details, see the [Key Features](#key-features) section.

### How do I cite the SKU-110k dataset in my research?

If you use the SKU-110k dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{goldman2019dense,
          author    = {Eran Goldman and Roei Herzig and Aviv Eisenschtat and Jacob Goldberger and Tal Hassner},
          title     = {Precise Detection in Densely Packed Scenes},
          booktitle = {Proc. Conf. Comput. Vision Pattern Recognition (CVPR)},
          year      = {2019}
        }
        ```

More information about the dataset can be found in the [Citations and Acknowledgments](#citations-and-acknowledgments) section.
