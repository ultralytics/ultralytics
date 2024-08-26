---
comments: true
description: Explore the SKU-110k dataset of densely packed retail shelf images, perfect for training and evaluating deep learning models in object detection tasks.
keywords: SKU-110k, dataset, object detection, retail shelf images, deep learning, computer vision, model training
---

# SKU-110k Dataset

The [SKU-110k](https://github.com/eg4000/SKU110K_CVPR19) dataset is a collection of densely packed retail shelf images, designed to support research in object detection tasks. Developed by Eran Goldman et al., the dataset contains over 110,000 unique store keeping unit (SKU) categories with densely packed objects, often looking similar or even identical, positioned in close proximity.

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

![Dataset sample image](https://user-images.githubusercontent.com/26833433/277141199-e7cdd803-237e-4b4a-9171-f95cba9388f9.jpg)

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

!!! Example "ultralytics/cfg/datasets/SKU-110K.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/SKU-110K.yaml"
    ```

## Usage

To train a YOLOv8n model on the SKU-110K dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! Example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="SKU-110K.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=SKU-110K.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

The SKU-110k dataset contains a diverse set of retail shelf images with densely packed objects, providing rich context for object detection tasks. Here are some examples of data from the dataset, along with their corresponding annotations:

![Dataset sample image](https://user-images.githubusercontent.com/26833433/277141197-b63e4aa5-12f6-4673-96a7-9a5207363c59.jpg)

- **Densely packed retail shelf image**: This image demonstrates an example of densely packed objects in a retail shelf setting. Objects are annotated with bounding boxes and SKU category labels.

The example showcases the variety and complexity of the data in the SKU-110k dataset and highlights the importance of high-quality data for object detection tasks.

## Citations and Acknowledgments

If you use the SKU-110k dataset in your research or development work, please cite the following paper:

!!! Quote ""

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

The SKU-110k dataset consists of densely packed retail shelf images designed to aid research in object detection tasks. Developed by Eran Goldman et al., it includes over 110,000 unique SKU categories. Its importance lies in its ability to challenge state-of-the-art object detectors with diverse object appearances and close proximity, making it an invaluable resource for researchers and practitioners in computer vision. Learn more about the dataset's structure and applications in our [SKU-110k Dataset](#sku-110k-dataset) section.

### How do I train a YOLOv8 model using the SKU-110k dataset?

Training a YOLOv8 model on the SKU-110k dataset is straightforward. Here's an example to train a YOLOv8n model for 100 epochs with an image size of 640:

!!! Example "Train Example"

    === "Python"
    
        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="SKU-110K.yaml", epochs=100, imgsz=640)
        ```
    

    === "CLI"
    
        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=SKU-110K.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

### What are the main subsets of the SKU-110k dataset?

The SKU-110k dataset is organized into three main subsets:

1. **Training set**: Contains images and annotations used for training object detection models.
2. **Validation set**: Consists of images and annotations used for model validation during training.
3. **Test set**: Designed for the final evaluation of trained object detection models.

Refer to the [Dataset Structure](#dataset-structure) section for more details.

### How do I configure the SKU-110k dataset for training?

The SKU-110k dataset configuration is defined in a YAML file, which includes details about the dataset's paths, classes, and other relevant information. The `SKU-110K.yaml` file is maintained at [SKU-110K.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/SKU-110K.yaml). For example, you can train a model using this configuration as shown in our [Usage](#usage) section.

### What are the key features of the SKU-110k dataset in the context of deep learning?

The SKU-110k dataset features images of store shelves from around the world, showcasing densely packed objects that pose significant challenges for object detectors:

- Over 110,000 unique SKU categories
- Diverse object appearances
- Annotations include bounding boxes and SKU category labels

These features make the SKU-110k dataset particularly valuable for training and evaluating deep learning models in object detection tasks. For more details, see the [Key Features](#key-features) section.

### How do I cite the SKU-110k dataset in my research?

If you use the SKU-110k dataset in your research or development work, please cite the following paper:

!!! Quote ""

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
