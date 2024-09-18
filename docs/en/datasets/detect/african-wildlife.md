---
comments: true
description: Explore our African Wildlife Dataset featuring images of buffalo, elephant, rhino, and zebra for training computer vision models. Ideal for research and conservation.
keywords: African Wildlife Dataset, South African animals, object detection, computer vision, YOLOv8, wildlife research, conservation, dataset
---

# African Wildlife Dataset

This dataset showcases four common animal classes typically found in South African nature reserves. It includes images of African wildlife such as buffalo, elephant, rhino, and zebra, providing valuable insights into their characteristics. Essential for training computer vision algorithms, this dataset aids in identifying animals in various habitats, from zoos to forests, and supports wildlife research.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/biIW5Z6GYl0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> African Wildlife Animals Detection using Ultralytics YOLOv8
</p>

## Dataset Structure

The African wildlife objects detection dataset is split into three subsets:

- **Training set**: Contains 1052 images, each with corresponding annotations.
- **Validation set**: Includes 225 images, each with paired annotations.
- **Testing set**: Comprises 227 images, each with paired annotations.

## Applications

This dataset can be applied in various computer vision tasks such as object detection, object tracking, and research. Specifically, it can be used to train and evaluate models for identifying African wildlife objects in images, which can have applications in wildlife conservation, ecological research, and monitoring efforts in natural reserves and protected areas. Additionally, it can serve as a valuable resource for educational purposes, enabling students and researchers to study and understand the characteristics and behaviors of different animal species.

## Dataset YAML

A YAML (Yet Another Markup Language) file defines the dataset configuration, including paths, classes, and other pertinent details. For the African wildlife dataset, the `african-wildlife.yaml` file is located at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/african-wildlife.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/african-wildlife.yaml).

!!! example "ultralytics/cfg/datasets/african-wildlife.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/african-wildlife.yaml"
    ```

## Usage

To train a YOLOv8n model on the African wildlife dataset for 100 epochs with an image size of 640, use the provided code samples. For a comprehensive list of available parameters, refer to the model's [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="african-wildlife.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=african-wildlife.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

!!! example "Inference Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/best.pt")  # load a brain-tumor fine-tuned model

        # Inference using the model
        results = model.predict("https://ultralytics.com/assets/african-wildlife-sample.jpg")
        ```

    === "CLI"

        ```bash
        # Start prediction with a finetuned *.pt model
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/african-wildlife-sample.jpg"
        ```

## Sample Images and Annotations

The African wildlife dataset comprises a wide variety of images showcasing diverse animal species and their natural habitats. Below are examples of images from the dataset, each accompanied by its corresponding annotations.

![African wildlife dataset sample image](https://github.com/ultralytics/docs/releases/download/0/african-wildlife-dataset-sample.avif)

- **Mosaiced Image**: Here, we present a training batch consisting of mosaiced dataset images. Mosaicing, a training technique, combines multiple images into one, enriching batch diversity. This method helps enhance the model's ability to generalize across different object sizes, aspect ratios, and contexts.

This example illustrates the variety and complexity of images in the African wildlife dataset, emphasizing the benefits of including mosaicing during the training process.

## Citations and Acknowledgments

The dataset has been released available under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

## FAQ

### What is the African Wildlife Dataset, and how can it be used in computer vision projects?

The African Wildlife Dataset includes images of four common animal species found in South African nature reserves: buffalo, elephant, rhino, and zebra. It is a valuable resource for training computer vision algorithms in object detection and animal identification. The dataset supports various tasks like object tracking, research, and conservation efforts. For more information on its structure and applications, refer to the [Dataset Structure](#dataset-structure) section and [Applications](#applications) of the dataset.

### How do I train a YOLOv8 model using the African Wildlife Dataset?

You can train a YOLOv8 model on the African Wildlife Dataset by using the `african-wildlife.yaml` configuration file. Below is an example of how to train the YOLOv8n model for 100 epochs with an image size of 640:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="african-wildlife.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=african-wildlife.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

For additional training parameters and options, refer to the [Training](../../modes/train.md) documentation.

### Where can I find the YAML configuration file for the African Wildlife Dataset?

The YAML configuration file for the African Wildlife Dataset, named `african-wildlife.yaml`, can be found at [this GitHub link](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/african-wildlife.yaml). This file defines the dataset configuration, including paths, classes, and other details crucial for training machine learning models. See the [Dataset YAML](#dataset-yaml) section for more details.

### Can I see sample images and annotations from the African Wildlife Dataset?

Yes, the African Wildlife Dataset includes a wide variety of images showcasing diverse animal species in their natural habitats. You can view sample images and their corresponding annotations in the [Sample Images and Annotations](#sample-images-and-annotations) section. This section also illustrates the use of mosaicing technique to combine multiple images into one for enriched batch diversity, enhancing the model's generalization ability.

### How can the African Wildlife Dataset be used to support wildlife conservation and research?

The African Wildlife Dataset is ideal for supporting wildlife conservation and research by enabling the training and evaluation of models to identify African wildlife in different habitats. These models can assist in monitoring animal populations, studying their behavior, and recognizing conservation needs. Additionally, the dataset can be utilized for educational purposes, helping students and researchers understand the characteristics and behaviors of different animal species. More details can be found in the [Applications](#applications) section.
