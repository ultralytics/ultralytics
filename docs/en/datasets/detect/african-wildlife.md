---
comments: true
description: African Wildlife objects detection, a leading dataset for object detection in forests, integrates with Ultralytics. Discover ways to use it for training YOLO models.
keywords: Ultralytics, African Wildlife dataset, object detection, YOLO, YOLO model training, object tracking, computer vision, deep learning models, forest research, animals tracking
---

# African Wildlife Dataset

This dataset showcases four common animal classes typically found in South African nature reserves. It includes images of African wildlife such as buffalo, elephant, rhino, and zebra, providing valuable insights into their characteristics. Essential for training computer vision algorithms, this dataset aids in identifying animals in various habitats, from zoos to forests, and supports wildlife research.

## Dataset Structure

The African wildlife objects detection dataset is split into three subsets:

- **Training set**: Contains 1052 images, each with corresponding annotations.
- **Validation set**: Includes 225 images, each with paired annotations.
- **Testing set**: Comprises 227 images, each with paired annotations.

## Applications

This dataset can be applied in various computer vision tasks such as object detection, object tracking, and research. Specifically, it can be used to train and evaluate models for identifying African wildlife objects in images, which can have applications in wildlife conservation, ecological research, and monitoring efforts in natural reserves and protected areas. Additionally, it can serve as a valuable resource for educational purposes, enabling students and researchers to study and understand the characteristics and behaviors of different animal species.

## Dataset YAML

A YAML (Yet Another Markup Language) file defines the dataset configuration, including paths, classes, and other pertinent details. For the African wildlife dataset, the `african-wildlife.yaml` file is located at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/african-wildlife.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/african-wildlife.yaml).

!!! Example "ultralytics/cfg/datasets/african-wildlife.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/african-wildlife.yaml"
    ```

## Usage

To train a YOLOv8n model on the African wildlife dataset for 100 epochs with an image size of 640, use the provided code samples. For a comprehensive list of available parameters, refer to the model's [Training](../../modes/train.md) page.

!!! Example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data='african-wildlife.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=african-wildlife.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

!!! Example "Inference Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('path/to/best.pt')  # load a brain-tumor fine-tuned model

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

![African wildlife dataset sample image](https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/919f8190-ccf3-4a96-a5f1-55d9eebc77ec)

- **Mosaiced Image**: Here, we present a training batch consisting of mosaiced dataset images. Mosaicing, a training technique, combines multiple images into one, enriching batch diversity. This method helps enhance the model's ability to generalize across different object sizes, aspect ratios, and contexts.

This example illustrates the variety and complexity of images in the African wildlife dataset, emphasizing the benefits of including mosaicing during the training process.

## Citations and Acknowledgments

The dataset has been released available under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).