---
comments: true
description: Discover the Objects365 dataset, designed for object detection research with a focus on diverse objects, featuring 365 categories, 2 million images, and 30 million bounding boxes.
keywords: Objects365 dataset, object detection, computer vision, deep learning, Ultralytics Docs
---

# Objects365 Dataset

The [Objects365](https://www.objects365.org/) dataset is a large-scale, high-quality dataset designed to foster object detection research with a focus on diverse objects in the wild. Created by a team of [Megvii](https://en.megvii.com/) researchers, the dataset offers a wide range of high-resolution images with a comprehensive set of annotated bounding boxes covering 365 object categories.

## Key Features

- Objects365 contains 365 object categories, with 2 million images and over 30 million bounding boxes.
- The dataset includes diverse objects in various scenarios, providing a rich and challenging benchmark for object detection tasks.
- Annotations include bounding boxes for objects, making it suitable for training and evaluating object detection models.
- Objects365 pre-trained models significantly outperform ImageNet pre-trained models, leading to better generalization on various tasks.

## Dataset Structure

The Objects365 dataset is organized into a single set of images with corresponding annotations:

- **Images**: The dataset includes 2 million high-resolution images, each containing a variety of objects across 365 categories.
- **Annotations**: The images are annotated with over 30 million bounding boxes, providing comprehensive ground truth information for object detection tasks.

## Applications

The Objects365 dataset is widely used for training and evaluating deep learning models in object detection tasks. The dataset's diverse set of object categories and high-quality annotations make it a valuable resource for researchers and practitioners in the field of computer vision.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. For the case of the Objects365 Dataset, the `Objects365.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/datasets/Objects365.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/datasets/Objects365.yaml).

!!! example "ultralytics/datasets/Objects365.yaml"

    ```yaml
    --8<-- "ultralytics/datasets/Objects365.yaml"
    ```

## Usage

To train a YOLOv8n model on the Objects365 dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
        
        # Train the model
        model.train(data='Objects365.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=Objects365.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

The Objects365 dataset contains a diverse set of high-resolution images with objects from 365 categories, providing rich context for object detection tasks. Here are some examples of the images in the dataset:

![Dataset sample image](https://user-images.githubusercontent.com/26833433/238215467-caf757dd-0b87-4b0d-bb19-d94a547f7fbf.jpg)

- **Objects365**: This image demonstrates an example of object detection, where objects are annotated with bounding boxes. The dataset provides a wide range of images to facilitate the development of models for this task.

The example showcases the variety and complexity of the data in the Objects365 dataset and highlights the importance of accurate object detection for computer vision applications.

## Citations and Acknowledgments

If you use the Objects365 dataset in your research or development work, please cite the following paper:

```bibtex
@inproceedings{shao2019objects365,
  title={Objects365: A Large-scale, High-quality Dataset for Object Detection},
  author={Shao, Shuai and Li, Zeming and Zhang, Tianyuan and Peng, Chao and Yu, Gang and Li, Jing and Zhang, Xiangyu and Sun, Jian},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={8425--8434},
  year={2019}
}
```

We would like to acknowledge the team of researchers who created and maintain the Objects365 dataset as a valuable resource for the computer vision research community. For more information about the Objects365 dataset and its creators, visit the [Objects365 dataset website](https://www.objects365.org/).