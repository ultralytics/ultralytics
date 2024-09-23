---
comments: true
description: Explore the comprehensive Argoverse dataset by Argo AI for 3D tracking, motion forecasting, and stereo depth estimation in autonomous driving research.
keywords: Argoverse dataset, autonomous driving, 3D tracking, motion forecasting, stereo depth estimation, Argo AI, LiDAR point clouds, high-resolution images, HD maps
---

# Argoverse Dataset

The [Argoverse](https://www.argoverse.org/) dataset is a collection of data designed to support research in autonomous driving tasks, such as 3D tracking, motion forecasting, and stereo depth estimation. Developed by Argo AI, the dataset provides a wide range of high-quality sensor data, including high-resolution images, LiDAR point clouds, and map data.

!!! note

    The Argoverse dataset `*.zip` file required for training was removed from Amazon S3 after the shutdown of Argo AI by Ford, but we have made it available for manual download on [Google Drive](https://drive.google.com/file/d/1st9qW3BeIwQsnR0t8mRpvbsSWIo16ACi/view?usp=drive_link).

## Key Features

- Argoverse contains over 290K labeled 3D object tracks and 5 million object instances across 1,263 distinct scenes.
- The dataset includes high-resolution camera images, LiDAR point clouds, and richly annotated HD maps.
- Annotations include 3D bounding boxes for objects, object tracks, and trajectory information.
- Argoverse provides multiple subsets for different tasks, such as 3D tracking, motion forecasting, and stereo depth estimation.

## Dataset Structure

The Argoverse dataset is organized into three main subsets:

1. **Argoverse 3D Tracking**: This subset contains 113 scenes with over 290K labeled 3D object tracks, focusing on 3D object tracking tasks. It includes LiDAR point clouds, camera images, and sensor calibration information.
2. **Argoverse Motion Forecasting**: This subset consists of 324K vehicle trajectories collected from 60 hours of driving data, suitable for motion forecasting tasks.
3. **Argoverse Stereo Depth Estimation**: This subset is designed for stereo depth estimation tasks and includes over 10K stereo image pairs with corresponding LiDAR point clouds for ground truth depth estimation.

## Applications

The Argoverse dataset is widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in autonomous driving tasks such as 3D object tracking, motion forecasting, and stereo depth estimation. The dataset's diverse set of sensor data, object annotations, and map information make it a valuable resource for researchers and practitioners in the field of autonomous driving.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. For the case of the Argoverse dataset, the `Argoverse.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Argoverse.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Argoverse.yaml).

!!! example "ultralytics/cfg/datasets/Argoverse.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/Argoverse.yaml"
    ```

## Usage

To train a YOLOv8n model on the Argoverse dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="Argoverse.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=Argoverse.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

The Argoverse dataset contains a diverse set of sensor data, including camera images, LiDAR point clouds, and HD map information, providing rich context for autonomous driving tasks. Here are some examples of data from the dataset, along with their corresponding annotations:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/argoverse-3d-tracking-sample.avif)

- **Argoverse 3D Tracking**: This image demonstrates an example of 3D object tracking, where objects are annotated with 3D bounding boxes. The dataset provides LiDAR point clouds and camera images to facilitate the development of models for this task.

The example showcases the variety and complexity of the data in the Argoverse dataset and highlights the importance of high-quality sensor data for autonomous driving tasks.

## Citations and Acknowledgments

If you use the Argoverse dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{chang2019argoverse,
          title={Argoverse: 3D Tracking and Forecasting with Rich Maps},
          author={Chang, Ming-Fang and Lambert, John and Sangkloy, Patsorn and Singh, Jagjeet and Bak, Slawomir and Hartnett, Andrew and Wang, Dequan and Carr, Peter and Lucey, Simon and Ramanan, Deva and others},
          booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
          pages={8748--8757},
          year={2019}
        }
        ```

We would like to acknowledge Argo AI for creating and maintaining the Argoverse dataset as a valuable resource for the autonomous driving research community. For more information about the Argoverse dataset and its creators, visit the [Argoverse dataset website](https://www.argoverse.org/).

## FAQ

### What is the Argoverse dataset and its key features?

The [Argoverse](https://www.argoverse.org/) dataset, developed by Argo AI, supports autonomous driving research. It includes over 290K labeled 3D object tracks and 5 million object instances across 1,263 distinct scenes. The dataset provides high-resolution camera images, LiDAR point clouds, and annotated HD maps, making it valuable for tasks like 3D tracking, motion forecasting, and stereo depth estimation.

### How can I train an Ultralytics YOLO model using the Argoverse dataset?

To train a YOLOv8 model with the Argoverse dataset, use the provided YAML configuration file and the following code:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="Argoverse.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=Argoverse.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

For a detailed explanation of the arguments, refer to the model [Training](../../modes/train.md) page.

### What types of data and annotations are available in the Argoverse dataset?

The Argoverse dataset includes various sensor data types such as high-resolution camera images, LiDAR point clouds, and HD map data. Annotations include 3D bounding boxes, object tracks, and trajectory information. These comprehensive annotations are essential for accurate model training in tasks like 3D object tracking, motion forecasting, and stereo depth estimation.

### How is the Argoverse dataset structured?

The dataset is divided into three main subsets:

1. **Argoverse 3D Tracking**: Contains 113 scenes with over 290K labeled 3D object tracks, focusing on 3D object tracking tasks. It includes LiDAR point clouds, camera images, and sensor calibration information.
2. **Argoverse Motion Forecasting**: Consists of 324K vehicle trajectories collected from 60 hours of driving data, suitable for motion forecasting tasks.
3. **Argoverse Stereo Depth Estimation**: Includes over 10K stereo image pairs with corresponding LiDAR point clouds for ground truth depth estimation.

### Where can I download the Argoverse dataset now that it has been removed from Amazon S3?

The Argoverse dataset `*.zip` file, previously available on Amazon S3, can now be manually downloaded from [Google Drive](https://drive.google.com/file/d/1st9qW3BeIwQsnR0t8mRpvbsSWIo16ACi/view?usp=drive_link).

### What is the YAML configuration file used for with the Argoverse dataset?

A YAML file contains the dataset's paths, classes, and other essential information. For the Argoverse dataset, the configuration file, `Argoverse.yaml`, can be found at the following link: [Argoverse.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Argoverse.yaml).

For more information about YAML configurations, see our [datasets](../index.md) guide.
