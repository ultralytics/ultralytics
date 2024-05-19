---
comments: true
description: Explore Argoverse, a comprehensive dataset for autonomous driving tasks including 3D tracking, motion forecasting and depth estimation used in YOLO.
keywords: Argoverse dataset, autonomous driving, YOLO, 3D tracking, motion forecasting, LiDAR data, HD maps, ultralytics documentation
---

# Argoverse Dataset

The [Argoverse](https://www.argoverse.org/) dataset is a collection of data designed to support research in autonomous driving tasks, such as 3D tracking, motion forecasting, and stereo depth estimation. Developed by Argo AI, the dataset provides a wide range of high-quality sensor data, including high-resolution images, LiDAR point clouds, and map data.

!!! Note

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

The Argoverse dataset is widely used for training and evaluating deep learning models in autonomous driving tasks such as 3D object tracking, motion forecasting, and stereo depth estimation. The dataset's diverse set of sensor data, object annotations, and map information make it a valuable resource for researchers and practitioners in the field of autonomous driving.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. For the case of the Argoverse dataset, the `Argoverse.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Argoverse.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Argoverse.yaml).

!!! Example "ultralytics/cfg/datasets/Argoverse.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/Argoverse.yaml"
    ```

## Usage

To train a YOLOv8n model on the Argoverse dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! Example "Train Example"

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

![Dataset sample image](https://www.argoverse.org/assets/images/reference_images/av2_ground_height.png)

- **Argoverse 3D Tracking**: This image demonstrates an example of 3D object tracking, where objects are annotated with 3D bounding boxes. The dataset provides LiDAR point clouds and camera images to facilitate the development of models for this task.

The example showcases the variety and complexity of the data in the Argoverse dataset and highlights the importance of high-quality sensor data for autonomous driving tasks.

## Citations and Acknowledgments

If you use the Argoverse dataset in your research or development work, please cite the following paper:

!!! Quote ""

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
