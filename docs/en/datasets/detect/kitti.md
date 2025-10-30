---
comments: true
description: Explore the Ultralytics Kitti dataset, a benchmark dataset for computer vision tasks such as 3D object detection, depth estimation, and autonomous driving perception.
keywords: Kitti, Ultralytics, dataset, object detection, 3D vision, YOLO11, training, validation, self-driving cars, computer vision
---

# Kitti Dataset Introduction

The Kitti dataset is one of the most influential benchmark datasets for autonomous driving and computer vision. Released by the Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago, it contains stereo camera, LiDAR, and GPS/IMU data collected from real-world driving scenarios. It is widely used for evaluating algorithms in object detection, depth estimation, optical flow, and visual odometry. The dataset offers diverse scenes, including urban, rural, and highway environments, providing rich, multi-modal data for perception systems.

The dataset is fully compatible with Ultralytics YOLO11 for 2D object detection tasks and can be easily integrated into the Ultralytics platform for training and evaluation.

## Dataset YAML

Ultralytics defines the Kitti dataset configuration using a YAML file. This file specifies dataset paths, class labels, and metadata required for training. The configuration file is available at https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml.

!!! example "ultralytics/cfg/datasets/kitti.yaml"

    ```yaml
      --8<-- "ultralytics/cfg/datasets/kitti.yaml"
    ```

The dataset typically contains 7,481 training images and 7,518 test images, with annotations covering classes such as Car, Pedestrian, Cyclist, Truck, Tram, and Misc.

## Usage

To train a YOLO11n model on the Kitti dataset for 100 epochs with an image size of 640, use the following commands. For more details, refer to the Training page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO11 model
        model = YOLO("yolo11n.pt")

        # Train on Kitti dataset
        results = model.train(data="kitti.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=kitti.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

You can also perform evaluation, inference, and export tasks directly from the command line or Python API using the same configuration file.

## Sample Images and Annotations

The Kitti dataset provides diverse driving scenarios captured from an ego-vehicle setup. Each image includes bounding box annotations for 2D object detection tasks.

<img src="https://github.com/ultralytics/docs/releases/download/0/kitti-dataset-sample.avif" alt="Kitti sample image" width="800">

These examples showcase the dataset’s rich variety, enabling robust model generalization across diverse real-world conditions.

## Citations and Acknowledgments

If you use the Kitti dataset in your research, please cite the following paper:

!!! quote

    === "BibTeX"

        ```bibtex
        @article{Geiger2013IJRR,
            author = {Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun},
            title = {Vision meets Robotics: The KITTI Dataset},
            journal = {International Journal of Robotics Research (IJRR)},
            year = {2013}
        }
        ```

We acknowledge the KITTI Vision Benchmark Suite for providing this comprehensive dataset that continues to shape progress in computer vision, robotics, and autonomous systems. Visit the Kitti website for more information.
