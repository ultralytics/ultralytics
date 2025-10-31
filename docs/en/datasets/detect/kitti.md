---
comments: true
description: Explore the Ultralytics Kitti dataset, a benchmark dataset for computer vision tasks such as 3D object detection, depth estimation, and autonomous driving perception.
keywords: Kitti, Ultralytics, dataset, object detection, 3D vision, YOLO11, training, validation, self-driving cars, computer vision
---

# Kitti Dataset

The Kitti dataset is one of the most influential benchmark datasets for autonomous driving and computer vision. Released by the Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago, it contains stereo camera, LiDAR, and GPS/IMU data collected from real-world driving scenarios. It is widely used for evaluating algorithms in object detection, depth estimation, optical flow, and visual odometry. The dataset offers diverse scenes, including urban, rural, and highway environments, providing rich, multi-modal data for perception systems.

The dataset is fully compatible with Ultralytics YOLO11 for 2D object detection tasks and can be easily integrated into the Ultralytics platform for training and evaluation.

## Dataset Structure

In total, the dataset includes 7,481 images, each paired with detailed annotations for objects such as cars, pedestrians, cyclists, and other road elements. The images capture a variety of urban, rural, and highway scenes, making it suitable for multiple vision-based perception tasks like object detection and 3D localization. The dataset is divided into two main subsets:

**Training set:** Contains 5,985 images with annotated labels used for model training.
**Validation set:** Includes 1,496 images with corresponding annotations used for performance evaluation and benchmarking.

## Applications

Using computer vision with the KITTI dataset enables advancements in autonomous driving and robotics, supporting tasks like:

- **Autonomous Vehicle Perception**: Training models to detect and track vehicles, pedestrians, and obstacles for safe navigation in self-driving systems.
- **3D Scene Understanding**: Supporting depth estimation, stereo vision, and 3D object localization to help machines understand spatial environments.
- **Optical Flow and Motion Prediction**: Enabling motion analysis to predict the movement of objects and improve trajectory planning in dynamic environments.
- **Computer Vision Benchmarking**: Serving as a standard benchmark for evaluating performance across multiple vision tasks, including object detection, and tracking.

## Dataset YAML

Ultralytics defines the Kitti dataset configuration using a YAML file. This file specifies dataset paths, class labels, and metadata required for training. The configuration file is available at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml).

!!! example "ultralytics/cfg/datasets/kitti.yaml"

    ```yaml
      --8<-- "ultralytics/cfg/datasets/kitti.yaml"
    ```

The dataset typically contains 7,481 training images and 7,518 test images, with annotations covering classes such as Car, Pedestrian, Cyclist, Truck, Tram, and Misc.

## Usage

To train a YOLO11n model on the kitti dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the following commands. For more details, refer to the [Training](../../modes/train.md) page.

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

The Kitti dataset provides diverse driving scenarios. Each image includes bounding box annotations for 2D object detection tasks. These examples showcase the dataset's rich variety, enabling robust model generalization across diverse real-world conditions.

<img src="https://github.com/ultralytics/docs/releases/download/0/kitti-dataset-sample.avif" alt="Kitti sample image" width="800">

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

We acknowledge the KITTI Vision Benchmark Suite for providing this comprehensive dataset that continues to shape progress in computer vision, robotics, and autonomous systems. Visit the [Kitti website](https://www.cvlibs.net/datasets/kitti/) for more information.

## FAQs

### What is the Kitti dataset used for?

The Kitti dataset is primarily used for computer vision research in autonomous driving, supporting tasks like object detection, depth estimation, optical flow, and 3D localization.

### How many images are included in the Kitti dataset?

The dataset contains 5,985 labeled training images and 1496 test images covering urban, rural, and highway driving scenarios. The original test images are not used here, since these don't have the ground-truth labels.

### Which object classes are annotated in the dataset?

Kitti includes annotations for objects such as cars, pedestrians, cyclists, trucks, trams, and miscellaneous road users.

### Can I train Ultralytics YOLO11 models using the Kitti dataset?

Yes, Kitti is fully compatible with Ultralytics YOLO11. You can [train](../../modes/train.md) and [validate]((../../modes/val.md), models directly using the provided YAML configuration file.

### Where can I find the Kitti dataset configuration file?

You can access the YAML file at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml).
