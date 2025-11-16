---
comments: true
description: Explore the Ultralytics kitti dataset, a benchmark dataset for computer vision tasks such as 3D object detection, depth estimation, and autonomous driving perception.
keywords: kitti, Ultralytics, dataset, object detection, 3D vision, YOLO11, training, validation, self-driving cars, computer vision
---

# KITTI Dataset

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-kitti-detection-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open KITTI Dataset In Colab"></a>

The kitti dataset is one of the most influential benchmark datasets for autonomous driving and computer vision. Released by the Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago, it contains stereo camera, LiDAR, and GPS/IMU data collected from real-world driving scenarios.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/NNeDlTbq9pA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train Ultralytics YOLO11 on the KITTI Dataset 🚀
</p>

It is widely used for evaluating algorithms in object detection, depth estimation, optical flow, and visual odometry. The dataset is fully compatible with Ultralytics YOLO11 for 2D object detection tasks and can be easily integrated into the Ultralytics platform for training and evaluation.

## Dataset Structure

!!! warning

    Kitti original test set is excluded here since it does not contain ground-truth annotations.

In total, the dataset includes 7,481 images, each paired with detailed annotations for objects such as cars, pedestrians, cyclists, and other road elements. The dataset is divided into two main subsets:

- **Training set:** Contains 5,985 images with annotated labels used for model training.
- **Validation set:** Includes 1,496 images with corresponding annotations used for performance evaluation and benchmarking.

## Applications

Kitti dataset enables advancements in autonomous driving and robotics, supporting tasks like:

- **Autonomous vehicle perception**: Training models to detect and track vehicles, pedestrians, and obstacles for safe navigation in self-driving systems.
- **3D scene understanding**: Supporting depth estimation, stereo vision, and 3D object localization to help machines understand spatial environments.
- **Optical flow and motion prediction**: Enabling motion analysis to predict the movement of objects and improve trajectory planning in dynamic environments.
- **Computer vision benchmarking**: Serving as a standard benchmark for evaluating performance across multiple vision tasks, including object detection, and tracking.

## Dataset YAML

Ultralytics defines the kitti dataset configuration using a YAML file. This file specifies dataset paths, class labels, and metadata required for training. The configuration file is available at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml).

!!! example "ultralytics/cfg/datasets/kitti.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/kitti.yaml"
    ```

## Usage

To train a YOLO11n model on the kitti dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the following commands. For more details, refer to the [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO11 model
        model = YOLO("yolo11n.pt")

        # Train on kitti dataset
        results = model.train(data="kitti.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=kitti.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

You can also perform evaluation, [inference](../../modes/predict.md), and [export](../../modes/export.md) tasks directly from the command line or Python API using the same configuration file.

## Sample Images and Annotations

The kitti dataset provides diverse driving scenarios. Each image includes bounding box annotations for 2D object detection tasks. The example showcase the dataset rich variety, enabling robust model generalization across diverse real-world conditions.

<img src="https://github.com/ultralytics/docs/releases/download/0/kitti-dataset-sample.avif" alt="Kitti sample image" width="800">

## Citations and Acknowledgments

If you use the kitti dataset in your research, please cite the following paper:

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

We acknowledge the KITTI Vision Benchmark Suite for providing this comprehensive dataset that continues to shape progress in computer vision, robotics, and autonomous systems. Visit the [kitti website](https://www.cvlibs.net/datasets/kitti/) for more information.

## FAQs

### What is the kitti dataset used for?

The kitti dataset is primarily used for computer vision research in autonomous driving, supporting tasks like object detection, depth estimation, optical flow, and 3D localization.

### How many images are included in the kitti dataset?

The dataset includes 5,985 labeled training images and 1,496 validation images captured across urban, rural, and highway scenes. The original test set is excluded here since it does not contain ground-truth annotations.

### Which object classes are annotated in the dataset?

kitti includes annotations for objects such as cars, pedestrians, cyclists, trucks, trams, and miscellaneous road users.

### Can I train Ultralytics YOLO11 models using the kitti dataset?

Yes, kitti is fully compatible with Ultralytics YOLO11. You can [train](../../modes/train.md) and [validate](../../modes/val.md), models directly using the provided YAML configuration file.

### Where can I find the kitti dataset configuration file?

You can access the YAML file at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/kitti.yaml).
