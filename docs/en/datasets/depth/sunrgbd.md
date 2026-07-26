---
comments: true
description: Explore the SUN RGB-D depth dataset for monocular depth estimation. Learn about its structure, usage, pretrained models, and role in YOLO26-Depth training.
keywords: Ultralytics, YOLO, depth estimation, SUN RGB-D, indoor RGB-D, multi-sensor, monocular depth, depth dataset
---

# SUN RGB-D Depth Dataset

[SUN RGB-D](https://rgbd.cs.princeton.edu/) is a real-world indoor scene-understanding benchmark captured with four different RGB-D sensors: Intel RealSense, Asus Xtion, and Microsoft Kinect v1 and v2. Its multi-sensor design makes it a valuable source of real indoor depth diversity for [monocular depth estimation](index.md).

## Key Features

- Captured with four different RGB-D sensors (Intel RealSense, Asus Xtion, Microsoft Kinect v1 and v2), providing real multi-sensor variety.
- Covers a broad range of real **indoor** scenes for scene-understanding research.
- Depth range up to approximately 10 m, typical of consumer indoor RGB-D capture.
- Sensor-derived depth ground truth aligned to the RGB frames.
- Contributes real multi-sensor indoor diversity to the Ultralytics depth pretraining mix.

## Dataset Structure

The SUN RGB-D depth dataset is split into two subsets:

1. **Train**: 9,245 images with paired depth maps for training.
2. **Val**: 1,090 images with paired depth maps for validation during model training.

Each sample consists of one RGB image and one paired `.npy` float32 depth map storing per-pixel distances in meters, following the [Ultralytics depth dataset format](index.md).

## Role in YOLO26-Depth

SUN RGB-D is a **training** source in the Ultralytics YOLO26-Depth multi-dataset pretraining mix of roughly 2.19M image–depth pairs. It contributes real multi-sensor indoor diversity, exposing the model to depth captured across several different consumer RGB-D devices. The resulting models are evaluated on the standard NYU, KITTI, Make3D, ETH3D, and iBims-1 benchmarks.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information.

!!! example "ultralytics/cfg/datasets/depth-sunrgbd.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/depth-sunrgbd.yaml"
    ```

## Usage

To train a YOLO26n-depth model on the SUN RGB-D dataset with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-depth.pt")  # load a pretrained depth model (recommended for training)

        # Train the model
        results = model.train(data="depth-sunrgbd.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo depth train data=depth-sunrgbd.yaml model=yolo26n-depth.pt epochs=100 imgsz=640
        ```

## Pretrained Models

The YOLO26 depth family is trained on the broad multi-dataset depth pretraining mix that SUN RGB-D is part of. These models auto-download from the latest Ultralytics release, for example [YOLO26x-depth](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-depth.pt) from v8.4.0, and span a range of sizes for different accuracy and resource requirements.

## Citations and Acknowledgments

If you use the SUN RGB-D dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{song2015sunrgbd,
              title={SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite},
              author={Song, Shuran and Lichtenberg, Samuel P. and Xiao, Jianxiong},
              booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
              year={2015}
        }
        ```

We would like to acknowledge the authors for creating and maintaining this valuable resource for the computer vision community.
