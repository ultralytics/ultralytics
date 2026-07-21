---
comments: true
description: Explore the Virtual KITTI 2 depth dataset for monocular depth estimation, a photorealistic synthetic recreation of KITTI driving scenes with dense per-pixel ground truth used to train Ultralytics YOLO26-Depth models.
keywords: Ultralytics, YOLO, depth estimation, Virtual KITTI 2, vKITTI2, synthetic driving dataset, dense depth, monocular depth, autonomous driving
---

# Virtual KITTI 2 Depth Dataset

[Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) (vKITTI2) is a photorealistic synthetic recreation of the KITTI driving scenes. It clones 5 sequences from the original KITTI dataset and re-renders them under varied weather and lighting conditions, providing dense per-pixel ground truth.

As a synthetic outdoor-driving dataset, vKITTI2 offers a dense counterpart to the sparse real KITTI LiDAR returns, making it a useful source of clean outdoor driving geometry for training monocular [depth estimation](https://www.ultralytics.com/glossary/depth-estimation) models.

## Key Features

- Photorealistic **synthetic** recreation of KITTI driving scenes.
- **5 cloned sequences** rendered under varied weather and lighting.
- **Outdoor driving** environments.
- Dense **per-pixel** ground truth (a dense counterpart to the sparse real KITTI LiDAR).
- Depth range **~80 m**.
- Contributes **42,520** images (25,780 train / 16,740 val) to the Ultralytics depth training mix.

## Dataset Structure

The Virtual KITTI 2 depth dataset is split into two subsets:

1. **Train**: 25,780 images with paired dense depth maps for training.
2. **Val**: 16,740 images with paired dense depth maps for validation during training.

Each RGB image is paired with a `.npy` float32 depth map storing per-pixel distances in meters, following the [Ultralytics depth dataset format](index.md).

## Role in YOLO26-Depth

Virtual KITTI 2 is one of the **training sources** in the broad multi-dataset mixture (~2.19M images) used to pretrain the Ultralytics YOLO26-Depth models. Within this mix, vKITTI2 provides a dense synthetic outdoor-driving counterpart to the sparse real KITTI LiDAR ground truth.

There is no standalone held-out vKITTI2 benchmark in this setup. Instead, the resulting models are evaluated on the standard monocular depth benchmarks: NYU Depth V2, KITTI, Make3D, ETH3D, and iBims-1.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. For Virtual KITTI 2, the `depth-vkitti2.yaml` file defines the paths and the single `depth` class.

!!! example "ultralytics/cfg/datasets/depth-vkitti2.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/depth-vkitti2.yaml"
    ```

## Usage

To train a YOLO26n-Depth model on the Virtual KITTI 2 dataset with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-depth.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="depth-vkitti2.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo depth train data=depth-vkitti2.yaml model=yolo26n-depth.pt epochs=100 imgsz=640
        ```

## Pretrained Models

The YOLO26 depth family (`yolo26n-depth.pt`, `yolo26s-depth.pt`, `yolo26m-depth.pt`, `yolo26l-depth.pt`, `yolo26x-depth.pt`) auto-downloads from the [v8.4.0 release](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-depth.pt) and is trained on the broad multi-dataset mix that Virtual KITTI 2 is part of.

## Citations and Acknowledgments

If you use the Virtual KITTI 2 dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{cabon2020vkitti2,
              title={Virtual KITTI 2},
              author={Yohann Cabon and Naila Murray and Martin Humenberger},
              journal={arXiv preprint arXiv:2001.10773},
              year={2020}
        }
        ```

We would like to acknowledge the creators of Virtual KITTI 2 for making this synthetic driving dataset available to the computer vision community.
