---
comments: true
description: Explore the TartanAir depth dataset for monocular depth estimation, a large synthetic dataset rendered in AirSim with diverse environments and dense ground truth used to train Ultralytics YOLO26-Depth models.
keywords: Ultralytics, YOLO, depth estimation, TartanAir, synthetic dataset, AirSim, visual SLAM, dense depth, monocular depth
---

# TartanAir Depth Dataset

[TartanAir](https://theairlab.org/tartanair-dataset/) is a large-scale synthetic dataset generated in the [AirSim](https://github.com/microsoft/AirSim) simulator. It was created to push the limits of visual SLAM and spans a wide variety of environments — indoor, outdoor, urban, and nature scenes — along with seasonal, weather, and lighting variation, and challenging conditions.

Because TartanAir is rendered in simulation, it provides dense depth ground truth across this diverse set of scenes, making it a strong source of environmental diversity and long-range geometry for training monocular [depth estimation](https://www.ultralytics.com/glossary/depth-estimation) models.

## Key Features

- **Synthetic** data generated in the AirSim simulator.
- Diverse **indoor and outdoor** environments (urban, nature) with seasonal, weather, and lighting variation, plus challenging conditions.
- **Dense** depth ground truth across all scenes.
- Depth range to **~80 m**.
- Contributes **61,470** images (55,660 train / 5,810 val) to the Ultralytics depth training mix.

## Dataset Structure

The TartanAir depth dataset is split into two subsets:

1. **Train**: 55,660 images with paired dense depth maps for training.
2. **Val**: 5,810 images with paired dense depth maps for validation during training.

Each RGB image is paired with a `.npy` float32 depth map storing per-pixel distances in meters, following the [Ultralytics depth dataset format](index.md).

## Role in YOLO26-Depth

TartanAir is one of the **training sources** in the broad multi-dataset mixture (~2.19M images) used to pretrain the Ultralytics YOLO26-Depth models. Within this mix, TartanAir contributes synthetic environmental diversity and long-range outdoor geometry that complement indoor and real-world sources.

There is no standalone held-out TartanAir benchmark in this setup. Instead, the resulting models are evaluated on the standard monocular depth benchmarks: NYU Depth V2, KITTI, Make3D, ETH3D, and iBims-1.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. For TartanAir, the `depth-tartanair.yaml` file defines the paths and the single `depth` class.

!!! example "ultralytics/cfg/datasets/depth-tartanair.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/depth-tartanair.yaml"
    ```

## Usage

To train a YOLO26n-Depth model on the TartanAir dataset with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-depth.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="depth-tartanair.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo depth train data=depth-tartanair.yaml model=yolo26n-depth.pt epochs=100 imgsz=640
        ```

## Pretrained Models

The YOLO26 depth family (`yolo26n-depth.pt`, `yolo26s-depth.pt`, `yolo26m-depth.pt`, `yolo26l-depth.pt`, `yolo26x-depth.pt`) auto-downloads from the [v8.4.0 release](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-depth.pt) and is trained on the broad multi-dataset mix that TartanAir is part of.

## Citations and Acknowledgments

If you use the TartanAir dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{wang2020tartanair,
              title={TartanAir: A Dataset to Push the Limits of Visual SLAM},
              author={Wenshan Wang and Delong Zhu and Xiangwei Wang and Yaoyu Hu and Yuheng Qiu and Chen Wang and Yafei Hu and Ashish Kapoor and Sebastian Scherer},
              booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
              year={2020}
        }
        ```

We would like to acknowledge the creators of TartanAir for making this diverse synthetic dataset available to the computer vision community.
