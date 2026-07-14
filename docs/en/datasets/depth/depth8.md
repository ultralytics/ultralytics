---
comments: true
description: Explore the Ultralytics Depth8 dataset, a compact set of 8 indoor RGB-D images for testing monocular depth estimation models and training pipelines.
keywords: Depth8, Ultralytics, dataset, depth estimation, monocular depth, YOLO26, training, validation, debugging, SUN RGB-D
---

# Depth8 Dataset

## Introduction

The [Ultralytics](https://www.ultralytics.com/) Depth8 dataset is a compact [monocular depth estimation](index.md) dataset with 8 images sampled from the [SUN RGB-D](sunrgbd.md) dataset: 4 for training and 4 for validation, one per RGB-D sensor (Kinect v1, Kinect v2, Intel RealSense, Asus Xtion) in each split. It is designed for rapid testing, debugging, and experimentation with [YOLO26](../../models/yolo26.md) depth estimation models and training pipelines — the 1.5 MB archive auto-downloads on first use, so `yolo depth train data=depth8.yaml` starts training within seconds.

!!! note

    Depth8 is for pipeline testing only, not benchmarking — its 8 images are far too few for meaningful depth metrics. Use the full [NYU Depth V2](nyu-depth-v2.md) or [SUN RGB-D](sunrgbd.md) validation sets for representative results.

## Dataset Structure

Depth8 follows the standard [Ultralytics depth dataset layout](index.md#supported-dataset-format): RGB images with paired `.npy` float32 depth maps in meters, matched by file stem.

```text
depth8/
├── images/
│   ├── train/  # 4 images
│   └── val/    # 4 images
└── depth/
    ├── train/  # 4 float32 .npy depth maps
    └── val/    # 4 float32 .npy depth maps
```

Depth values are real indoor sensor captures ranging from roughly 0.6 m to 9 m, matching the ≤10 m range of the full SUN RGB-D dataset.

## Dataset YAML

The Depth8 dataset configuration is defined in a dataset YAML file, which specifies dataset paths, class names, and the download URL for the small packaged subset.

!!! example "ultralytics/cfg/datasets/depth8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/depth8.yaml"
    ```

## Usage

To train a YOLO26n-depth model on the Depth8 dataset with an image size of 640, use the following examples. For a full list of training options, see the [YOLO Training documentation](../../modes/train.md).

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO26n-depth model
        model = YOLO("yolo26n-depth.pt")

        # Train the model on Depth8
        results = model.train(data="depth8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Train YOLO26n-depth on Depth8 using the command line
        yolo depth train data=depth8.yaml model=yolo26n-depth.pt epochs=100 imgsz=640
        ```

## Citations and Acknowledgments

Depth8 is sampled from SUN RGB-D — see the full [SUN RGB-D dataset page](sunrgbd.md#citations-and-acknowledgments) for license details.

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

## FAQ

### What is the Ultralytics Depth8 dataset used for?

The Ultralytics Depth8 dataset is designed for rapid testing and debugging of [monocular depth estimation](../../tasks/depth.md) pipelines. With only 8 images (4 train, 4 val) in a 1.5 MB auto-downloading archive, it verifies the full train / val / predict cycle — paired depth-map loading, augmentation, loss computation, and metrics — in seconds, before scaling to a full dataset such as [SUN RGB-D](sunrgbd.md) or [NYU Depth V2](nyu-depth-v2.md).

### How does Depth8 differ from the full SUN RGB-D dataset?

Depth8 samples 8 images from SUN RGB-D's 9,245-train/1,090-val split, keeping one image per RGB-D sensor per split. It uses the identical directory layout and `.npy` depth format, so a pipeline that runs on Depth8 runs unmodified on the full dataset — just point `data=` at `depth-sunrgbd.yaml` instead of `depth8.yaml`. Unlike the full dataset, Depth8 downloads in seconds and needs no conversion step.

### Should I use Depth8 for benchmarking?

No. Depth8 is too small for meaningful model comparison and is intended for training and evaluation pipeline checks. Use the full [NYU Depth V2](nyu-depth-v2.md) or [SUN RGB-D](sunrgbd.md) validation sets when you need representative depth estimation metrics.
