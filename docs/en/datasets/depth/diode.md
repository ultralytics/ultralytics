---
comments: true
description: Explore the DIODE depth dataset for monocular depth estimation. Learn about its dense indoor/outdoor structure, usage, pretrained models, and role in YOLO26-Depth training.
keywords: Ultralytics, YOLO, depth estimation, DIODE, dense depth, indoor outdoor, FARO laser scanner, monocular depth, depth dataset
---

# DIODE Depth Dataset

[DIODE](https://diode-dataset.org/) (Dense Indoor/Outdoor DEpth) is a real-world dataset with very high-quality dense depth ground truth captured by a FARO Focus survey-grade laser scanner. Uniquely, it covers both indoor and outdoor scenes with the same sensor, making it a high-precision bridge between short-range indoor and long-range outdoor depth for [monocular depth estimation](index.md).

## Key Features

- Very high-quality **dense** depth ground truth from a FARO Focus survey-grade laser scanner.
- Covers **both indoor and outdoor** scenes captured with the same sensor.
- Depth range spans short indoor distances to long outdoor distances, up to approximately 80 m in the Ultralytics mix.
- Dense, accurate per-pixel ground truth aligned to the RGB frames.
- Provides high-precision dense ground truth that bridges indoor and outdoor domains.

## Dataset Structure

The DIODE depth dataset is split into two subsets:

1. **Train**: 25,458 images with paired depth maps for training.
2. **Val**: 771 images with paired depth maps for validation during model training.

Each sample consists of one RGB image and one paired `.npy` float32 depth map storing per-pixel distances in meters, following the [Ultralytics depth dataset format](index.md).

## Role in YOLO26-Depth

DIODE is a **training** source in the Ultralytics YOLO26-Depth multi-dataset pretraining mix of roughly 2.19M image–depth pairs. It contributes high-precision dense ground truth that bridges indoor and outdoor domains within a single sensor, helping the model generalize across both short-range and long-range scenes. The resulting models are evaluated on the standard NYU, KITTI, Make3D, ETH3D, and iBims-1 benchmarks.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information.

!!! example "ultralytics/cfg/datasets/depth-diode.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/depth-diode.yaml"
    ```

## Usage

To train a YOLO26n-depth model on the DIODE dataset with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-depth.pt")  # load a pretrained depth model (recommended for training)

        # Train the model
        results = model.train(data="depth-diode.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo depth train data=depth-diode.yaml model=yolo26n-depth.pt epochs=100 imgsz=640
        ```

## Pretrained Models

The YOLO26 depth family is trained on the broad multi-dataset depth pretraining mix that DIODE is part of. These models auto-download from the latest Ultralytics release, for example [YOLO26x-depth](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-depth.pt) from v8.4.0, and span a range of sizes for different accuracy and resource requirements.

## Citations and Acknowledgments

If you use the DIODE dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{vasiljevic2019diode,
              title={DIODE: A Dense Indoor and Outdoor DEpth Dataset},
              author={Vasiljevic, Igor and Kolkin, Nick and Zhang, Shanyi and Luo, Ruotian and Wang, Haochen and Dai, Falcon Z. and Daniele, Andrea F. and Mostajabi, Mohammadreza and Basart, Steven and Walter, Matthew R. and Shakhnarovich, Gregory},
              journal={arXiv preprint arXiv:1908.00463},
              year={2019}
        }
        ```

We would like to acknowledge the authors for creating and maintaining this valuable resource for the computer vision community.
