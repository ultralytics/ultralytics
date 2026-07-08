---
comments: true
description: Explore the NYU Depth V2 indoor benchmark for monocular depth estimation. Learn about its structure, usage, pretrained models, and role as the primary YOLO26-Depth evaluation benchmark.
keywords: Ultralytics, YOLO, depth estimation, NYU Depth V2, indoor RGB-D, Kinect, monocular depth, Eigen split, depth benchmark
---

# NYU Depth V2 Depth Dataset

[NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) is the standard indoor benchmark for [monocular depth estimation](index.md). It consists of RGB-D video sequences of a wide variety of indoor scenes recorded with a Microsoft Kinect v1. It is the primary benchmark used to report YOLO26-Depth accuracy.

## Key Features

- Captured with a Microsoft Kinect v1 RGB-D sensor.
- Covers a wide variety of real **indoor** scenes (homes, offices, classrooms, and similar spaces).
- Depth range up to approximately 10 m, typical of consumer indoor RGB-D capture.
- Evaluation is performed on the standard Eigen test split of 654 images.
- The primary benchmark for reporting monocular depth estimation accuracy.

## Role in YOLO26-Depth

NYU Depth V2 is the **primary zero-shot evaluation benchmark** for the YOLO26-Depth family, and the headline metrics on the depth task page are reported on it. The published YOLO26-Depth models are not trained on NYU; although the dataset includes a train split, it is left unused and only held-out test results are reported.

Evaluation uses multi-scale and horizontal-flip test-time augmentation (TTA), followed by log-least-squares scale alignment between the predicted and ground-truth depth maps before metrics are computed.

## Results

The table below reports the `delta1` accuracy (percentage of pixels within a 1.25× threshold, higher is better) on the NYU Depth V2 Eigen test split by model size.

| Model         | delta1 |
| ------------- | ------ |
| YOLO26n-depth | 0.882  |
| YOLO26s-depth | 0.855  |
| YOLO26m-depth | 0.919  |
| YOLO26l-depth | 0.927  |
| YOLO26x-depth | 0.923  |

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information.

!!! example "ultralytics/cfg/datasets/nyu-depth.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/nyu-depth.yaml"
    ```

## Usage

To evaluate a YOLO26-Depth model on the NYU Depth V2 benchmark, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Validation](../../modes/val.md) page.

!!! example "Validation Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26x-depth.pt")  # load a pretrained depth model

        # Evaluate on the NYU Depth V2 benchmark
        results = model.val(data="nyu-depth.yaml")
        ```

    === "CLI"

        ```bash
        # Evaluate a pretrained *.pt model
        yolo depth val data=nyu-depth.yaml model=yolo26x-depth.pt
        ```

## Pretrained Models

The YOLO26 depth family is evaluated zero-shot on the NYU Depth V2 benchmark. These models auto-download from the latest Ultralytics release, for example [YOLO26x-depth](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-depth.pt) from v8.4.0, and span a range of sizes (yolo26n/s/m/l/x-depth) for different accuracy and resource requirements.

## Citations and Acknowledgments

If you use the NYU Depth V2 dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{silberman2012indoor,
              title={Indoor Segmentation and Support Inference from RGBD Images},
              author={Silberman, Nathan and Hoiem, Derek and Kohli, Pushmeet and Fergus, Rob},
              booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
              year={2012}
        }
        ```

We would like to acknowledge the authors for creating and maintaining this valuable resource for the computer vision community.
