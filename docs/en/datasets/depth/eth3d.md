---
comments: true
description: Explore the ETH3D high-resolution benchmark for monocular depth estimation. Learn about its structure, usage, pretrained models, and role as a YOLO26-Depth zero-shot evaluation benchmark.
keywords: Ultralytics, YOLO, depth estimation, ETH3D, multi-view stereo, laser scanner, monocular depth, indoor outdoor benchmark
---

# ETH3D Depth Dataset

[ETH3D](https://www.eth3d.net/) is a high-resolution multi-view-stereo benchmark used for [monocular depth estimation](index.md). It provides survey-grade laser-scanner ground truth across both indoor and outdoor scenes.

## Key Features

- Survey-grade depth ground truth captured with a high-precision laser scanner.
- High-resolution images covering both **indoor and outdoor** scenes.
- Depth range up to approximately 60 m.
- Evaluation is performed on 423 images.
- A high-quality multi-view-stereo benchmark with accurate dense ground truth.

## Role in YOLO26-Depth

ETH3D is a **zero-shot evaluation benchmark** for the YOLO26-Depth family; the published models are not trained on it. Its mix of indoor and outdoor scenes with accurate laser-scanner ground truth makes it a strong test of cross-domain generalization.

Evaluation uses multi-scale and horizontal-flip test-time augmentation (TTA), followed by log-least-squares scale alignment between the predicted and ground-truth depth maps before metrics are computed.

## Results

The table below reports the `delta1` accuracy (percentage of pixels within a 1.25× threshold, higher is better) on the ETH3D evaluation images by model size.

| Model         | delta1 |
| ------------- | ------ |
| YOLO26n-depth | 0.905  |
| YOLO26s-depth | 0.876  |
| YOLO26m-depth | 0.943  |
| YOLO26l-depth | 0.945  |
| YOLO26x-depth | 0.953  |

## Evaluation

ETH3D is not shipped with a bundled dataset YAML. It is evaluated through a dedicated evaluation script that loads the ETH3D images and laser-scanner depth ground truth, applies the standard TTA and log-least-squares scale alignment, and reports the depth metrics.

## Usage

ETH3D is an external benchmark, so models are typically run with `predict` on its images. For a comprehensive list of available arguments, refer to the model [Prediction](../../modes/predict.md) page.

!!! example "Predict Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26x-depth.pt")  # load a pretrained depth model

        # Predict depth on ETH3D images
        results = model.predict("path/to/eth3d/images")
        ```

    === "CLI"

        ```bash
        # Predict depth with a pretrained *.pt model
        yolo depth predict model=yolo26x-depth.pt source=path/to/eth3d/images
        ```

## Pretrained Models

The YOLO26 depth family is evaluated zero-shot on the ETH3D benchmark. These models auto-download from the latest Ultralytics release, for example [YOLO26x-depth](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-depth.pt) from v8.4.0, and span a range of sizes (yolo26n/s/m/l/x-depth) for different accuracy and resource requirements.

## Citations and Acknowledgments

If you use the ETH3D dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{schops2017eth3d,
              title={A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera Videos},
              author={Sch{\"o}ps, Thomas and Sch{\"o}nberger, Johannes L. and Galliani, Silvano and Sattler, Torsten and Schindler, Konrad and Pollefeys, Marc and Geiger, Andreas},
              booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
              year={2017}
        }
        ```

We would like to acknowledge the authors for creating and maintaining this valuable resource for the computer vision community.
