---
comments: true
description: Explore the iBims-1 indoor benchmark for monocular depth estimation. Learn about its structure, usage, pretrained models, and role as a YOLO26-Depth zero-shot evaluation benchmark.
keywords: Ultralytics, YOLO, depth estimation, iBims-1, indoor depth, laser scanner, monocular depth, depth edges, planar surfaces
---

# iBims-1 Depth Dataset

[iBims-1](https://www.asg.ed.tum.de/lmf/ibims1/) (independent Benchmark images and matched scans) is a high-quality indoor benchmark for [monocular depth estimation](index.md). It pairs RGB images with survey-grade laser-scanner ground truth and is designed to test sharp depth edges and planar surfaces.

## Key Features

- High-quality depth ground truth captured with a survey-grade laser scanner.
- Covers real **indoor** scenes.
- Depth range up to approximately 10 m.
- Evaluation is performed on 100 images.
- Specifically designed to evaluate sharp depth edges and the quality of planar surfaces.

## Role in YOLO26-Depth

iBims-1 is a **zero-shot evaluation benchmark** for the YOLO26-Depth family; the published models are not trained on it. Its focus on sharp depth discontinuities and planar regions makes it a precise test of depth-map quality beyond aggregate accuracy.

Evaluation uses multi-scale and horizontal-flip test-time augmentation (TTA), followed by log-least-squares scale alignment between the predicted and ground-truth depth maps before metrics are computed.

## Results

The table below reports the `delta1` accuracy (percentage of pixels within a 1.25× threshold, higher is better) on the iBims-1 evaluation images by model size.

| Model        | delta1 |
| ------------ | ------ |
| YOLO26n-depth | pending (in training) |
| YOLO26s-depth | 0.899 |
| YOLO26m-depth | 0.953 |
| YOLO26l-depth | 0.952 |
| YOLO26x-depth | 0.961 |

## Evaluation

iBims-1 is not shipped with a bundled dataset YAML. It is evaluated through a dedicated evaluation script that loads the iBims-1 images and laser-scanner depth ground truth, applies the standard TTA and log-least-squares scale alignment, and reports the depth metrics.

## Usage

iBims-1 is an external benchmark, so models are typically run with `predict` on its images. For a comprehensive list of available arguments, refer to the model [Prediction](../../modes/predict.md) page.

!!! example "Predict Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26x-depth.pt")  # load a pretrained depth model

        # Predict depth on iBims-1 images
        results = model.predict("path/to/ibims1/images")
        ```

    === "CLI"

        ```bash
        # Predict depth with a pretrained *.pt model
        yolo depth predict model=yolo26x-depth.pt source=path/to/ibims1/images
        ```

## Pretrained Models

The YOLO26 depth family is evaluated zero-shot on the iBims-1 benchmark. These models auto-download from the latest Ultralytics release, for example [YOLO26x-depth](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-depth.pt) from v8.4.0, and span a range of sizes (yolo26n/s/m/l/x-depth) for different accuracy and resource requirements.

## Citations and Acknowledgments

If you use the iBims-1 dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{koch2018ibims,
              title={Evaluation of CNN-based Single-Image Depth Estimation Methods},
              author={Koch, Tobias and Liebel, Lukas and Fraundorfer, Friedrich and K{\"o}rner, Marco},
              booktitle={Proceedings of the European Conference on Computer Vision (ECCV) Workshops},
              year={2018}
        }
        ```

We would like to acknowledge the authors for creating and maintaining this valuable resource for the computer vision community.
