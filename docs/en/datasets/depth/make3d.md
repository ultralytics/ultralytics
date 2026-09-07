---
comments: true
license:
    name: Other
    url: http://make3d.cs.cornell.edu/data.html
description: Explore the Make3D outdoor benchmark for monocular depth estimation. Learn about its structure, usage, pretrained models, and role as a YOLO26-Depth zero-shot generalization benchmark.
keywords: Ultralytics, YOLO, depth estimation, Make3D, outdoor depth, laser scanner, monocular depth, out-of-distribution benchmark
---

# Make3D Depth Dataset

[Make3D](http://make3d.cs.cornell.edu/data.html) is a classic outdoor benchmark for [monocular depth estimation](index.md). It contains images of campus scenes paired with depth ground truth captured by a custom 3D laser scanner, and is widely used to probe out-of-distribution generalization.

## Key Features

- Depth ground truth captured with a custom 3D laser scanner.
- Covers real **outdoor** campus scenes.
- Depth range up to approximately 70 m.
- Evaluation is performed on the standard test set of 134 images.
- A classic out-of-distribution generalization benchmark.

## Role in YOLO26-Depth

Make3D is a **zero-shot evaluation benchmark** for the YOLO26-Depth family; the published models are not trained on it. As an out-of-distribution outdoor set, it is the hardest benchmark for all models, and absolute `delta1` values are low across the board, which is expected for this dataset.

Evaluation uses multi-scale and horizontal-flip test-time augmentation (TTA), followed by log-least-squares scale alignment between the predicted and ground-truth depth maps before metrics are computed.

## Results

The table below reports the `delta1` accuracy (percentage of pixels within a 1.25× threshold, higher is better) on the Make3D test set by model size.

| Model         | delta1 |
| ------------- | ------ |
| YOLO26n-depth | 0.307  |
| YOLO26s-depth | 0.311  |
| YOLO26m-depth | 0.293  |
| YOLO26l-depth | 0.297  |
| YOLO26x-depth | 0.299  |

## Evaluation

Make3D is not shipped with a bundled dataset YAML. It is evaluated through a dedicated evaluation script that loads the Make3D images and laser-scanner depth ground truth, applies the standard TTA and log-least-squares scale alignment, and reports the depth metrics.

## Usage

Make3D is an external benchmark, so models are typically run with `predict` on its images. For a comprehensive list of available arguments, refer to the model [Prediction](../../modes/predict.md) page.

!!! example "Predict Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26x-depth.pt")  # load a pretrained depth model

        # Predict depth on Make3D images
        results = model.predict("path/to/make3d/images")
        ```

    === "CLI"

        ```bash
        # Predict depth with a pretrained *.pt model
        yolo depth predict model=yolo26x-depth.pt source=path/to/make3d/images
        ```

## Pretrained Models

The YOLO26 depth family is evaluated zero-shot on the Make3D benchmark. These models auto-download from the latest Ultralytics release, for example [YOLO26x-depth](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x-depth) from v8.4.0, and span a range of sizes (yolo26n/s/m/l/x-depth) for different accuracy and resource requirements.

## Citations and Acknowledgments

If you use the Make3D dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{saxena2009make3d,
              title={Make3D: Learning 3D Scene Structure from a Single Still Image},
              author={Saxena, Ashutosh and Sun, Min and Ng, Andrew Y.},
              journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
              year={2009}
        }
        ```

We would like to acknowledge the authors for creating and maintaining this valuable resource for the computer vision community.

## FAQ

### What is Make3D used for in YOLO26-Depth?

Make3D is a zero-shot, out-of-distribution outdoor benchmark of 134 campus images with custom laser-scanner depth to roughly 70 m. The released YOLO26-Depth models are not trained on it, so it measures how well they generalize to unfamiliar outdoor scenes.

### Why are the Make3D delta1 scores so low?

Make3D is the hardest of the five evaluation benchmarks for every model because its scenes and depth statistics differ strongly from the training data. Absolute delta1 values around 0.3 are expected for this dataset, and the relative comparison between models is what matters.

### Is there a dataset YAML for Make3D?

No. Make3D is evaluated with a dedicated script using multi-scale and flip test-time augmentation and log-least-squares scale alignment. To run a model on Make3D images, use [predict mode](../../modes/predict.md) as shown in the [Usage](#usage) section.
