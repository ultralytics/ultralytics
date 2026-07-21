---
comments: true
description: Explore the ARKitScenes depth dataset for monocular depth estimation. Learn about its structure, usage, pretrained models, and role in YOLO26-Depth training.
keywords: Ultralytics, YOLO, depth estimation, ARKitScenes, indoor RGB-D, LiDAR, monocular depth, depth dataset
---

# ARKitScenes Depth Dataset

[ARKitScenes](https://github.com/apple/ARKitScenes) is a large-scale real-world indoor RGB-D dataset captured with Apple's ARKit on iPad Pro devices equipped with a LiDAR scanner. It is the largest real-world indoor RGB-D dataset of its kind, providing diverse, naturally captured indoor scenes with accurate depth ground truth for [monocular depth estimation](index.md).

## Key Features

- Captured with the LiDAR scanner and RGB camera of Apple's ARKit on iPad Pro, producing real (non-synthetic) sensor data.
- Covers diverse **indoor** scenes for 3D indoor scene understanding.
- Short depth range, approximately 0.5–6 m (median maximum depth around 2.4 m), typical of handheld indoor capture.
- Dense LiDAR-derived depth ground truth aligned to the RGB frames.
- The single largest real-world source in the Ultralytics depth pretraining mix.

## Dataset Structure

The ARKitScenes depth dataset is split into two subsets:

1. **Train**: 676,080 images with paired depth maps for training.
2. **Val**: 21,559 images with paired depth maps for validation during model training.

Each sample consists of one RGB image and one paired `.npy` float32 depth map storing per-pixel distances in meters, following the [Ultralytics depth dataset format](index.md).

## Obtain the Data

ARKitScenes has no autodownload — the data is distributed by Apple, and downloading requires accepting the license terms in the [ARKitScenes repository](https://github.com/apple/ARKitScenes). Clone the repository and download the depth-upsampling subset, which pairs RGB frames with registered depth:

```bash
git clone https://github.com/apple/ARKitScenes && cd ARKitScenes
python3 download_data.py upsampling --split Training --download_dir ./data
python3 download_data.py upsampling --split Validation --download_dir ./data
```

Depth frames are `uint16` PNGs in **millimeters** (0 = invalid), and RGB/depth filenames are capture timestamps that do not match exactly — pair each depth frame with the nearest-timestamp RGB frame. Reference conversion to the [Ultralytics depth dataset format](index.md):

```python
from pathlib import Path

import cv2
import numpy as np

src, dst = Path("data/upsampling"), Path("datasets/depth-arkitscenes")
for split, out in (("Training", "train"), ("Validation", "val")):
    (dst / f"images/{out}").mkdir(parents=True, exist_ok=True)
    (dst / f"depth/{out}").mkdir(parents=True, exist_ok=True)
    for video in sorted((src / split).iterdir()):
        rgbs = {float(p.stem.split("_")[-1]): p for p in (video / "lowres_wide").glob("*.png")}
        if not rgbs:
            continue
        for depth_png in sorted((video / "lowres_depth").glob("*.png")):
            t = float(depth_png.stem.split("_")[-1])
            rgb = rgbs[min(rgbs, key=lambda k: abs(k - t))]  # nearest-timestamp RGB frame
            name = f"{video.name}_{depth_png.stem}"
            depth = cv2.imread(str(depth_png), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # mm → m
            np.save(dst / f"depth/{out}/{name}.npy", depth)
            cv2.imwrite(str(dst / f"images/{out}/{name}.png"), cv2.imread(str(rgb)))
```

## Role in YOLO26-Depth

ARKitScenes is a **training** source in the Ultralytics YOLO26-Depth multi-dataset pretraining mix of roughly 2.19M image–depth pairs. As the single largest real source in this mix, it supplies abundant real-world indoor LiDAR depth that anchors the model's short-range indoor accuracy. The resulting models are evaluated on the standard NYU, KITTI, Make3D, ETH3D, and iBims-1 benchmarks.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information.

!!! example "ultralytics/cfg/datasets/depth-arkitscenes.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/depth-arkitscenes.yaml"
    ```

## Usage

To train a YOLO26n-depth model on the ARKitScenes dataset with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-depth.pt")  # load a pretrained depth model (recommended for training)

        # Train the model
        results = model.train(data="depth-arkitscenes.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo depth train data=depth-arkitscenes.yaml model=yolo26n-depth.pt epochs=100 imgsz=640
        ```

## Pretrained Models

The YOLO26 depth family is trained on the broad multi-dataset depth pretraining mix that ARKitScenes is part of. These models auto-download from the latest Ultralytics release, for example [YOLO26x-depth](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-depth.pt) from v8.4.0, and span a range of sizes for different accuracy and resource requirements.

## Citations and Acknowledgments

If you use the ARKitScenes dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{baruch2021arkitscenes,
              title={ARKitScenes: A Diverse Real-World Dataset for 3D Indoor Scene Understanding Using Mobile RGB-D Data},
              author={Baruch, Gilad and Chen, Zhuoyuan and Dehghan, Afshin and Dimry, Tal and Feigin, Yuri and Fu, Peter and Gebauer, Thomas and Joffe, Brandon and Kurz, Daniel and Schwartz, Arik and Shulman, Elad},
              booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
              year={2021}
        }
        ```

We would like to acknowledge the authors for creating and maintaining this valuable resource for the computer vision community.
