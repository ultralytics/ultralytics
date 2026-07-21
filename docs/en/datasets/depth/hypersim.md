---
comments: true
description: Explore the Hypersim depth dataset for monocular depth estimation, a photorealistic synthetic indoor dataset with dense per-pixel ground truth used to train Ultralytics YOLO26-Depth models.
keywords: Ultralytics, YOLO, depth estimation, Hypersim, synthetic indoor dataset, photorealistic, dense depth, monocular depth
---

# Hypersim Depth Dataset

[Hypersim](https://github.com/apple/ml-hypersim) is a photorealistic synthetic dataset of indoor scenes designed for holistic indoor scene understanding. Its images are ray-traced from professionally created Evermotion 3D interiors, producing highly realistic renderings paired with perfect, dense per-pixel depth ground truth.

Because Hypersim is fully synthetic, every pixel has an exact depth value with no sensor noise, missing returns, or measurement artifacts. This makes it an excellent source of clean, dense indoor geometry for training monocular [depth estimation](https://www.ultralytics.com/glossary/depth-estimation) models.

## Key Features

- Photorealistic **synthetic** indoor scenes, ray-traced from professional Evermotion 3D interiors.
- **Indoor** environments only.
- Dense, perfect **per-pixel depth** ground truth with no sensor noise or missing values.
- Depth range mostly **≤10 m** (with some larger distances).
- Contributes **74,619** images (68,242 train / 6,377 val) to the Ultralytics depth training mix.

## Dataset Structure

The Hypersim depth dataset is split into two subsets:

1. **Train**: 68,242 images with paired dense depth maps for training.
2. **Val**: 6,377 images with paired dense depth maps for validation during training.

Each RGB image is paired with a `.npy` float32 depth map storing per-pixel distances in meters, following the [Ultralytics depth dataset format](index.md).

## Obtain the Data

Hypersim has no autodownload — the dataset (~1.9 TB of per-scene ZIPs) is distributed by Apple under the [CC BY-SA 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/) and downloaded with the official script from the [ml-hypersim repository](https://github.com/apple/ml-hypersim) (a selective downloader is available in `contrib/99991`):

```bash
git clone https://github.com/apple/ml-hypersim && cd ml-hypersim
python code/python/tools/dataset_download_images.py --downloads_dir ./downloads --decompress_dir ./scenes
```

RGB frames are the `frame.*.tonemap.jpg` previews and depth lives in `frame.*.depth_meters.hdf5`. The stored values are **ray distances to the camera center, not planar depth** — convert before saving, and map the NaN pixels (windows, sky) to 0 (invalid). Use the official `metadata_images_split_scene_v1.csv` scene split for train/val assignment. Reference conversion to the [Ultralytics depth dataset format](index.md):

```python
import shutil
from pathlib import Path

import h5py
import numpy as np

W, H, FOCAL = 1024, 768, 886.81  # Hypersim camera intrinsics
x, y = np.meshgrid(np.linspace(-W / 2, W / 2, W), np.linspace(-H / 2, H / 2, H))
ray2plane = (FOCAL / np.sqrt(x**2 + y**2 + FOCAL**2)).astype(np.float32)  # distance → planar depth

src, dst = Path("scenes"), Path("datasets/depth-hypersim")
out = "train"  # assign per scene from metadata_images_split_scene_v1.csv
(dst / f"images/{out}").mkdir(parents=True, exist_ok=True)
(dst / f"depth/{out}").mkdir(parents=True, exist_ok=True)
for h5 in sorted(src.rglob("*.depth_meters.hdf5")):
    scene, cam, frame = h5.parts[-4], h5.parent.name.split("_geometry")[0], h5.name.split(".")[1]
    rgb = h5.parents[1] / f"{cam}_final_preview" / f"frame.{frame}.tonemap.jpg"
    dist = np.asarray(h5py.File(h5)["dataset"], np.float32)
    depth = np.nan_to_num(dist * ray2plane, nan=0.0)  # NaN (sky, glass) → 0 = invalid
    name = f"{scene}_{cam}_{frame}"
    np.save(dst / f"depth/{out}/{name}.npy", depth)
    shutil.copy(rgb, dst / f"images/{out}/{name}.jpg")
```

## Role in YOLO26-Depth

Hypersim is one of the **training sources** in the broad multi-dataset mixture (~2.19M images) used to pretrain the Ultralytics YOLO26-Depth models. Within this mix, Hypersim contributes clean, dense indoor geometry that complements noisier real-world sensor data.

There is no standalone held-out Hypersim benchmark in this setup. Instead, the resulting models are evaluated on the standard monocular depth benchmarks: NYU Depth V2, KITTI, Make3D, ETH3D, and iBims-1.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. For Hypersim, the `depth-hypersim.yaml` file defines the paths and the single `depth` class.

!!! example "ultralytics/cfg/datasets/depth-hypersim.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/depth-hypersim.yaml"
    ```

## Usage

To train a YOLO26n-Depth model on the Hypersim dataset with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-depth.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="depth-hypersim.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo depth train data=depth-hypersim.yaml model=yolo26n-depth.pt epochs=100 imgsz=640
        ```

## Pretrained Models

The YOLO26 depth family (`yolo26n-depth.pt`, `yolo26s-depth.pt`, `yolo26m-depth.pt`, `yolo26l-depth.pt`, `yolo26x-depth.pt`) auto-downloads from the [v8.4.0 release](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-depth.pt) and is trained on the broad multi-dataset mix that Hypersim is part of.

## Citations and Acknowledgments

If you use the Hypersim dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{roberts2021hypersim,
              title={Hypersim: A Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding},
              author={Mike Roberts and Jason Ramapuram and Anurag Ranjan and Atulit Kumar and Miguel Angel Bautista and Nathan Paczan and Russ Webb and Joshua M. Susskind},
              booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
              year={2021}
        }
        ```

We would like to acknowledge the creators of Hypersim for making this photorealistic synthetic indoor dataset available to the computer vision community.
