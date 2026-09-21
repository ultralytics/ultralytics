---
comments: true
license:
    name: Other
    url: https://github.com/apple/ARKitScenes/blob/main/LICENSE
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

Each sample consists of one RGB image and one paired uint16 depth PNG in millimeters (`depth_scale: 1000`), following the [Ultralytics depth dataset format](index.md). These counts are the subset used for the released models, built from 4,347 of the 4,520 Training videos and 102 of the 551 Validation videos; converting the complete splits yields more pairs.

## Obtain the Data

ARKitScenes has no autodownload — the data is distributed by Apple, and downloading requires accepting the license terms in the [ARKitScenes repository](https://github.com/apple/ARKitScenes). Clone the repository and download the `lowres_wide` (RGB) and `lowres_depth` assets of the **raw** subset, which covers the full 4,520-video Training and 551-video Validation splits. A video list is mandatory, and passing the split CSV without `--split` fetches both folds in one call:

```bash
git clone https://github.com/apple/ARKitScenes && cd ARKitScenes
python3 download_data.py raw --video_id_csv raw/raw_train_val_splits.csv \
  --raw_dataset_assets lowres_wide lowres_depth --download_dir ./data
```

Frames land in `data/raw/{Training,Validation}/<video_id>/{lowres_wide,lowres_depth}/`. Budget roughly 2.5 TB of disk: the raw streams run at ~60 FPS, and the conversion below keeps every 30th frame (~2 Hz).

Depth frames are `uint16` PNGs in **millimeters** (0 = invalid), and RGB/depth filenames are capture timestamps that do not match exactly — pair each RGB frame with the nearest-timestamp depth frame. Reference conversion to the [Ultralytics depth dataset format](index.md):

```python
import shutil
from bisect import bisect_left
from pathlib import Path

src, dst = Path("data/raw"), Path("datasets/depth-arkitscenes")
for split, out in (("Training", "train"), ("Validation", "val")):
    (dst / f"images/{out}").mkdir(parents=True, exist_ok=True)
    (dst / f"depth/{out}").mkdir(parents=True, exist_ok=True)
    for video in sorted((src / split).iterdir()):
        depths = sorted((video / "lowres_depth").glob("*.png"), key=lambda p: float(p.stem.split("_")[-1]))
        if not depths:
            continue
        times = [float(p.stem.split("_")[-1]) for p in depths]
        rgbs = sorted((video / "lowres_wide").glob("*.png"), key=lambda p: float(p.stem.split("_")[-1]))
        for rgb in rgbs[30::30]:  # every 30th frame of the ~60 FPS capture (~2 Hz)
            t = float(rgb.stem.split("_")[-1])
            i = min(bisect_left(times, t), len(times) - 1)
            if i and t - times[i - 1] < times[i] - t:
                i -= 1  # nearest-timestamp depth frame
            name = f"{out}_{video.name}_{depths[i].stem}"
            shutil.copy2(rgb, dst / f"images/{out}/{name}.png")
            shutil.copy2(depths[i], dst / f"depth/{out}/{name}.png")
```

## Role in YOLO26-Depth

ARKitScenes is a **training** source in the Ultralytics YOLO26-Depth multi-dataset pretraining mix of roughly 2.19M image–depth pairs. As the single largest real source in this mix, it supplies abundant real-world indoor LiDAR depth that anchors the model's short-range indoor accuracy. The resulting models are evaluated on the standard NYU, KITTI, Make3D, ETH3D, and iBims-1 benchmarks.

## Dataset YAML

A YAML file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information.

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

The YOLO26 depth family is trained on the broad multi-dataset depth pretraining mix that ARKitScenes is part of. These models auto-download from the latest Ultralytics release, for example [YOLO26x-depth](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x-depth) from v8.4.0, and span a range of sizes for different accuracy and resource requirements.

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

## FAQ

### What role does ARKitScenes play in YOLO26-Depth?

ARKitScenes is the single largest real-world source in the roughly 2.19M-image YOLO26-Depth pretraining mix, contributing 676,080 training and 21,559 validation images captured with the LiDAR scanner of Apple iPad Pro devices. Its dense indoor depth anchors the short-range accuracy of the released models, which are then evaluated on [NYU Depth V2](nyu-depth-v2.md), [KITTI](kitti.md), [ETH3D](eth3d.md), [Make3D](make3d.md), and [iBims-1](ibims-1.md).

### How do I download ARKitScenes for Ultralytics training?

ARKitScenes has no automatic download. Accept the license terms in the [ARKitScenes repository](https://github.com/apple/ARKitScenes), download the `lowres_wide` and `lowres_depth` assets of the raw subset with the official `download_data.py` script, then keep every 30th RGB frame and pair it with its nearest-timestamp depth frame using the conversion script in [Obtain the Data](#obtain-the-data). The result follows the standard [Ultralytics depth layout](index.md) with uint16 millimeter PNGs.

### What depth range does ARKitScenes cover?

ARKitScenes is a handheld indoor dataset with depths of roughly 0.5 to 6 m and a median maximum depth around 2.4 m, so it complements longer-range sources such as [KITTI](kitti.md) and [TartanAir](tartanair.md) in the training mix.
