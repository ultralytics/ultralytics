---
comments: true
description: Learn how to prepare depth estimation datasets for Ultralytics YOLO, including 16-bit PNG depth maps, dataset YAML fields, directory layout, and supported datasets.
keywords: Ultralytics, YOLO, depth estimation, depth dataset format, PNG depth maps, NYU Depth V2, monocular depth, per-pixel depth
---

# Depth Estimation Datasets Overview

Monocular depth estimation assigns a depth value in meters to every pixel in an image. The training target is a self-describing 16-bit grayscale PNG that stores its linear meter range in PNG metadata.

This guide explains the dataset format used by Ultralytics YOLO depth estimation models and lists the built-in dataset configurations available for training and validation.

## Supported Dataset Format

### PNG depth map format

Each training sample consists of one RGB image and one paired `.png` depth file. Code `0` means invalid; writers use codes `256–65535` to linearly cover the valid per-image minimum and maximum stored in the PNG metadata. Starting at 256 preserves the nearest valid band when browsers display the same 16-bit asset as 8-bit.

- Depth files use the `.png` extension and the Ultralytics `linear-u16` metadata convention.
- Each depth file should have the same stem as its matching image file (e.g., `scene_001.png` pairs with `scene_001.jpg`).
- The dataset loader finds depth files by replacing the `images` directory component with `depth` and swapping the image extension for `.png`.
- Pixels with depth `≤ 0` are treated as invalid and excluded from loss and metric computation.

The standard layout keeps images and depth maps in parallel folders:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── depth/
    ├── train/
    └── val/
```

For example, an image at `images/train/scene_001.jpg` is paired with a depth map at `depth/train/scene_001.png`.

### Dataset YAML format

Depth estimation datasets are configured with YAML files. The main fields are:

| Key     | Description                                                    |
| ------- | -------------------------------------------------------------- |
| `path`  | Dataset root directory.                                        |
| `train` | Training image path relative to `path`, or an absolute path.   |
| `val`   | Validation image path relative to `path`, or an absolute path. |
| `test`  | Optional test image path.                                      |
| `nc`    | Number of classes — always `1` for depth estimation.           |
| `names` | Class name mapping — always `{0: depth}`.                      |

!!! example "ultralytics/cfg/datasets/nyu-depth.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/nyu-depth.yaml"
    ```

## Usage

Train a YOLO26 depth estimation model with Python or CLI:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained depth model
        model = YOLO("yolo26n-depth.pt")

        # Train on the NYU Depth V2 dataset
        results = model.train(data="nyu-depth.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo depth train data=nyu-depth.yaml model=yolo26n-depth.pt epochs=100 imgsz=640
        ```

## Supported Datasets

The YOLO26 depth models are pretrained on a broad multi-dataset mix (~2.19M images) spanning indoor (≤10 m) to outdoor (~80 m) ranges, then evaluated zero-shot across five benchmarks. Each dataset has a dedicated page:

**Debugging**

- [Depth8](depth8.md) — 8 SUN RGB-D images in a 1.3 MB auto-downloading archive, for rapid pipeline testing

**Pretraining sources**

- [ARKitScenes](arkitscenes.md) — real indoor, Apple ARKit LiDAR (largest real source)
- [SUN RGB-D](sunrgbd.md) — real indoor, multi-sensor RGB-D
- [DIODE](diode.md) — real indoor + outdoor, dense laser-scanner ground truth
- [Hypersim](hypersim.md) — synthetic photorealistic indoor
- [TartanAir](tartanair.md) — synthetic, diverse environments
- [Virtual KITTI 2](vkitti2.md) — synthetic outdoor driving
- [KITTI](kitti.md) — real outdoor driving, Velodyne LiDAR (also an evaluation benchmark)
- [ImageNet (pseudo-labeled)](imagenet-pseudo.md) — pseudo-labeled distillation set, the largest single source

**Evaluation benchmarks**

- [NYU Depth V2](nyu-depth-v2.md) — primary indoor benchmark
- [KITTI Eigen](kitti.md) — outdoor driving benchmark
- [ETH3D](eth3d.md) — high-precision indoor + outdoor
- [Make3D](make3d.md) — outdoor, out-of-distribution
- [iBims-1](ibims-1.md) — high-quality indoor (edges and planar surfaces)

Per-model accuracy on these benchmarks and the downloadable pretrained weights are listed on the [Depth Estimation task page](../../tasks/depth.md#models). A dataset YAML whose `train` field lists multiple image directories combines these sources for large-scale mixed training.

## Adding Your Own Dataset

1. Save RGB images under split folders such as `images/train` and `images/val`.
2. Save one 16-bit depth PNG per image under the matching `depth/train` and `depth/val` folders using the same file stem as the image. Use `save_depth_png()` to write the required metadata.
3. Ensure depth values are in meters and that invalid or missing pixels use `0` or negative values.
4. Create a dataset YAML with `path`, `train`, `val`, `nc: 1`, and `names: {0: depth}`.

```yaml
path: path/to/my-depth-dataset
train: images/train
val: images/val

nc: 1
names:
    0: depth
```

## FAQ

### What file format should depth maps use?

Depth maps must be self-describing 16-bit PNGs written with `ultralytics.data.utils.save_depth_png()`. The loader reconstructs meter-valued float32 arrays and preserves code `0` as invalid.

### How are invalid depth pixels handled?

Pixels with depth values `≤ 0` are treated as invalid and masked out from both loss computation and metric evaluation. This covers sensor noise, sky regions, and reflective surfaces where depth cannot be reliably measured.

### What metrics are used for evaluation?

Depth estimation validation reports the standard Depth Anything metric set:

- **delta1 / delta2 / delta3** — percentage of pixels within 1.25×, 1.25²×, 1.25³× thresholds. Higher is better.
- **abs_rel** — mean absolute relative error. Lower is better.
- **rmse** — root mean squared error in meters. Lower is better.
- **silog** — scale-invariant logarithmic error. Lower is better.

### Do depth file names need to match image file names?

Yes. Each depth `.png` file must share the same stem as the corresponding image. The loader derives the depth path by replacing the `images` directory component with `depth` and substituting the image extension for `.png`. Images whose depth file is missing or unreadable are dropped during the cached dataset scan with a warning.
