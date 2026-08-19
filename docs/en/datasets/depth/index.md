---
comments: true
description: Learn how to prepare depth estimation datasets for Ultralytics YOLO, including PNG and NPY depth maps, dataset YAML fields, directory layout, and supported datasets.
keywords: Ultralytics, YOLO, depth estimation, depth dataset format, PNG depth maps, NPY depth maps, NYU Depth V2, monocular depth, per-pixel depth
---

# Depth Estimation Datasets Overview

Monocular depth estimation assigns a depth value in meters to every pixel in an image. Depth targets use either scaled 16-bit grayscale PNGs or floating-point NPY arrays in meters.

This guide explains the dataset format used by Ultralytics YOLO depth estimation models and lists the built-in dataset configurations available for training and validation.

## Supported Dataset Format

### Depth map format

Each training sample consists of one RGB image and one paired depth file. PNG values are divided by the optional dataset-level `depth_scale` to produce meters. The default is `1000`, so a value of `1500` represents `1.5` meters. Code `0` means invalid; PNGs need no embedded metadata.

- Depth files use `.png` (preferred) or `.npy`. PNG maps must be 2D uint16 grayscale images. NPY arrays must be 2D and floating-point, with values in meters.
- Set `depth_scale` only when PNGs do not use the default millimeter convention. For example, KITTI uses `256` and Virtual KITTI 2 uses `100`.
- Each depth file should have the same stem as its matching image file (e.g., `scene_001.png` pairs with `scene_001.jpg`).
- Depth maps may be smaller than their RGB images as long as the aspect ratio matches; training resizes them in memory.
- The dataset loader finds depth files by replacing the `images` directory component with `depth`, preferring `.png` and falling back to `.npy`.
- Pixels with depth `≤ 0` are treated as invalid and excluded from loss and metric computation.

Because code `0` is reserved for invalid pixels, a uint16 PNG provides 65,535 positive depth values. The scale controls both precision and range:

| Convention            | `depth_scale` | Resolution |  Maximum depth |
| --------------------- | ------------: | ---------: | -------------: |
| Default / ARKitScenes |        `1000` |       1 mm |       65.535 m |
| KITTI                 |         `256` | 3.90625 mm | 255.99609375 m |
| Virtual KITTI 2       |         `100` |       1 cm |       655.35 m |

These are storage limits, not recommended training caps. A dataset can set a smaller `max_depth` independently for loss calibration or evaluation.

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

| Key           | Description                                                    |
| ------------- | -------------------------------------------------------------- |
| `path`        | Dataset root directory.                                        |
| `train`       | Training image path relative to `path`, or an absolute path.   |
| `val`         | Validation image path relative to `path`, or an absolute path. |
| `test`        | Optional test image path.                                      |
| `nc`          | Number of classes — always `1` for depth estimation.           |
| `names`       | Class name mapping — always `{0: depth}`.                      |
| `depth_scale` | Optional PNG units per meter; defaults to `1000`.              |

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
2. Save one depth PNG or NPY per image under the matching `depth/train` and `depth/val` folders using the same file stem as the image. Use `save_depth_png()` to convert meter arrays into compact millimeter PNGs.
3. Ensure the decoded depth values are in meters and that invalid or missing pixels use `0` or negative values.
4. Create a dataset YAML with `path`, `train`, `val`, `nc: 1`, and `names: {0: depth}`.

```yaml
path: path/to/my-depth-dataset
train: images/train
val: images/val

nc: 1
names:
    0: depth

# Optional: PNG integer units per meter (default 1000)
depth_scale: 1000
```

## FAQ

### What file format should depth maps use?

Use scaled 16-bit grayscale PNGs for compact datasets. By default each integer step is one millimeter; set `depth_scale` in the dataset YAML for another scale. `ultralytics.data.utils.save_depth_png()` converts meter arrays to the default format. Floating-point NPY maps in meters also work directly.

### How are invalid depth pixels handled?

Pixels with depth values `≤ 0` are treated as invalid and masked out from both loss computation and metric evaluation. This covers sensor noise, sky regions, and reflective surfaces where depth cannot be reliably measured.

### What metrics are used for evaluation?

Depth estimation validation reports the standard Depth Anything metric set:

- **delta1 / delta2 / delta3** — percentage of pixels within 1.25×, 1.25²×, 1.25³× thresholds. Higher is better.
- **abs_rel** — mean absolute relative error. Lower is better.
- **rmse** — root mean squared error in meters. Lower is better.
- **silog** — scale-invariant logarithmic error. Lower is better.

### Do depth file names need to match image file names?

Yes. Each depth `.png` or `.npy` file must share the same stem as the corresponding image. The loader derives the depth path by replacing the `images` directory component with `depth`, preferring PNG and falling back to NPY. Images whose depth file is missing or unreadable are dropped during the cached dataset scan with a warning.
