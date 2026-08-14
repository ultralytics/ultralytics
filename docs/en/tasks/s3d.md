---
comments: true
description: Learn how to use YOLO26 for stereo 3D object detection. Train, validate, and predict 3D bounding boxes from stereo image pairs with KITTI evaluation.
keywords: stereo 3D detection, YOLO26, Ultralytics, 3D object detection, KITTI, depth estimation, stereo vision, autonomous driving
---

# Stereo 3D Detection

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/s3d-examples.avif" alt="YOLO26 stereo 3D detection with 3D wireframe bounding boxes on KITTI driving scenes">

Stereo 3D detection is a computer vision task that estimates full 3D bounding boxes — including depth, dimensions, and orientation — from calibrated stereo image pairs. Unlike standard 2D detection which only produces flat bounding boxes, stereo 3D detection recovers the spatial geometry of objects in the scene by leveraging the disparity between left and right camera views.

The output of a stereo 3D detection model includes a 3D center location `[x, y, z]` in camera coordinates, physical dimensions `[length, width, height]` in meters, and a rotation angle around the vertical axis. This makes it essential for autonomous driving and robotics applications where precise spatial understanding is required.

!!! tip

    YOLO26 _stereo 3D detection_ models use the `-s3d` suffix, i.e., `yolo26s-s3d.pt`. These models are trained on the [KITTI Stereo](../datasets/detect/kitti-stereo.md) dataset and use a siamese backbone that processes left and right images through shared weights.

    The siamese architecture splits standard 3-channel input into separate left/right streams, enabling 100% compatibility with pretrained YOLO26 backbone weights. A stereo cost volume module fuses the two views to estimate depth, while auxiliary prediction heads output 3D dimensions, orientation, and lateral distance.

## Models

Ultralytics YOLO26 Stereo 3D Detection models use a siamese backbone over the [KITTI Stereo](../datasets/detect/kitti-stereo.md) dataset. The table below reports the only configuration measured on a **drive-disjoint** split; see the note beneath it before comparing against any other stereo-3D number.

| Model                                                                                                         | Params | Car AP3D@0.5 (E/M/H)   | Car AP3D@0.7 (E/M/H)   | mAP3D@0.5 (Mod) | mAP3D@0.7 (Mod) |
| ------------------------------------------------------------------------------------------------------------- | ------ | ---------------------- | ---------------------- | --------------- | --------------- |
| [YOLO26n-s3d](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26/yolo26-s3d.yaml) | 4.4M   | 59.5 / **47.0** / 40.6 | 19.9 / **13.5** / 11.3 | 19.1            | 4.5             |
| [YOLO26s-s3d](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26/yolo26-s3d.yaml) | 12.8M  | 57.3 / **46.5** / 40.6 | 17.6 / **13.6** / 11.3 | 19.5            | 4.5             |

- These are the **published checkpoints** (the `v8.4.0` assets), evaluated on the held-out Chen `test` split (3769 images) via `yolo val ... data=kitti-stereo-chen.yaml split=test` with the shipped defaults. This is what you get from `YOLO("yolo26n-s3d.pt")` — not a locally trained reference.
- **Car** columns are the KITTI headline (Car at IoU 0.7 is the number published stereo-3D work leads with). **mAP3D** columns are the unweighted mean over Car/Pedestrian/Cyclist and are much lower because Cyclist AP is near zero on this split — the two are not interchangeable.
- Decoded with the shipped `score_k=2.5` confidence weighting.
- Cross-checked against a port of the official KITTI devkit on identical predictions: Ultralytics' R40 reads **0.15-0.65 AP lower** across all nine class/difficulty cells. The gap is mainly the devkit's neighbour-class don't-care rule, which needs the `Van`/`Person_sitting` boxes this 3-class dataset does not carry. These figures are a floor, not an optimistic reading.
- `n` and `s` land within ~0.5 AP of each other, so the released `s` is not currently buying its extra capacity. For reference, an `s` trained from scratch for 400 epochs on Chen's full 3712-frame training set (`train` **and** `dev`, not the shipped 3172-frame `train`) reaches **54.0 / 20.1** Car Moderate. Both are leakage-free — `test` is never seen either way — so the gap is training budget, not hygiene and not architecture: the published models hold `dev` out for model selection, which costs roughly 5.6 AP@0.5 and 3.1 AP@0.7 on this split. Training on `train` + `dev` is a legitimate way to recover it if you do not need per-epoch selection.
- `m`, `l` and `x` are pending re-benchmarking on the drive-disjoint split. Reproduce any row with `python -m ultralytics.data.scripts.benchmark_s3d --data kitti-stereo-chen.yaml --split test`.

!!! warning "Earlier published numbers for n/s/m/l/x were measured on a leaked split and have been removed"

    A previous version of this table reported 24.5-31.6% AP3D@0.7 (Mod) across five model sizes. Those runs used the `kitti-stereo.yaml` config, which split frames by a contiguous index cutoff and placed 128 of KITTI's 141 raw drives on **both** sides — 99.6% of its validation frames shared a drive with training. That config has since been removed from the repository. Re-measured on a drive-disjoint split, the same class of model scores roughly **5x lower** at AP3D@0.7, which is why only the row above remains.

    Numbers for `n`, `m`, `l` and `x` are pending re-benchmarking on the drive-disjoint split. Reproduce any of them with `python -m ultralytics.data.scripts.benchmark_s3d --data kitti-stereo-chen.yaml --split test`. **Do not restore the old figures.**

## Train

Train a YOLO26 stereo 3D detection model on the KITTI Stereo dataset.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26s-s3d.yaml")  # build a new model from YAML
        model = YOLO("yolo26s-s3d.pt")  # load a pretrained model (recommended)
        model = YOLO("yolo26s-s3d.yaml").load("yolo26s-s3d.pt")  # build and transfer weights

        # Train the model (quick-start with mini dataset)
        results = model.train(data="kitti-stereo8.yaml", epochs=5, imgsz=[384, 1248])

        # Full training on KITTI Stereo dataset
        # results = model.train(data="kitti-stereo-chen.yaml", epochs=1000, imgsz=[384, 1248], optimizer="SGD", lr0=0.01, cos_lr=True)
        ```

    === "CLI"

        ```bash
        # Quick-start with mini dataset (auto-downloads ~12 MB)
        yolo s3d train data=kitti-stereo8.yaml model=yolo26s-s3d.yaml epochs=5 imgsz=384,1248

        # Full training on KITTI Stereo dataset (~1.9 GB download)
        yolo s3d train data=kitti-stereo-chen.yaml model=yolo26s-s3d.pt epochs=1000 imgsz=384,1248 optimizer=SGD lr0=0.01 cos_lr=True
        ```

### Dataset format

The KITTI Stereo dataset format uses 18-value labels per object containing left/right 2D bounding boxes, 3D dimensions, 3D location, rotation, and truncation/occlusion metadata. See the [KITTI Stereo Dataset Guide](../datasets/detect/kitti-stereo.md) for full format details.

Training requires calibrated stereo pairs (left + right images) with a calibration file per frame providing the projection matrices needed for depth computation.

## Val

Validate a trained stereo 3D detection model using KITTI R40 evaluation protocol.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26s-s3d.pt")  # load a pretrained model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.results_dict["ap3d_50"]  # mean AP3D @ IoU=0.5 (Moderate)
        metrics.results_dict["ap3d_70"]  # mean AP3D @ IoU=0.7 (Moderate)
        metrics.results_dict["AP3D_Car_Mod_50"]  # per-class per-difficulty
        ```

    === "CLI"

        ```bash
        yolo s3d val model=yolo26s-s3d.pt  # val pretrained model
        yolo s3d val model=path/to/best.pt # val custom model
        ```

The KITTI R40 evaluation uses 40-point interpolated precision-recall curves with three difficulty levels:

- **Easy**: bbox height >= 40px, occlusion == 0, truncation <= 0.15
- **Moderate**: bbox height >= 25px, occlusion <= 1, truncation <= 0.30
- **Hard**: bbox height >= 25px, occlusion <= 2, truncation <= 0.50

The primary metric is **AP3D@0.5 (Moderate)** — the mean 3D Average Precision at IoU threshold 0.5 across all classes at Moderate difficulty.

## Predict

Use a trained stereo 3D detection model to predict 3D bounding boxes from stereo image pairs.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26s-s3d.pt")  # load a pretrained model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with a stereo pair (left, right)
        results = model([("left.jpg", "right.jpg")])

        # Access 3D detection results
        for result in results:
            for box in result.boxes3d:
                print(box.center_3d)  # (x, y, z) in camera coordinates (meters)
                print(box.dimensions)  # (length, width, height) in meters
                print(box.orientation)  # rotation around Y axis (radians)
        ```

    === "CLI"

        ```bash
        yolo s3d predict model=yolo26s-s3d.pt source='left.jpg,right.jpg'
        ```

Stereo prediction requires paired left/right images. In Python, pass a list of `(left_path, right_path)` tuples. In the CLI, use comma-separated paths.

## Export

Export a stereo 3D detection model to ONNX or TensorRT format. The exported model has **two inputs** (`left_img`, `right_img`) each with shape `[B, 3, H, W]`, and a single output tensor containing both 2D detections and 3D auxiliary predictions.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26s-s3d.pt")  # load a pretrained model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Export to ONNX
        model.export(format="onnx", imgsz=[384, 1248])

        # Export to TensorRT (requires CUDA)
        model.export(format="engine", imgsz=[384, 1248])
        ```

    === "CLI"

        ```bash
        yolo s3d export model=yolo26s-s3d.pt format=onnx imgsz=384,1248
        yolo s3d export model=path/to/best.pt format=engine imgsz=384,1248
        ```

The exported ONNX model can be used directly with ONNX Runtime:

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("yolo26s-s3d.onnx")
left = np.random.randn(1, 3, 384, 1248).astype(np.float32)
right = np.random.randn(1, 3, 384, 1248).astype(np.float32)
output = sess.run(None, {"left_img": left, "right_img": right})[0]
# output shape: [1, nc+4+74, anchors] where 74 = 1(lr_dist) + 3(dims) + 6(orient) + 64(depth_bins)
# orient is MultiBin: NUM_ORIENT_BINS*(1 conf + 2 residual) = 6. depth_bins follows the model YAML's
# `training: depth_bins:` (default 64), so re-check this width if you override it.
```

## FAQ

### What is stereo 3D detection and how does it differ from standard object detection?

Standard object detection produces 2D bounding boxes `[x, y, width, height]` in pixel coordinates. Stereo 3D detection goes further by estimating full 3D bounding boxes including the object's physical location in 3D space `[x, y, z]`, real-world dimensions `[length, width, height]` in meters, and orientation (rotation angle). It achieves this by processing calibrated stereo image pairs and leveraging the disparity between left and right views to estimate depth.

### How do I train a stereo 3D detection model on a custom dataset?

Your custom dataset needs calibrated stereo image pairs (left + right cameras), calibration files with projection matrices, and 18-value labels per object. Follow the [KITTI Stereo format](../datasets/detect/kitti-stereo.md), then train:

```python
from ultralytics import YOLO

model = YOLO("yolo26s-s3d.yaml")
results = model.train(data="your-stereo-dataset.yaml", epochs=1000, imgsz=[384, 1248], optimizer="SGD", cos_lr=True)
```

Use rectangular `imgsz` matching your camera's aspect ratio. A long schedule matters: the benchmarked model used 400 epochs, and shorter schedules (e.g., 200 epochs) may not converge fully.

### What metrics does KITTI R40 evaluation use?

KITTI R40 evaluation computes 3D Average Precision (AP3D) using 40-point interpolated precision-recall curves. Results are reported at three difficulty levels (Easy, Moderate, Hard) based on object size, occlusion, and truncation. The standard IoU thresholds are 0.5 and 0.7 for 3D bounding box overlap. The primary benchmark metric is **AP3D@0.5 at Moderate difficulty**, averaged across all evaluated classes.

### What pretrained stereo 3D detection models are available?

YOLO26 siamese stereo models are configured in sizes n/s/m/l/x via the `-s3d` suffix. `YOLO26n-s3d` and `YOLO26s-s3d` are published and benchmarked on a drive-disjoint split (47.0 and 46.5 Car AP3D@0.5 Moderate respectively). `m`, `l` and `x` are pending re-benchmarking — earlier figures for them came from a split with 99.6% drive leakage and were withdrawn. See the [Models section](#models) for the details and how to reproduce.
