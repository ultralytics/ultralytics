---
comments: true
description: Learn how to use YOLO26 for stereo 3D object detection. Train, validate, and predict 3D bounding boxes from stereo image pairs with KITTI evaluation.
keywords: stereo 3D detection, YOLO26, Ultralytics, 3D object detection, KITTI, depth estimation, stereo vision, autonomous driving
---

# Stereo 3D Detection

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/s3d-examples.avif" alt="YOLO26 stereo 3D detection with 3D wireframe bounding boxes on KITTI driving scenes">

Stereo 3D detection is a computer vision task that estimates full 3D bounding boxes — including depth, dimensions, and orientation — from calibrated stereo image pairs. Unlike standard 2D detection which only produces flat bounding boxes, stereo 3D detection recovers the spatial geometry of objects in the scene by leveraging the disparity between left and right camera views.

The output of a stereo 3D detection model includes a 3D center location `[x, y, z]` in camera coordinates, physical dimensions `[height, width, length]` in meters, and a rotation angle around the vertical axis. This makes it essential for autonomous driving and robotics applications where precise spatial understanding is required.

!!! tip

    YOLO26 _stereo 3D detection_ models use the `-s3d` suffix, i.e., `yolo26s-s3d.pt`. These models are trained on the [KITTI Stereo](../datasets/detect/kitti-stereo.md) dataset and use a siamese backbone that processes left and right images through shared weights.

    The siamese architecture splits standard 3-channel input into separate left/right streams, enabling 100% compatibility with pretrained YOLO26 backbone weights. A stereo cost volume module fuses the two views to estimate depth, while auxiliary prediction heads output 3D dimensions, orientation, and lateral distance.

## Models

Ultralytics YOLO26 pretrained Stereo 3D Detection models are shown here. All models are trained on the [KITTI Stereo](../datasets/detect/kitti-stereo.md) dataset with SGD for 1000 epochs.

| Model                                                                                                         | Params | AP3D@0.5 (Mod) | AP3D@0.7 (Mod) |
| ------------------------------------------------------------------------------------------------------------- | ------ | -------------- | -------------- |
| [YOLO26n-s3d](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26/yolo26-s3d.yaml) | 3.6M   | 48.1%          | 29.9%          |
| [YOLO26s-s3d](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26/yolo26-s3d.yaml) | 11.6M  | 48.3%          | 29.4%          |
| [YOLO26m-s3d](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26/yolo26-s3d.yaml) | 26.8M  | 49.0%          | 29.1%          |
| [YOLO26l-s3d](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26/yolo26-s3d.yaml) | 31.2M  | 50.9%          | 31.6%          |
| [YOLO26x-s3d](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26/yolo26-s3d.yaml) | 69.9M  | 43.4%          | 24.5%          |

- **AP3D** values are KITTI R40 Moderate mean across Car/Pedestrian/Cyclist classes.
- All models trained with `imgsz=[384, 1248]`, SGD optimizer, cosine LR schedule for 1000 epochs.
- YOLO26l achieves the best accuracy. The x-size model overfits on KITTI's relatively small training set (3712 images).

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
        # results = model.train(data="kitti-stereo.yaml", epochs=1000, imgsz=[384, 1248], optimizer="SGD", lr0=0.01, cos_lr=True)
        ```

    === "CLI"

        ```bash
        # Quick-start with mini dataset (auto-downloads ~12 MB)
        yolo s3d train data=kitti-stereo8.yaml model=yolo26s-s3d.yaml epochs=5 imgsz=384,1248

        # Full training on KITTI Stereo dataset (~1.9 GB download)
        yolo s3d train data=kitti-stereo.yaml model=yolo26s-s3d.pt epochs=1000 imgsz=384,1248 optimizer=SGD lr0=0.01 cos_lr=True
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
# output shape: [1, nc+4+22, anchors] where 22 = 1(lr_dist) + 3(dims) + 2(orient) + 16(depth_bins)
```

## Auto-Labeling

Stereo 3D detection benefits from multi-class training data to prevent depth feature collapse. When training on datasets with only one annotated class (e.g., Car-only), the auto-labeling tool generates pseudo-labels for additional classes using a pretrained 2D YOLO detector.

!!! example

    ```bash
    python -m ultralytics.models.yolo.s3d.auto_label --data kitti-stereo.yaml
    ```

The auto-labeler:

1. Runs a YOLO 2D detector on both left and right images
2. Matches detections across views by class, scanline proximity, and size
3. Triangulates depth via `z = fx * baseline / disparity` for stereo-matched pairs
4. Writes 18-value pseudo-labels appended to existing label files

Pseudo-labels are marked with occlusion values to distinguish them: `occ=10` for stereo-matched labels (triangulated depth), `occ=20` for mono-only labels (estimated depth). Configure the pseudo-label loss weighting in your dataset YAML:

```yaml
pseudo_labels:
    weight: 0.5 # loss weight for stereo-matched pseudo-labels
    mono_weight: 0.0 # loss weight for mono-only pseudo-labels (0 = ignore)
    cutoff: 0.9 # minimum 2D detection confidence
```

## FAQ

### What is stereo 3D detection and how does it differ from standard object detection?

Standard object detection produces 2D bounding boxes `[x, y, width, height]` in pixel coordinates. Stereo 3D detection goes further by estimating full 3D bounding boxes including the object's physical location in 3D space `[x, y, z]`, real-world dimensions `[height, width, length]` in meters, and orientation (rotation angle). It achieves this by processing calibrated stereo image pairs and leveraging the disparity between left and right views to estimate depth.

### How do I train a stereo 3D detection model on a custom dataset?

Your custom dataset needs calibrated stereo image pairs (left + right cameras), calibration files with projection matrices, and 18-value labels per object. Follow the [KITTI Stereo format](../datasets/detect/kitti-stereo.md), then train:

```python
from ultralytics import YOLO

model = YOLO("yolo26s-s3d.yaml")
results = model.train(data="your-stereo-dataset.yaml", epochs=1000, imgsz=[384, 1248], optimizer="SGD", cos_lr=True)
```

Use rectangular `imgsz` matching your camera's aspect ratio. SGD with cosine LR for 1000 epochs is recommended — shorter schedules (e.g., 200 epochs) may not converge fully.

### What metrics does KITTI R40 evaluation use?

KITTI R40 evaluation computes 3D Average Precision (AP3D) using 40-point interpolated precision-recall curves. Results are reported at three difficulty levels (Easy, Moderate, Hard) based on object size, occlusion, and truncation. The standard IoU thresholds are 0.5 and 0.7 for 3D bounding box overlap. The primary benchmark metric is **AP3D@0.5 at Moderate difficulty**, averaged across all evaluated classes.

### How does auto-labeling prevent depth collapse?

When training with only one object class, the backbone learns spatial shortcuts (position-to-depth mapping) instead of real depth features, causing depth predictions to collapse. Auto-labeling adds pseudo-labels for additional classes (e.g., Pedestrian, Cyclist) from a pretrained 2D detector, providing the visual diversity needed to learn generalizable depth features. The pseudo-labels are down-weighted in the loss function via `pseudo_labels.weight` to avoid dominating the real annotations.

### What pretrained stereo 3D detection models are available?

Five YOLO26 siamese models are available in sizes n/s/m/l/x, ranging from 3.6M to 69.9M parameters. The best-performing model is YOLO26l (31.2M params) achieving 50.9% AP3D@0.5 and 31.6% AP3D@0.7 on KITTI Moderate. The x-size model has diminishing returns due to overfitting on KITTI's small training set. See the [Models section](#models) for the full comparison table.
