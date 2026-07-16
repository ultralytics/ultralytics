---
comments: true
description: Learn about monocular depth estimation using YOLO26. Predict per-pixel depth maps from single RGB images with NYU Depth V2 and custom dataset support.
keywords: monocular depth estimation, YOLO26, depth map, per-pixel depth, NYU Depth V2, DPT, dense prediction, Ultralytics
model_name: yolo26n-depth
---

# Monocular Depth Estimation

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/depth-estimation-examples.avif" alt="Monocular depth estimation examples">

Monocular depth estimation predicts a per-pixel depth map from a single RGB image. Each output pixel holds a depth value in meters representing the estimated distance from the camera to that surface point.

The output of a depth model is a dense float map of shape `(H, W)` aligned to the input image. This per-pixel representation makes monocular depth estimation well-suited for 3D scene reconstruction, robot navigation, AR/VR content creation, and any application that requires spatial layout from a single camera.

!!! tip

    Use `task=depth` or the `yolo depth` CLI task for monocular depth estimation. YOLO26 depth model files use the `-depth` suffix, such as `yolo26n-depth.pt`.

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

YOLO26 depth models pretrained on a broad multi-dataset mix (indoor + outdoor, ~2.19M images) are shown below. The metrics columns are reported on the [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) Eigen test split.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                                                            | size<br><sup>(pixels)</sup> | delta1<sup>NYU</sup> | abs_rel<sup>NYU</sup> | rmse<sup>NYU</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
| ------------------------------------------------------------------------------------------------ | --------------------------- | -------------------- | --------------------- | ------------------ | ------------------------ | ----------------------- |
| [YOLO26n-depth](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-depth.pt) | 768                         | 0.882                | 0.109                 | 0.414              | 6.4                      | 46.9                    |
| [YOLO26s-depth](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-depth.pt) | 768                         | 0.896                | 0.104                 | 0.399              | 13.2                     | 67.9                    |
| [YOLO26m-depth](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-depth.pt) | 768                         | 0.921                | 0.089                 | 0.364              | 23.3                     | 130.7                   |
| [YOLO26l-depth](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-depth.pt) | 768                         | 0.930                | 0.083                 | 0.351              | 27.7                     | 157.2                   |
| [YOLO26x-depth](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-depth.pt) | 768                         | 0.933                | 0.080                 | 0.344              | 57.0                     | 302.0                   |

- **delta1<sup>NYU</sup>** is the percentage of pixels where the predicted depth is within a factor of 1.25 of the ground truth, on the NYU Depth V2 Eigen test split (654 images) with multi-scale + horizontal-flip TTA and log-least-squares alignment.
- **abs_rel** is the mean absolute relative error between predicted and ground-truth depth values.
- **rmse** is the root mean squared error in meters.
- **params** and **FLOPs** are measured at 768×768, the training resolution of the released weights.

## Depth range and the log-depth head

The depth head supports two output parameterizations, selected by the model YAML:

| Mode                | Output                                                      | YAML                        | Use when                                  |
| ------------------- | ----------------------------------------------------------- | --------------------------- | ----------------------------------------- |
| **`log`** (default) | `exp(logit)` — **unbounded** (~0.02–150 m)                  | `yolo26-depth.yaml`         | General use, mixed or unknown depth range |
| `sigmoid`           | `sigmoid(logit) × max_depth` — **bounded** `[0, max_depth]` | `yolo26-depth-sigmoid.yaml` | Fixed-range / safety-constrained rigs     |

The default `log` head **decouples scene shape from absolute scale**: the network predicts a relative log-depth field, and absolute meters are set by a separate two-parameter transform (`exp(a·log d + b)`) recovered at evaluation, by lightweight calibration, or by fine-tuning. A bounded `sigmoid × max_depth` head instead bakes a fixed ceiling into the architecture, so any depth beyond `max_depth` is clipped — which prevents training on, and predicting, longer-range scenes.

### Why the default is `log`: evidence across depth ranges

**Controlled A/B — same data, same schedule, only the head differs.** Training both heads from scratch on an identical mix of indoor (≤10 m) and outdoor (≤80 m) data:

| Head                         | Output range   | Mixed-range val δ1 | Training behavior       |
| ---------------------------- | -------------- | -----------------: | ----------------------- |
| `sigmoid × max_depth` (10 m) | bounded 0–10 m |              0.221 | **plateaus at epoch 1** |
| `log` (unbounded)            | 0–~150 m       |   **0.367** (+66%) | improves throughout     |

The bounded head cannot represent the >10 m majority of outdoor pixels, so it stops improving almost immediately; the `log` head learns from the full range.

**The failure mode it fixes — single-dataset fine-tuning, stratified by range.** Fine-tuning a bounded (10 m) head on individual datasets works when the data fits the cap and collapses when it exceeds it:

| Dataset   | Depth range  |                                    Bounded-head δ1 |
| --------- | ------------ | -------------------------------------------------: |
| SUN RGB-D | ≤10 m        |                                            0.78 ✅ |
| Hypersim  | mostly ≤10 m |                                            0.74 ✅ |
| KITTI     | ~80 m        | 0.08 ❌ (abs_rel ≈ 7.0 — the 80/10 scale mismatch) |
| vKITTI2   | 80 m         |                                            0.04 ❌ |

The collapse lands exactly at the cap boundary. The `log` head has no such boundary.

**Released models — cross-range performance.** The shipped `log`-head family, evaluated across benchmarks spanning 10 m (NYU, iBims-1) to 80 m (KITTI), with scale-aligned δ1:

| Benchmark   | Range | YOLO26x-depth (`log`) | prior bounded release |
| ----------- | ----- | --------------------: | --------------------: |
| NYU Eigen   | 10 m  |                 0.933 |                 0.934 |
| iBims-1     | 10 m  |                 0.961 |                 0.945 |
| ETH3D       | ~60 m |                 0.959 |                 0.931 |
| Make3D      | ~70 m |                 0.302 |                 0.296 |
| KITTI Eigen | 80 m  |             **0.942** |                 0.891 |
| **Mean**    | —     |             **0.819** |                 0.799 |

The largest gains are on the longer-range outdoor benchmarks (KITTI, ETH3D) — exactly where a fixed 10 m ceiling hurts most — while indoor performance is retained.

## Train

Train YOLO26n-depth on the [Depth8](../datasets/depth/depth8.md) dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 640. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-depth.yaml")  # build a new model from YAML
        model = YOLO("yolo26n-depth.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo26n-depth.yaml").load("yolo26n-depth.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="depth8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo depth train data=depth8.yaml model=yolo26n-depth.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo depth train data=depth8.yaml model=yolo26n-depth.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo depth train data=depth8.yaml model=yolo26n-depth.yaml pretrained=yolo26n-depth.pt epochs=100 imgsz=640
        ```

See full `train` mode details in the [Train](../modes/train.md) page.

### Fine-tuning on your own data

When adapting a pretrained depth model to a custom dataset, **lower the learning rate and use the AdamW optimizer**. The default optimizer settings are tuned for training from scratch (SGD with `lr0=0.01`); applied to an already-converged depth model they can overwrite the pretrained knowledge and degrade results — especially when fine-tuning on a single domain.

!!! example "Recommended fine-tuning recipe"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26s-depth.pt")  # start from pretrained weights

        model.train(
            data="path/to/your_dataset.yaml",
            epochs=20,
            imgsz=640,
            optimizer="AdamW",
            lr0=1e-4,  # ~100x below the from-scratch default
            warmup_bias_lr=1e-4,  # keep warmup gentle too
        )
        ```

    === "CLI"

        ```bash
        yolo depth train data=path/to/your_dataset.yaml model=yolo26s-depth.pt \
          epochs=20 imgsz=640 optimizer=AdamW lr0=1e-4 warmup_bias_lr=1e-4
        ```

Additional tips:

- **Augmentation is controlled by the standard args** (`degrees`, `translate`, `scale`, `shear`, `perspective`, `flipud`, `fliplr`, `hsv_h`, `hsv_s`, `hsv_v`); the geometric warp and flips are applied identically to the paired depth map. To see the exact recipe used for the released YOLO26-Depth weights, inspect the `train_args` stored in the checkpoint — see [Inspecting YOLO26 Checkpoint Training Args](../guides/yolo26-training-recipe.md#inspecting-yolo26-checkpoint-training-args).
- **Any depth range works out of the box.** The default `log`-head models predict unbounded depth, so they adapt to short-range (macro) or long-range (outdoor/driving) data without changes. If you use the bounded `yolo26-depth-sigmoid.yaml` variant instead, set `max_depth:` in your dataset YAML to your scene's maximum depth (in meters).
- **Retain general performance.** If you need the model to stay accurate on scenes beyond your training set, mix a small fraction (~5–10%) of diverse general-purpose images into your training data; this substantially reduces forgetting during fine-tuning.
- **Train from scratch** (`model=yolo26s-depth.yaml`) only if your domain is very different and you have a large dataset — there the default SGD `lr0=0.01` is appropriate, since there are no pretrained weights to preserve.

### Calibrating the depth scale

The depth head separates **shape** (relative scene structure) from **scale** (absolute meters). If a model already produces good relative depth on your scenes but the absolute values are off for your camera, you can correct the scale in seconds with `model.calibrate()` — a closed-form fit of a two-parameter log-affine against a small labeled set, with **no gradient training and no change to the network weights**, so it cannot degrade the relative structure.

!!! example "Scale calibration"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26s-depth.pt")

        # Fit scale on a labeled split (~100+ images is enough), then persist it
        model.calibrate(data="path/to/your_dataset.yaml")
        model.save("yolo26s-depth-calibrated.pt")
        ```

Calibration needs ground-truth depth to fit against, so it runs on a labeled split — it is not something that can happen at blind inference. Use it when relative depth is already good and only the scale/range is wrong; if the relative structure itself needs to change for your domain, [fine-tune](#fine-tuning-on-your-own-data) instead.

Training does this for you automatically: after `model.train(...)` completes, the best and last checkpoints are calibrated on the validation set so they output metric-scaled depth out of the box.

The released `yolo26*-depth.pt` checkpoints ship with this calibration already baked in, fit on the pretraining validation mix. It is a single global scale across all domains, so for the most accurate absolute depth on a specific camera or scene type, run `model.calibrate()` on a small labeled split from your own data — it replaces the baked-in fit.

### Dataset format

Depth estimation datasets pair each RGB image with a corresponding depth file. Depth targets are stored as `.npy` arrays containing float32 values in meters. The dataset YAML points to an `images/` directory; the loader derives the depth file path by replacing the `images` component with `depth` and replacing the image extension with `.npy`.

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── depth/
    ├── train/
    └── val/
```

For example, an image at `images/train/scene_001.jpg` is paired with a depth map at `depth/train/scene_001.npy`. See the [Depth Estimation Dataset Guide](../datasets/depth/index.md) for the full format specification.

## Val

Validate a trained YOLO26n-depth model [accuracy](https://www.ultralytics.com/glossary/accuracy) on a depth estimation dataset. Pass `data` explicitly so validation uses the intended dataset YAML. The released weights are trained at `imgsz=768`, so validate and predict at that size for best accuracy.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-depth.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val(data="nyu-depth.yaml")
        metrics.delta1  # percentage of pixels within threshold δ=1.25
        metrics.abs_rel  # mean absolute relative error
        metrics.rmse  # root mean squared error (meters)
        metrics.silog  # scale-invariant logarithmic error
        ```

    === "CLI"

        ```bash
        yolo depth val model=yolo26n-depth.pt data=nyu-depth.yaml   # validate official model
        yolo depth val model=path/to/best.pt data=path/to/data.yaml # validate custom model
        ```

## Predict

Use a trained YOLO26n-depth model to run predictions on images.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-depth.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

        # Access the results
        for result in results:
            depth_map = result.depth.data  # NumPy float32 array, shape (H, W), values in meters
        ```

    === "CLI"

        ```bash
        yolo depth predict model=yolo26n-depth.pt source='https://ultralytics.com/images/bus.jpg' # predict with official model
        yolo depth predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

### Results Output

YOLO depth estimation returns one `Results` object per image. Each result stores one dense float depth map for the full image.

| Attribute           | Type            | Shape   | Description                |
| ------------------- | --------------- | ------- | -------------------------- |
| `result.depth`      | `DepthMap`      | `(H,W)` | Dense per-pixel depth map. |
| `result.depth.data` | NumPy `float32` | `(H,W)` | Depth values in meters.    |
| `result.boxes`      | -               | -       | No instance boxes.         |
| `result.masks`      | -               | -       | No instance masks.         |

For task-specific `Results` fields across every task, see the [Predict Results by Task](../modes/predict.md#results-by-task) section.

## Export

Export a YOLO26n-depth model to a different format like ONNX, CoreML, etc.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-depth.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-depth.pt format=onnx # export official model
        yolo export model=path/to/best.pt format=onnx  # export custom model
        ```

Available YOLO26 depth estimation export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo26n-depth.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### How do I train a YOLO26 depth estimation model on a custom dataset?

Prepare paired RGB images and `.npy` depth files, then create a dataset YAML pointing to your `images/` directory. The loader finds depth files automatically by replacing `images` with `depth` in the path and swapping the image extension for `.npy`.

Start from pretrained weights and use a low learning rate with AdamW so the fine-tune retains what the model already knows (see [Fine-tuning on your own data](#fine-tuning-on-your-own-data) for why):

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO26 depth model
        model = YOLO("yolo26s-depth.pt")

        # Fine-tune on a custom depth dataset
        results = model.train(
            data="path/to/your_dataset.yaml",
            epochs=20,
            imgsz=640,
            optimizer="AdamW",
            lr0=1e-4,
            warmup_bias_lr=1e-4,
        )
        ```

    === "CLI"

        ```bash
        yolo depth train data=path/to/your_dataset.yaml model=yolo26s-depth.pt \
          epochs=20 imgsz=640 optimizer=AdamW lr0=1e-4 warmup_bias_lr=1e-4
        ```

Check the [Configuration](../usage/cfg.md) page for more available arguments.

### What metrics does YOLO26 depth estimation report?

Depth estimation validation reports the metric set used by Depth Anything and related monocular-depth work:

- **delta1 / delta2 / delta3** — percentage of pixels where the ratio of predicted to ground-truth depth (or its inverse) is below 1.25, 1.25², and 1.25³ respectively. Higher is better.
- **abs_rel** — mean absolute relative error. Lower is better.
- **rmse** — root mean squared error in meters. Lower is better.
- **silog** — scale-invariant logarithmic error. Lower is better.

Each prediction is median-aligned to its ground truth per image, and the statistics are then pooled over every valid pixel of the validation set (images with more valid depth pixels weigh proportionally more). Papers that instead average per-image metrics can report slightly different values on the same predictions.

### What is the depth map output format?

The depth map is stored as a NumPy `float32` array of shape `(H, W)` where each value is the predicted depth in meters. Access it through `result.depth.data` after running prediction. The map is aligned to the input image resolution.

### What datasets are supported for depth estimation?

Ultralytics YOLO26 provides built-in dataset YAML configurations for several depth estimation datasets including NYU Depth V2, KITTI, Hypersim, SUN RGB-D, and ARKitScenes. See the [Depth Estimation Dataset Guide](../datasets/depth/index.md) for the full list and format details.

### How do I validate a pretrained YOLO26 depth estimation model?

Validate a pretrained YOLO26 depth model by supplying the dataset YAML used for evaluation:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n-depth.pt")

        # Validate the model
        metrics = model.val(data="nyu-depth.yaml")
        print("delta1:", metrics.delta1)
        print("abs_rel:", metrics.abs_rel)
        print("rmse:", metrics.rmse)
        ```

    === "CLI"

        ```bash
        yolo depth val model=yolo26n-depth.pt data=nyu-depth.yaml
        ```

### How can I export a YOLO26 depth estimation model to ONNX format?

Export a YOLO26 depth model to ONNX with Python or CLI:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n-depth.pt")

        # Export the model to ONNX format
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-depth.pt format=onnx
        ```

For more details on exporting to various formats, refer to the [Export](../modes/export.md) page.
