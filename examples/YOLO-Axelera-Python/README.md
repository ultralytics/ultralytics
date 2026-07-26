# Ultralytics YOLO on Axelera Voyager SDK

This document explains how to build and run complete Ultralytics YOLO inference
pipelines on Axelera Metis devices using the Axelera runtime.

The typical workflow has two phases:

1. **Model preparation.** You use Ultralytics, PyTorch, and the Voyager SDK
   compiler to train, export, and compile your model to `.axm` format.
2. **Application development.** You use `axelera-rt` to build and run an
   end-to-end video pipeline. `axelera-rt` is a lightweight, self-contained
   library with minimal package dependencies. It lets you describe pipelines
   that read frames from a camera, preprocess them, run model inference, and
   postprocess the results. The runtime can also fuse pipeline stages together
   automatically to minimize computation and data transfers.

A pipeline looks like this:

```python
pipeline = op.seq(
    op.color_convert("RGB", src="BGR"),  # OpenCV reads BGR; models expect RGB
    op.letterbox(640, 640),
    op.to_tensor(),
    op.load("yolo26n-pose.axm"),
    ConfidenceFilter(threshold=0.25),  # custom operator — see below
    op.to_image_space(keypoint_cols=range(6, 57, 3)),
).optimized()  # runtime fuses ops for maximum throughput

poses = pipeline(frame)  # frame in, raw (N, 57) NumPy array out — drops straight into existing post-processing
```

Note that this example returns the model's raw NumPy rows rather than typed
`Results` objects — see the two examples below.

## Examples

The two examples take different approaches; pick whichever is closer to how you
already work.

`yolo26-pose-tracker.py` keeps OpenCV for capture, drawing, and post-processing, and
only runs the model (and, optionally, a tracker) through Axelera. The pose-only
pipeline returns the model's raw NumPy rows, so it slots into NumPy post-processing
you already have. Uses YOLO26 (NMS-free).

`yolo11-seg.py` runs the whole pipeline through the runtime instead: hardware decode
operators, typed `Results` objects, `cv.create_source()` for capture, and the
`display.App` renderer, so there's no manual drawing code. Uses YOLO11 (its decode
path is shared with YOLOv8).

| Script                   | Task                       | Pipeline output                          | Capture / render                       |
| ------------------------ | -------------------------- | ---------------------------------------- | -------------------------------------- |
| `yolo26-pose-tracker.py` | Pose (+ optional tracking) | raw NumPy; `TrackedObject` when tracking | OpenCV `VideoCapture` + manual drawing |
| `yolo11-seg.py`          | Instance segmentation      | typed `SegmentedObject`                  | `cv.create_source()` + `display.App`   |

Tracking is the one runtime feature the first example opts into: `op.tracker()` needs
typed objects, so the tracking path adds `op.ax_pose()` and yields `TrackedObject`;
`--tracker none` skips that and stays raw NumPy.

## Quick Start

### Install

If you followed the
[Axelera integration guide](../../docs/en/integrations/axelera.md), `axelera-rt`
is already installed — running `yolo predict` or `yolo val` with an Axelera
model auto-installs the runtime dependencies. If you need to install manually:

```bash
pip install axelera-rt==1.7.0 --no-cache-dir --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple
```

You will also need `opencv-python` and `numpy` (likely already present).

### Compile

Export and compile your trained model directly to Axelera format using the
Ultralytics CLI:

```bash
yolo export model=your-model.pt format=axelera
```

To reproduce these examples, export the pretrained Ultralytics models:

```bash
yolo export model=yolo26n-pose.pt format=axelera
yolo export model=yolo11n-seg.pt format=axelera
```

The compiled models are saved to `yolo26n-pose_axelera_model/` and
`yolo11n-seg_axelera_model/` respectively. Pass the `.axm` file inside to
`--model`.

> [!NOTE] Each model directory also contains `metadata.yaml` with the model's
> class names and configuration. You can load this info directly to avoid manual
> label entry in your own application.

### Run

#### Pose Estimation (Ultralytics YOLO26)

```bash
python yolo26-pose-tracker.py --model yolo26n-pose.axm --source 0 --tracker none         # pose only # webcam
python yolo26-pose-tracker.py --model yolo26n-pose.axm --source image.jpg --tracker none # pose only # image
python yolo26-pose-tracker.py --model yolo26n-pose.axm --source video.mp4 --tracker none # pose only # video
```

#### Pose Tracking (Ultralytics YOLO26 + TrackTrack)

```bash
python yolo26-pose-tracker.py --model yolo26n-pose.axm --source video.mp4              # video with tracking
python yolo26-pose-tracker.py --model yolo26n-pose.axm --source 0 --tracker tracktrack # webcam with tracking
```

#### Instance Segmentation (Ultralytics YOLO11)

```bash
python yolo11-seg.py --model yolo11n-seg.axm --source 0                              # webcam
python yolo11-seg.py --model yolo11n-seg.axm --source video.mp4 --conf 0.3 --iou 0.5 # display
python yolo11-seg.py --model yolo11n-seg.axm --source video.mp4 --output out.mp4     # display + save video
python yolo11-seg.py --model yolo11n-seg.axm --source video.mp4 --no-display         # headless (no save)
```

> [!NOTE] `yolo11-seg.py` uses the runtime's `display.App` renderer and
> `pipeline.stream()` instead of OpenCV display; results are drawn by the
> built-in renderer with no manual drawing code. Use `--no-display` for headless
> runs.
>
> Saving video (`--output`) has two limitations from the display backend:
>
> - It renders through the OpenCV display, so it requires a display and cannot
>   be combined with `--no-display`.
> - The saved video is written at the display window's canvas size (`800x500`),
>   not the source's native resolution. Change the `create_window(...)` size in
>   the script to write at a different resolution.

### Arguments

| Argument       | Default             | Scripts   | Description                                                                                                                                                      |
| -------------- | ------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model`      | _required_          | all       | Path to compiled `.axm` model                                                                                                                                    |
| `--source`     | `0` (pose)          | all       | Image path, video path, or webcam index. _Required_ for `yolo11-seg.py`.                                                                                         |
| `--conf`       | `0.25`              | pose, seg | Confidence threshold                                                                                                                                             |
| `--iou`        | `0.45`              | seg       | NMS IoU threshold                                                                                                                                                |
| `--tracker`    | `tracktrack`        | pose      | Tracking algorithm: `bytetrack`, `oc-sort`, `sort`, `tracktrack`, `none`                                                                                         |
| `--no-display` | _off_               | all       | Disable the GUI window (headless mode)                                                                                                                           |
| `--output`     | `output.mp4` (pose) | pose, seg | Output video path. Pose: written in headless mode (`--no-display`). Seg: off by default; renders via the display app, so cannot be combined with `--no-display`. |

## What You'll See in the Code

### Custom Operators

The pipeline is fully composable — drop in your own operators anywhere in the
sequence. Just subclass `op.Operator` and implement `__call__`:

```python
class ConfidenceFilter(op.Operator):
    threshold: float = 0.25

    def __call__(self, x):
        """Filter detections by confidence score."""
        # Each operator receives the output of the previous one in op.seq().
        # Here x is the output of the pose detection model — an array of
        # shape (batch, num_detections, num_values) where each row holds
        # [bbox, confidence, class_id, keypoints…]. Column 4 is the
        # confidence score. Squeeze the batch dim and filter by score.
        if x.ndim == 3:
            x = x[0]
        return x[x[:, 4] >= self.threshold]
```

This means they are fused, scheduled, and optimized alongside built-in ops like
`op.letterbox` and the model itself — adding a custom filter carries no special
overhead.

### Video Sources (`cv.create_source`)

`cv.create_source(path)` opens a `VideoSource`: an iterable of decoded `Image` frames
that also exposes metadata (`fps`, `frame_count`) and works as a context manager. The
path can be a video file, a folder of images, a USB camera (`/dev/video0`), or an
RTSP/RTP/RTMP/SRT URL — live sources drop frames to stay real-time, while files block
so none are skipped. Hand the source straight to `pipeline.stream()` to get pipelined
`(image, result)` pairs in source order:

```python
from axelera.runtime import cv

with cv.create_source("clip.mp4") as source:
    print(source.fps, source.frame_count)  # available before iterating
    for image, segments in pipeline.stream(source):
        ...
```

`backend` (`"ffmpeg"` default, or `"opencv"`), `buffer_size`, and `live_source` are
available when you need to tune decoding. `yolo11-seg.py` uses this for capture.

### Rendering (`display.App`)

`display.App(renderer=...)` runs the built-in renderer as a context manager:
`"auto"` opens an OpenCV window when a display is available (and falls back to console
output otherwise), `"none"` disables rendering for headless runs. Create a window with
`create_window(title, size)` and push frames with `vis(image, result)` — typed
`Results` objects render themselves via their `.draw()` method, so masks, boxes, poses,
and tracks are drawn with no manual OpenCV code:

```python
from axelera.runtime import cv, display

with display.App(renderer="auto") as app:
    vis = app.create_window("segmentation", (800, 500))
    with cv.create_source("clip.mp4") as source:
        for image, segments in pipeline.stream(source):
            if vis.is_closed:  # window closed by the user
                break
            vis(image, segments)  # SegmentedObject.draw() paints masks + boxes
```

Pass `expose_surface=True` to `create_window()` to enable `vis.save_output_video(path,
fps)` for recording (OpenCV renderer only). `yolo11-seg.py` uses this for display and
`--output`.

### Multi-Object Tracking

Adding tracking to a detection or pose pipeline is essentially a single line via
`op.tracker()`. The tracker works on typed objects, so the pipeline must produce
them (via `op.ax_pose()` / `op.ax_detection()`) before tracking:

```python
pipeline = op.seq(
    # ... your detection or pose pipeline ...
    op.ax_pose(num_keypoints=17, class_id_type=op.CocoClasses),  # raw rows -> list[PoseObject]
    op.tracker(algo="tracktrack"),  # list[PoseObject] -> list[TrackedObject]
)
```

`tracktrack` is the default. Pass `--tracker` to select another algorithm:

| Algorithm  | Key Strength                                         | Reference                |
| ---------- | ---------------------------------------------------- | ------------------------ |
| TrackTrack | Iterative matching with track-aware NMS (SOTA)       | CVPR 2025                |
| ByteTrack  | Handles low-confidence detections via dual threshold | Zhang et al., ECCV 2022  |
| OC-SORT    | Observation-centric re-update + virtual trajectory   | Cao et al., CVPR 2023    |
| SORT       | Simple, fast IoU-only baseline                       | Bewley et al., ICIP 2016 |

The tracker operates on bounding boxes, but the original detection (with all its
metadata) is preserved — use `tracked.tracked` on each result to access the
original keypoints, masks, or other model outputs. You'll see this pattern in
`yolo26-pose-tracker.py`.

## Learn More

The [Voyager SDK](https://github.com/axelera-ai-hub/voyager-sdk) has everything
beyond these introductory examples: advanced pipeline patterns, ReID and
appearance-based tracking, the high-performance display system, production-ready
application templates, and the full API reference.

Questions or feedback? Join the
[Axelera community](https://community.axelera.ai/).
