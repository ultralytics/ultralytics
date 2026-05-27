---
comments: true
description: Discover efficient, flexible, and customizable multi-object tracking with Ultralytics YOLO. Learn to track real-time video streams with ease.
keywords: multi-object tracking, Ultralytics YOLO, video analytics, real-time tracking, object detection, AI, machine learning
---

# Multi-Object Tracking with Ultralytics YOLO

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/multi-object-tracking-examples.avif" alt="YOLO multi-object tracking with trajectory paths">

Object tracking identifies the location and class of objects within a frame and maintains a unique ID for each detected object as the video progresses. Applications range from surveillance and security to real-time sports analytics.

## Why Choose Ultralytics YOLO for Object Tracking?

Ultralytics trackers output the same fields as standard [object detection](https://www.ultralytics.com/glossary/object-detection) with one addition: object IDs. This makes it easy to follow objects through video streams and perform downstream analytics.

- **Efficiency:** Process video streams in real-time without compromising [accuracy](https://www.ultralytics.com/glossary/accuracy).
- **Flexibility:** Supports multiple tracking algorithms and configurations.
- **Ease of Use:** Simple Python API and CLI options for quick integration and deployment.
- **Customizability:** Works with custom-trained YOLO models for domain-specific applications.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/qQkzKISt5GE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Run Multi-Object Tracking with Ultralytics YOLO26 | BoT-SORT & ByteTrack | VisionAI 🚀
</p>

## Real-world Applications

|           Transportation           |              Retail              |         Aquaculture          |
| :--------------------------------: | :------------------------------: | :--------------------------: |
| ![Vehicle Tracking][vehicle track] | ![People Tracking][people track] | ![Fish Tracking][fish track] |
|          Vehicle Tracking          |         People Tracking          |        Fish Tracking         |

## Quick Start

Run tracking on a video with the default BoT-SORT tracker. Swap to another tracker by changing the `tracker` argument.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Default tracker (BoT-SORT)
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)

        # Switch to ByteTrack
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")
        ```

    === "CLI"

        ```bash
        # Default tracker (BoT-SORT)
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4" show

        # Switch to ByteTrack
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4" tracker="bytetrack.yaml"
        ```

Tracking is available for Detect, Segment, and Pose models on videos or streaming sources.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")       # Detect model
        model = YOLO("yolo26n-seg.pt")   # Segment model
        model = YOLO("yolo26n-pose.pt")  # Pose model
        model = YOLO("path/to/best.pt")  # Custom-trained model

        results = model.track("https://youtu.be/LNwODJXcvt4", show=True)
        ```

    === "CLI"

        ```bash
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4"      # Detect
        yolo track model=yolo26n-seg.pt source="https://youtu.be/LNwODJXcvt4"  # Segment
        yolo track model=yolo26n-pose.pt source="https://youtu.be/LNwODJXcvt4" # Pose
        yolo track model=path/to/best.pt source="https://youtu.be/LNwODJXcvt4" # Custom
        ```

## Supported Trackers

Ultralytics YOLO ships with six built-in trackers. Enable one by passing its YAML config file to the `tracker` argument.

| Tracker                                              | Config file        | Motion model               | Appearance / ReID         | Camera motion compensation | Occlusion handling                              |
| ---------------------------------------------------- | ------------------ | -------------------------- | ------------------------- | -------------------------- | ----------------------------------------------- |
| **[BoT-SORT](#bot-sort)**                            | `botsort.yaml`     | Linear Kalman              | Optional (`with_reid`)    | Yes (sparseOptFlow / ECC)  | Track buffer + ReID rebinding                   |
| **[ByteTrack](#bytetrack)**                          | `bytetrack.yaml`   | Linear Kalman              | None                      | No                         | Two-stage low-conf rescue                       |
| **[OC-SORT](#oc-sort)**                              | `ocsort.yaml`      | Observation-centric Kalman | None                      | No                         | ORU, OCM, OCR re-update from last observation   |
| **[Deep OC-SORT](#deep-oc-sort)**                    | `deepocsort.yaml`  | Observation-centric Kalman | Optional (`with_reid`)    | Optional (`gmc_method`)    | OC-SORT + adaptive appearance EMA               |
| **[FastTracker](#fasttracker)**                      | `fasttrack.yaml`   | Linear Kalman + rollback   | None                      | No                         | Kalman rollback + bbox enlargement on occlusion |
| **[TrackTrack](#tracktrack)**                        | `tracktrack.yaml`  | Linear Kalman (NSA)        | Optional (HMIoU fallback) | Yes (sparseOptFlow / ECC)  | Iterative multi-cue association + TAI           |

### Which Tracker Should I Use?

Use this flow to pick a starting point:

1. **Need the fastest, simplest baseline?** → **ByteTrack** (no ReID, no camera-motion compensation, minimum overhead).
2. **Handheld, drone, or moving-camera footage?** → **BoT-SORT** (default; adds camera-motion compensation and optional ReID).
3. **Non-linear motion (sports, dancing, abrupt turns) and no ReID?** → **OC-SORT** (observation-centric corrections without appearance cost).
4. **Crowded moving-camera scenes where ID swaps are the main problem?** → **Deep OC-SORT** or **TrackTrack** (both add adaptive appearance fusion; TrackTrack also adds multi-cue association and duplicate-ID suppression).
5. **Frequent partial overlap in real-time, no ReID budget?** → **FastTracker** (occlusion-aware ByteTrack variant with Kalman rollback).

## Switching Trackers

Pass the tracker config filename to `tracker=`. All other code stays the same.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        results = model.track(source="path/to/video.mp4", tracker="bytetrack.yaml")
        results = model.track(source="path/to/video.mp4", tracker="ocsort.yaml")
        results = model.track(source="path/to/video.mp4", tracker="tracktrack.yaml")
        ```

    === "CLI"

        ```bash
        yolo track model=yolo26n.pt source="path/to/video.mp4" tracker="bytetrack.yaml"
        ```

## Configuration

### Tracking Arguments

Tracking shares arguments with Predict mode, such as `conf`, `iou`, and `show`. See the [Predict](../modes/predict.md#inference-arguments) page for the full list.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.1, iou=0.7, show=True)
        ```

    === "CLI"

        ```bash
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.1 iou=0.7 show
        ```

### Custom Tracker Configuration

Copy an existing tracker config file from [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers), modify parameters (do not change `tracker_type`), and pass it to the `tracker` argument.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker="custom_tracker.yaml")
        ```

    === "CLI"

        ```bash
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
        ```

### Shared Tracker Arguments

The following parameters are shared across all tracker YAML files:

!!! warning "Tracker Threshold Information"

    If a detection's confidence score falls below `track_high_thresh`, the tracker will not update that object, resulting in no active tracks.

| Parameter       | Valid Values or Ranges                                                | Description                                                                                                                                                                                                         |
| --------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tracker_type`   | `botsort`, `bytetrack`, `ocsort`, `deepocsort`, `fasttrack`, `tracktrack` | Specifies the tracker type.                                                                                                                                                                                     |
| `track_high_thresh` | `0.0-1.0`                                                         | Threshold for the first association. Affects how confidently a detection is matched to an existing track.                                                                                                         |
| `track_low_thresh`  | `0.0-1.0`                                                         | Threshold for the second association. Used when the first association fails, with more lenient criteria.                                                                                                          |
| `new_track_thresh`  | `0.0-1.0`                                                         | Threshold to initialize a new track if the detection does not match any existing tracks.                                                                                                                          |
| `track_buffer`      | `>=0`                                                             | Frames lost tracks are kept alive before removal. Higher value means more tolerance for occlusion.                                                                                                                |
| `match_thresh`      | `0.0-1.0`                                                         | Threshold for matching tracks. Higher values make matching more lenient.                                                                                                                                            |
| `fuse_score`        | `True`, `False`                                                   | Whether to fuse confidence scores with IoU distances before matching.                                                                                                                                               |
| `gmc_method`        | `sparseOptFlow`, `orb`, `sift`, `ecc`, `none`                     | Global motion compensation method. Helps account for camera movement.                                                                                                                                               |
| `proximity_thresh`  | `0.0-1.0`                                                         | Minimum IoU required for a valid ReID match. Ensures spatial closeness before using appearance cues.                                                                                                                |
| `appearance_thresh` | `0.0-1.0`                                                         | Minimum appearance similarity required for ReID.                                                                                                                                                                    |
| `with_reid`         | `True`, `False`                                                   | Enable appearance-based matching for better tracking across occlusions. Supported by BoT-SORT, Deep OC-SORT, and TrackTrack.                                                                                        |
| `model`             | `auto` or path to an exported file                                | ReID model. `auto` uses native YOLO backbone features when available; otherwise falls back to `yolo26n-cls.pt`. Pass a `.torchscript`, `.onnx`, `.engine`, `.openvino`, … file for a custom encoder.              |

#### Tracker-specific Arguments

Each algorithm exposes additional knobs on top of the shared parameters. See the per-tracker sections below for descriptions and tuning advice, or refer directly to the config files:

- [`botsort.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/botsort.yaml)
- [`bytetrack.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml)
- [`ocsort.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/ocsort.yaml)
- [`deepocsort.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/deepocsort.yaml)
- [`fasttrack.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/fasttrack.yaml)
- [`tracktrack.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/tracktrack.yaml)

### Enabling Re-Identification (ReID)

ReID is disabled by default to minimize overhead. Enable it by setting `with_reid: True` in a tracker config file.

**ReID model options:**

- **`model: auto`** — Uses native YOLO detector features, adding minimal overhead. Ideal when you need some ReID without a large performance hit. Falls back to `yolo26n-cls.pt` if the detector does not expose compatible features.
- **Exported ReID model** — Point `model:` at an exported file (`.torchscript`, `.onnx`, `.engine`, `.openvino`, etc.) for more discriminative embeddings at the cost of an extra forward pass per crop. The encoder is loaded via `AutoBackend`, so any export format Ultralytics supports works without code changes.

For better performance with a separate classification model, export it to a faster backend like TensorRT:

!!! example "Exporting a ReID model to TensorRT"

    ```python
    from torch import nn

    from ultralytics import YOLO

    # Load the classification model
    model = YOLO("yolo26n-cls.pt")

    # Add average pooling layer
    head = model.model.model[-1]
    pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1))
    pool.f, pool.i = head.f, head.i
    model.model.model[-1] = pool

    # Export to TensorRT
    model.export(format="engine", half=True, dynamic=True, batch=32)
    ```

Once exported, point to the TensorRT model path in your tracker config.

## Tracker Details

Expand the sections below for each tracker's design, specific parameters, and tuning tips.

#### BoT-SORT

[BoT-SORT](https://github.com/NirAharon/BoT-SORT) (Aharon et al., 2022) is the default tracker. It extends ByteTrack with camera-motion compensation and optional ReID:

- **Camera Motion Compensation (CMC):** an affine warp estimated each frame (sparse optical flow by default; ORB / ECC also available) is applied to Kalman states before IoU matching.
- **Optional ReID:** appearance embeddings can be fused into the cost matrix. Disabled by default; enable with `with_reid: True`.

**Best for:** general-purpose tracking, especially moving cameras. Add ReID only when look-alike crowds cause ID swaps.

**BoT-SORT-specific arguments:**

| Parameter       | Valid Values or Ranges                    | Description                                                                                                          |
| ---------------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `gmc_method`     | `sparseOptFlow`, `orb`, `sift`, `ecc`, `none` | Camera-motion-compensation backend. `sparseOptFlow` is the default. `none` disables CMC.                          |
| `with_reid`      | `True`, `False`                           | Enable appearance-based matching. Off by default.                                                                    |
| `model`          | `auto` or path to a ReID model            | ReID model. `auto` uses native YOLO features when available; otherwise pass a `.torchscript` / `.onnx` / `.engine` path. |
| `proximity_thresh`  | `0.0-1.0`                              | Minimum IoU before appearance features are considered.                                                               |
| `appearance_thresh` | `0.0-1.0`                              | Minimum cosine similarity required for a ReID match. Raise to reduce identity swaps.                                 |

**Tuning tips:**

- **Static camera:** set `gmc_method: none` to save a few ms/frame.
- **Heavy camera motion:** keep `sparseOptFlow`; `ecc` is more accurate but slower.
- **Look-alike crowds:** turn on `with_reid: True` and raise `appearance_thresh` (e.g. `0.85+`).

#### ByteTrack

[ByteTrack](https://github.com/FoundationVision/ByteTrack) (Zhang et al., ECCV 2022) is the lightweight baseline. It uses linear Kalman + IoU with a two-stage association:

- **Stage 1:** match high-score detections against active tracks.
- **Stage 2:** retry unmatched tracks against low-score detections to recover through brief partial occlusion.

There is no appearance model and no camera-motion compensation.

**Best for:** static or near-static cameras where detector cost dominates and you want minimum tracker overhead.

**ByteTrack-specific arguments:** None beyond the [shared tracker arguments](#shared-tracker-arguments).

**Tuning tips:**

- **Noisy detector:** lower `track_low_thresh` so the second stage has more candidates.
- **High-recall detector:** raise `track_high_thresh` to reduce fragmented IDs.
- **Frequent ID flicker:** raise `track_buffer` so briefly-missed tracks survive.

#### OC-SORT

[OC-SORT](https://arxiv.org/abs/2203.14360) (Cao et al., CVPR 2023) is an observation-centric extension of SORT. It keeps SORT's lightweight design (no appearance features) and adds three corrections:

- **Observation-Centric Re-update (ORU):** replays a virtual trajectory between the last observation and the current detection, re-running the Kalman update to repair drifted velocity.
- **Observation-Centric Momentum (OCM):** penalizes detections moving in the wrong direction via a velocity-consistency term.
- **Observation-Centric Recovery (OCR):** re-checks unmatched detections against recently lost tracks using their last observation rather than the predicted state.

**Best for:** non-linear motion without the cost of a ReID model.

**OC-SORT-specific arguments:**

| Parameter  | Valid Values or Ranges | Description                                                                                                           |
| ---------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `delta_t`  | `>=1`                  | Temporal window (frames) for velocity-direction computation in OCM. Larger values smooth more.                        |
| `inertia`  | `0.0-1.0`              | Weight of the velocity-consistency cost. Higher values penalize sudden direction changes.                             |
| `use_byte` | `True`, `False`        | Enable a ByteTrack-style second association pass over low-confidence detections.                                      |

**Tuning tips:**

- **Non-linear motion:** raise `inertia` (e.g. `0.3-0.4`).
- **Sparse detections:** enable `use_byte: True`.
- **Long occlusions:** raise `track_buffer` so OCR has more lost tracks to rebind.

#### Deep OC-SORT

[Deep OC-SORT](https://arxiv.org/abs/2302.11813) augments OC-SORT with appearance information and camera-motion compensation:

- **Adaptive appearance fusion:** detection embeddings are fused into the cost matrix with weight modulated by detection confidence and overlap.
- **Dynamic appearance EMA:** track embeddings update with an EMA whose smoothing factor adapts to detection confidence.
- **Camera Motion Compensation:** Kalman states are warped frame-to-frame via sparse optical flow, ORB, or ECC.

**Best for:** crowded or moving-camera scenes where ID swaps between visually different but spatially close objects are common.

**Deep OC-SORT-specific arguments:**

| Parameter       | Valid Values or Ranges                    | Description                                                                                                                                       |
| ---------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `with_reid`      | `True`, `False`                           | Enable appearance-based matching. Off by default.                                                                                                 |
| `model`          | `auto`, exported ReID model file          | ReID model. `auto` reuses native YOLO features; otherwise pass an exported file (`.torchscript`, `.onnx`, `.engine`, …).                          |
| `proximity_thresh`  | `0.0-1.0`                              | Minimum IoU before appearance features are considered.                                                                                            |
| `appearance_thresh` | `0.0-1.0`                              | Minimum cosine similarity required for a ReID match.                                                                                              |
| `alpha_fixed_emb`   | `0.0-1.0`                              | Base EMA factor for track-embedding updates. Higher values preserve the older embedding longer.                                                   |
| `gmc_method`        | `sparseOptFlow`, `orb`, `sift`, `ecc`, `none` | Global motion compensation method.                                                                                                           |

**Tuning tips:**

- **Identity swaps in crowds:** raise `appearance_thresh` (e.g. `0.92-0.95`) and lower `alpha_fixed_emb` so embeddings adapt more slowly.
- **Moving camera:** keep `gmc_method: sparseOptFlow`.
- **Lower latency:** keep `with_reid: False` (default) for motion + CMC only; enable ReID only when ID swaps dominate errors.

#### FastTracker

[FastTracker](https://arxiv.org/abs/2508.14370) is an occlusion-aware ByteTrack variant with no appearance model:

- **Occlusion detection:** flags tracks occluded when coverage by other active tracks exceeds `occ_cover_thresh`.
- **Kalman rollback on occlusion:** rolls the Kalman state back to a pre-occlusion frame using ring-buffered history.
- **Motion dampening and search expansion:** velocity is dampened and the predicted bbox is enlarged while occluded.
- **Init-IoU suppression:** prevents new tracks from spawning on top of active tracks.

**Best for:** real-time detection-only pipelines with frequent target-on-target overlap (crowds, queues, sports).

**FastTracker-specific arguments:**

| Parameter               | Valid Values or Ranges | Description                                                                                                           |
| ----------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `reset_velocity_offset_occ` | `>=0`              | History frames back to restore Kalman velocity on occlusion onset.                                                    |
| `reset_pos_offset_occ`      | `>=0`              | History frames back to restore Kalman position on occlusion onset.                                                    |
| `enlarge_bbox_occ`          | `>=1.0`            | Height scaling applied to the predicted bbox while occluded (width scales via XYAH aspect ratio).                     |
| `dampen_motion_occ`         | `0.0-1.0`          | Velocity multiplier while occluded. Lower values make the track "slow down" through occlusion.                        |
| `active_occ_to_lost_thresh` | `>=1`              | Max consecutive occluded frames before an active track is moved to lost.                                              |
| `occ_cover_thresh`          | `0.0-1.0`          | Fraction of a track's area covered by another active track to declare occlusion.                                      |
| `occ_reappear_window`       | `>=0`              | Frames a recently-occluded lost track stays preferentially re-findable.                                               |
| `init_iou_suppress`         | `0.0-1.0`          | Suppress new-track initialization if its IoU with any active track exceeds this. Set to `1.0` to disable.             |

**Tuning tips:**

- **Frequent partial occlusions:** lower `occ_cover_thresh` (e.g. `0.5-0.6`).
- **Duplicate IDs around overlap:** lower `init_iou_suppress` (e.g. `0.5`).
- **Long occlusions:** raise `occ_reappear_window` and `track_buffer` together.
- **Fast-moving targets:** raise `dampen_motion_occ` (closer to `1.0`) and lower `enlarge_bbox_occ`.

#### TrackTrack

[TrackTrack](https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf) (Shim et al., CVPR 2025) reasons from each track's perspective with multi-cue iterative association:

- **Track-Perspective-Based Association (TPA):** combines HMIoU, cosine ReID distance, confidence-projection distance, and corner-angle distance. Assignment is solved iteratively with a relaxing threshold.
- **Track-Aware Initialization (TAI):** suppresses duplicate spawns before a new ID is created.

**Best for:** crowded scenes with frequent occlusion where duplicate IDs are a problem.

**TrackTrack-specific arguments:**

| Parameter     | Valid Values or Ranges | Description                                                                                                                    |
| ------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `iou_weight`  | `0.0-1.0`              | Weight of HMIoU distance in the multi-cue cost matrix.                                                                         |
| `reid_weight` | `0.0-1.0`              | Weight of cosine ReID distance. Falls back to HMIoU if ReID is disabled.                                                       |
| `conf_weight` | `0.0-1.0`              | Weight of confidence-projection distance.                                                                                      |
| `angle_weight`| `0.0-1.0`              | Weight of corner-angle distance.                                                                                               |
| `det_thr`     | `0.0-1.0`              | Detection threshold separating high- from low-confidence detections.                                                           |
| `penalty_p`   | `0.0-1.0`              | Cost penalty for low-confidence detections.                                                                                    |
| `penalty_q`   | `0.0-1.0`              | Cost penalty for detections recovered by secondary NMS.                                                                        |
| `reduce_step` | `0.0-1.0`              | Match-threshold relaxation per iteration.                                                                                      |
| `tai_thr`     | `0.0-1.0`              | IoU threshold for Track-Aware Initialization NMS.                                                                              |
| `min_track_len`| `>=0`                 | Minimum successful updates before a new track is confirmed.                                                                    |
| `lost_match_thr`| `0.0-1.0`            | Looser cost gate for relaxed lost-rebind pass; `0` disables it.                                                                |
| `gmc_downscale`| `>=1`                 | Downscale factor for GMC input image. Higher is faster, less accurate.                                                         |
| `gmc_max_corners`| `>=1`               | Max keypoints for `sparseOptFlow`.                                                                                             |
| `gmc_skip_frames`| `>=0`               | Skip N frames between GMC updates. `0` = recompute every frame.                                                                |

**Tuning tips:**

- **Crowded pedestrians:** lower `tai_thr` (e.g. `0.45`) to suppress more duplicate spawns; raise `track_buffer` for longer occlusions.
- **Fast camera motion:** keep `gmc_method: sparseOptFlow`, lower `gmc_downscale` (e.g. `2`), or raise `gmc_max_corners`. If GMC is a bottleneck, set `gmc_skip_frames: 1` or `2`.
- **Small/fast objects:** raise `angle_weight` slightly and lower `min_track_len`.
- **Enable ReID only when needed:** it adds inference cost; for short occlusions, the default multi-cue cost is usually sufficient.

## Python Examples

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/leOPZhE0ckg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Build Interactive Object Tracking with Ultralytics YOLO | Click to Crop & Display ⚡
</p>

### Persisting Tracks Loop

Use `persist=True` when processing a video frame-by-frame so the tracker treats each frame as the next in a sequence.

!!! example "Streaming for-loop with tracking"

    ```python
    import cv2

    from ultralytics import YOLO

    # Load the YOLO26 model
    model = YOLO("yolo26n.pt")

    # Open the video file
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO26 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO26 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    ```

### Plotting Tracks Over Time

Draw movement paths by retaining the center points of bounding boxes across frames.

!!! example "Plotting tracks over multiple video frames"

    ```python
    from collections import defaultdict

    import cv2
    import numpy as np

    from ultralytics import YOLO

    # Load the YOLO26 model
    model = YOLO("yolo26n.pt")

    # Open the video file
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO26 tracking on the frame, persisting tracks between frames
            result = model.track(frame, persist=True)[0]

            # Get the boxes and track IDs
            if result.boxes and result.boxes.is_track:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

                # Visualize the result on the frame
                frame = result.plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display the annotated frame
            cv2.imshow("YOLO26 Tracking", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    ```

### Multithreaded Tracking

Run tracking on multiple video streams concurrently with Python's `threading` module.

!!! example "Multithreaded tracking implementation"

    ```python
    import threading

    import cv2

    from ultralytics import YOLO

    # Define model names and video sources
    MODEL_NAMES = ["yolo26n.pt", "yolo26n-seg.pt"]
    SOURCES = ["path/to/video.mp4", "0"]  # local video, 0 for webcam


    def run_tracker_in_thread(model_name, filename):
        """Run YOLO tracker in its own thread for concurrent processing.

        Args:
            model_name (str): The YOLO26 model object.
            filename (str): The path to the video file or the identifier for the webcam/external camera source.
        """
        model = YOLO(model_name)
        results = model.track(filename, save=True, stream=True)
        for r in results:
            pass


    # Create and start tracker threads using a for loop
    tracker_threads = []
    for video_file, model_name in zip(SOURCES, MODEL_NAMES):
        thread = threading.Thread(target=run_tracker_in_thread, args=(model_name, video_file), daemon=True)
        tracker_threads.append(thread)
        thread.start()

    # Wait for all tracker threads to finish
    for thread in tracker_threads:
        thread.join()

    # Clean up and close windows
    cv2.destroyAllWindows()
    ```

This example can be extended to more video files and models by adding more threads.

## FAQ

### What is Multi-Object Tracking and how does Ultralytics YOLO support it?

Multi-object tracking identifies objects and maintains a unique ID for each across video frames. Ultralytics YOLO supports real-time tracking with object IDs, shipping six built-in trackers configurable via YAML. See [Supported Trackers](#supported-trackers) for a comparison.

### How do I run object tracking on multiple video streams simultaneously?

Use Python's `threading` module. See the [Multithreaded Tracking](#multithreaded-tracking) example above.

### When should I use TrackTrack instead of BoT-SORT or ByteTrack?

TrackTrack is preferred when identity preservation is critical in crowded scenes with frequent occlusion. It uses multi-cue iterative association and Track-Aware Initialization to reduce duplicate IDs. ByteTrack remains the fastest option for simple scenes, and BoT-SORT is a strong general-purpose baseline with optional ReID.

## Contribute New Trackers

Are you proficient in multi-object tracking and have successfully implemented or adapted a tracking algorithm with Ultralytics YOLO? We invite you to contribute to our Trackers section in [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)! Your real-world applications and solutions could be invaluable for users working on tracking tasks.

By contributing to this section, you help expand the scope of tracking solutions available within the Ultralytics YOLO framework, adding another layer of functionality and utility for the community.

To initiate your contribution, please refer to our [Contributing Guide](../help/contributing.md) for comprehensive instructions on submitting a Pull Request (PR) 🛠️. We are excited to see what you bring to the table!

Together, let's enhance the tracking capabilities of the Ultralytics YOLO ecosystem 🙏!

[fish track]: https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/fish-tracking.avif
[people track]: https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/people-tracking.avif
[vehicle track]: https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/vehicle-tracking.avif
