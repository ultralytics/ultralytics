---
comments: true
description: Discover efficient, flexible, and customizable multi-object tracking with Ultralytics YOLO. Learn to track real-time video streams with ease.
keywords: multi-object tracking, Ultralytics YOLO, video analytics, real-time tracking, object detection, AI, machine learning
---

# Multi-Object Tracking with Ultralytics YOLO

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/multi-object-tracking-examples.avif" alt="YOLO multi-object tracking with trajectory paths">

Object tracking in the realm of video analytics is a critical task that not only identifies the location and class of objects within the frame but also maintains a unique ID for each detected object as the video progresses. The applications are limitless—ranging from surveillance and security to real-time sports analytics.

## Why Choose Ultralytics YOLO for Object Tracking?

The output from Ultralytics trackers is consistent with standard [object detection](https://www.ultralytics.com/glossary/object-detection) but has the added value of object IDs. This makes it easy to track objects in video streams and perform subsequent analytics. Here's why you should consider using Ultralytics YOLO for your object tracking needs:

- **Efficiency:** Process video streams in real-time without compromising [accuracy](https://www.ultralytics.com/glossary/accuracy).
- **Flexibility:** Supports multiple tracking algorithms and configurations.
- **Ease of Use:** Simple Python API and CLI options for quick integration and deployment.
- **Customizability:** Easy to use with custom-trained YOLO models, allowing integration into domain-specific applications.

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

## Features at a Glance

Ultralytics YOLO extends its object detection features to provide robust and versatile object tracking:

- **Real-Time Tracking:** Seamlessly track objects in high-frame-rate videos.
- **Multiple Tracker Support:** Choose from a variety of established tracking algorithms.
- **Customizable Tracker Configurations:** Tailor the tracking algorithm to meet specific requirements by adjusting various parameters.

## Available Trackers

Ultralytics YOLO supports the following tracking algorithms. They can be enabled by passing the relevant YAML configuration file such as `tracker=tracker_type.yaml`:

- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) Kalman + IoU tracker with optional ReID and global motion compensation. Strong default for moving-camera scenes. Use `botsort.yaml`.
- [ByteTrack](https://github.com/FoundationVision/ByteTrack) Two-stage motion-only tracker that recovers low-confidence detections in a second association pass. Lightweight baseline. Use `bytetrack.yaml`.
- [OC-SORT](https://arxiv.org/abs/2203.14360) Observation-centric SORT with virtual-trajectory re-update (ORU), velocity-direction consistency (OCM), and last-observation recovery (OCR). No appearance model. Robust to non-linear motion. Use `ocsort.yaml`.
- [Deep OC-SORT](https://arxiv.org/abs/2302.11813) OC-SORT plus adaptive ReID appearance fusion, dynamic appearance EMA, and camera motion compensation. Best of motion + appearance for moving cameras and crowded scenes. Use `deepocsort.yaml`.
- [FastTracker](https://arxiv.org/abs/2508.14370) Occlusion-aware ByteTrack variant with Kalman rollback, motion dampening during occlusion, and init-IoU suppression to reduce duplicate IDs. Real-time, no extra network. Use `fasttrack.yaml`.
- [TrackTrack](https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf) Multi-cue (HMIoU + ReID + confidence + corner-angle) iterative association with Track-Aware Initialization. Strong on crowded pedestrians where duplicate IDs are a problem. Use `tracktrack.yaml`.

The default tracker is BoT-SORT.

### Quick comparison

| Tracker      | Motion model               | Appearance / ReID         | Camera motion compensation | Occlusion handling                              | Best for                                                       |
| ------------ | -------------------------- | ------------------------- | -------------------------- | ----------------------------------------------- | -------------------------------------------------------------- |
| ByteTrack    | Linear Kalman              | None                      | No                         | Two-stage low-conf rescue                       | Lightweight baseline; static cameras                           |
| BoT-SORT     | Linear Kalman              | Optional (`with_reid`)    | Yes (sparseOptFlow / ECC)  | Track buffer + ReID rebinding                   | Default; moving cameras with optional ReID                     |
| OC-SORT      | Observation-centric Kalman | None                      | No                         | ORU, OCM, OCR re-update from last observation   | Non-linear motion (sports, abrupt turns) without ReID overhead |
| Deep OC-SORT | Observation-centric Kalman | Optional (`with_reid`)    | Optional (`gmc_method`)    | OC-SORT + adaptive appearance EMA               | Crowded / moving-camera scenes where ID swaps matter           |
| FastTracker  | Linear Kalman + rollback   | None                      | No                         | Kalman rollback + bbox enlargement on occlusion | Real-time crowd / queue / sports with frequent partial overlap |
| TrackTrack   | Linear Kalman (NSA)        | Optional (HMIoU fallback) | Yes (sparseOptFlow / ECC)  | Iterative multi-cue association + TAI           | Crowded pedestrians where duplicate IDs are a problem          |

### BoT-SORT

[BoT-SORT](https://github.com/NirAharon/BoT-SORT) (Aharon et al., 2022) is the default tracker in Ultralytics YOLO. It builds on top of ByteTrack's two-stage matching and adds two pieces that help under camera motion and short occlusions:

- **Camera Motion Compensation (CMC / GMC):** an affine warp estimated each frame (sparse optical flow by default; ORB / ECC also available) is applied to every track's Kalman state so the predicted box compensates for camera motion before IoU is computed.
- **Optional ReID:** appearance embeddings can be attached to detections and used in the cost matrix as `min(IoU, appearance)` fusion, gated by `proximity_thresh` and `appearance_thresh`. Disabled by default; enable with `with_reid: True` and a `model:` path. See [Enabling Re-Identification (ReID)](#enabling-re-identification-reid).

#### When to use BoT-SORT

- General-purpose default picks up where ByteTrack falls short under handheld / drone / vehicle-mounted camera motion.
- Scenes with short partial occlusions where the IoU+Kalman fallback already does most of the work, and you want a low-friction starting point.
- Add ReID only when ID swaps in look-alike crowds become the dominant error.

#### BoT-SORT-specific arguments

In addition to the [shared tracker arguments](#tracker-arguments), BoT-SORT exposes:

| **Parameter**       | **Valid Values or Ranges**                    | **Description**                                                                                                          |
| ------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `gmc_method`        | `sparseOptFlow`, `orb`, `sift`, `ecc`, `none` | Camera-motion-compensation backend. `sparseOptFlow` is the speed/quality default. `none` disables CMC.                   |
| `with_reid`         | `True`, `False`                               | Enable appearance-based matching. Off by default.                                                                        |
| `model`             | `auto` or path to a ReID model                | ReID model. `auto` uses native YOLO features when available; otherwise pass a `.torchscript` / `.onnx` / `.engine` path. |
| `proximity_thresh`  | `0.0-1.0`                                     | Minimum IoU before appearance features are considered. Prevents far-away embeddings from polluting matches.              |
| `appearance_thresh` | `0.0-1.0`                                     | Minimum cosine similarity required for a ReID match. Raise to reduce identity swaps in look-alike scenes.                |

#### Example: running BoT-SORT

!!! example "BoT-SORT (default)"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        # `tracker="botsort.yaml"` is the default; passing it explicitly is optional.
        results = model.track("https://youtu.be/LNwODJXcvt4", show=True, tracker="botsort.yaml")
        ```

    === "CLI"

        ```bash
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4" tracker="botsort.yaml"
        ```

#### Tuning tips

- **Static camera:** set `gmc_method: none` to skip the warp computation; saves a few ms/frame with no quality loss.
- **Heavy camera motion:** stick with `sparseOptFlow`; `ecc` is more accurate but slower.
- **Look-alike crowds:** turn on `with_reid: True` and tune `appearance_thresh` upward (e.g. 0.85+) to require strong appearance agreement before swapping IDs.

### ByteTrack

[ByteTrack](https://github.com/FoundationVision/ByteTrack) (Zhang et al., ECCV 2022) is the lightweight baseline. Its single trick is the two-stage association:

- **Stage 1 (high-confidence):** match high-score detections (`scores ≥ track_high_thresh`) against active tracks using IoU on Kalman predictions.
- **Stage 2 (low-confidence rescue):** the leftover unmatched tracks are re-tried against low-score detections (`track_low_thresh < scores < track_high_thresh`). Detections that the detector almost rejected can keep an existing track alive through brief partial occlusion.

There is no appearance model and no camera-motion compensation—just linear Kalman + IoU + the second pass.

#### When to use ByteTrack

- Static or near-static cameras where the detector is the dominant cost and you want minimum tracker overhead.
- Detection-only pipelines where motion is mostly linear and you don't need ReID or CMC.
- Reproducing baselines or comparing other trackers against the simplest sensible reference.

#### ByteTrack-specific arguments

ByteTrack uses only the [shared tracker arguments](#tracker-arguments). There are no ByteTrack-specific extras. The `track_high_thresh` / `track_low_thresh` / `match_thresh` triple controls almost all of its behavior.

#### Example: running ByteTrack

!!! example "ByteTrack"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        results = model.track("https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")
        ```

    === "CLI"

        ```bash
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4" tracker="bytetrack.yaml"
        ```

#### Tuning tips

- **Detector is noisy:** lower `track_low_thresh` so the second-stage pass has more candidates to rescue tracks.
- **Detector is high-recall already:** raise `track_high_thresh` so only confident detections start tracks; reduces fragmented IDs.
- **Frequent ID flicker:** raise `track_buffer` so briefly-missed tracks survive; raise `match_thresh` only if you see false re-bindings.

### OC-SORT

[OC-SORT](https://arxiv.org/abs/2203.14360) (Cao et al., CVPR 2023) is an observation-centric extension of [SORT](https://arxiv.org/abs/1602.00763) that targets the failure modes of pure motion-based trackers under occlusion and non-linear motion. It keeps SORT's lightweight design (no appearance features by default) and adds three observation-centric corrections:

- **Observation-Centric Re-update (ORU):** when a track is re-associated after a gap, OC-SORT replays a virtual trajectory between the last reliable observation and the current detection, then re-runs the Kalman update along that path. This repairs the velocity that drifted while the track was unmatched.
- **Observation-Centric Momentum (OCM):** the cost matrix adds a velocity-direction-consistency term computed from observations within a `delta_t` window, so detections moving in the wrong direction are penalized even if they are spatially close.
- **Observation-Centric Recovery (OCR):** unmatched detections are re-checked against recently lost tracks using IoU against their last observation rather than the predicted state, recovering tracks the Kalman filter has already drifted away from.

#### When to use OC-SORT

- Detection-only pipelines where you don't want the cost or dependency of a ReID model.
- Scenes with non-linear motion (sports, dancing, abrupt direction changes) where vanilla SORT/ByteTrack drift after short occlusions.
- Static-camera footage; OC-SORT itself does not perform global motion compensation.

#### OC-SORT-specific arguments

In addition to the shared arguments (`track_high_thresh`, `track_low_thresh`, `new_track_thresh`, `track_buffer`, `match_thresh`, `fuse_score`), OC-SORT exposes the following parameters in [`ocsort.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/ocsort.yaml):

| **Parameter** | **Valid Values or Ranges** | **Description**                                                                                                           |
| ------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `delta_t`     | `>=1`                      | Temporal window (in frames) used to compute the observation-based velocity direction for OCM. Larger values smooth more.  |
| `inertia`     | `0.0-1.0`                  | Weight of the velocity-consistency cost added to the association matrix. Higher values penalize sudden direction changes. |
| `use_byte`    | `True`, `False`            | Enable a ByteTrack-style second association pass over low-confidence detections (`track_low_thresh`). Improves recall.    |

#### Example: running OC-SORT

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        results = model.track(
            source="https://youtu.be/LNwODJXcvt4",
            tracker="ocsort.yaml",
            show=True,
        )
        ```

    === "CLI"

        ```bash
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4" tracker="ocsort.yaml"
        ```

#### Tuning tips

- **Non-linear motion:** raise `inertia` (e.g. `0.3-0.4`) so direction consistency dominates near occlusions.
- **Sparse detections:** enable `use_byte: True` to recover low-confidence boxes via a ByteTrack-style second pass.
- **Long occlusions:** raise `track_buffer` so OCR has more lost tracks to rebind against.

### Deep OC-SORT

[Deep OC-SORT](https://arxiv.org/abs/2302.11813) (Maggiolino et al., 2023) augments [OC-SORT](#oc-sort) with appearance information and camera-motion compensation, while keeping its observation-centric corrections. It addresses the two main remaining weaknesses of OC-SORT drift on moving cameras and ID swaps between visually different but spatially close objects:

- **Adaptive appearance fusion:** detection embeddings are fused into the cost matrix on top of IoU, with the appearance weight modulated by detection confidence and pairwise box overlap so noisy embeddings can't override a strong motion cue.
- **Dynamic appearance EMA (`alpha_fixed_emb`):** each track's embedding is updated with an EMA whose smoothing factor adapts to the current detection's confidence; high-confidence updates change the embedding faster, low-confidence ones barely move it.
- **Camera Motion Compensation (CMC/GMC):** Kalman states are warped frame-to-frame using a rigid transform estimated by sparse optical flow, ORB, or ECC, removing apparent motion caused by the camera itself.

#### When to use Deep OC-SORT

- Crowded or visually diverse scenes where OC-SORT alone produces ID swaps between spatially-close objects.
- Moving-camera footage (drones, dashcams, body cams) where motion compensation makes the difference between drift and stable tracks.
- You can afford the extra inference cost of a ReID model. ReID is opt-in (`with_reid: True`). Deep OC-SORT still works with motion + CMC only when ReID is off.

#### Deep OC-SORT-specific arguments

In addition to all [OC-SORT arguments](#oc-sort-specific-arguments) and the shared tracking arguments, Deep OC-SORT adds the following in [`deepocsort.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/deepocsort.yaml):

| **Parameter**       | **Valid Values or Ranges**                    | **Description**                                                                                                                                       |
| ------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `with_reid`         | `True`, `False`                               | Enable appearance-based matching. Off by default; turn on to add the ReID cost term.                                                                  |
| `model`             | `auto`, exported ReID model file              | ReID model. `auto` reuses native YOLO features; otherwise pass an exported file (`.torchscript`, `.onnx`, `.engine`, …) loaded via `AutoBackend`.     |
| `proximity_thresh`  | `0.0-1.0`                                     | Minimum IoU before appearance features are considered in the cost. Prevents far-away embeddings from polluting matches.                               |
| `appearance_thresh` | `0.0-1.0`                                     | Minimum cosine similarity required for a ReID match. Raise to reduce identity swaps in look-alike scenes.                                             |
| `alpha_fixed_emb`   | `0.0-1.0`                                     | Base EMA factor for track-embedding updates. Higher values preserve the older embedding longer; the actual rate is modulated by detection confidence. |
| `gmc_method`        | `sparseOptFlow`, `orb`, `sift`, `ecc`, `none` | Global motion compensation method. Use `sparseOptFlow` for moving cameras; `none` for static cameras.                                                 |

#### Example: running Deep OC-SORT

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # By default Deep OC-SORT runs motion + OC-SORT corrections only.
        # To enable appearance ReID and/or GMC, copy deepocsort.yaml and flip
        # `with_reid: True` and/or `gmc_method: sparseOptFlow`.
        model = YOLO("yolo26n.pt")
        results = model.track(
            source="path/to/dashcam.mp4",
            tracker="deepocsort.yaml",
            persist=True,
            show=True,
        )
        ```

    === "CLI"

        ```bash
        yolo track model=yolo26n.pt source="path/to/dashcam.mp4" tracker="deepocsort.yaml"
        ```

#### Tuning tips

- **Identity swaps in crowds:** raise `appearance_thresh` (e.g. `0.92-0.95`) and lower `alpha_fixed_emb` so embeddings adapt more slowly.
- **Moving camera:** keep `gmc_method: sparseOptFlow`. If GMC is the bottleneck, downscale the input or fall back to OC-SORT for static segments.
- **Need lower latency:** keep `with_reid: False` (the default) for motion + CMC only; turn ReID on only when ID swaps in look-alike crowds dominate the error.

### FastTracker

[FastTracker](https://arxiv.org/abs/2508.14370) is an occlusion-aware extension of [ByteTrack](https://github.com/FoundationVision/ByteTrack) designed for real-time use. It keeps ByteTrack's two-stage association (high- then low-confidence) and adds lightweight, per-track occlusion handling that runs without any appearance model:

- **Occlusion detection:** at each frame, FastTracker measures how much of every active track is covered by other active tracks. Once coverage exceeds `occ_cover_thresh`, the track is flagged occluded.
- **Kalman rollback on occlusion onset:** when occlusion is first detected, the track's Kalman state is rolled back to a recent pre-occlusion frame using ring-buffered history (`reset_pos_offset_occ` for position, `reset_velocity_offset_occ` for velocity), so the filter doesn't carry an already-corrupted estimate forward.
- **Motion dampening and search expansion while occluded:** velocity is dampened by `dampen_motion_occ` and the predicted bbox is enlarged by `enlarge_bbox_occ` so the track can re-acquire the target without drifting.
- **Re-appearance window:** a short sticky flag (`occ_reappear_window`) keeps recently-occluded lost tracks re-findable for longer than ordinary lost tracks.
- **Init-IoU suppression:** new tracks are not initialized on top of an already-active track when their IoU exceeds `init_iou_suppress`, which sharply reduces duplicate IDs around partial occlusions.

#### When to use FastTracker

- Real-time detection-only pipelines where ByteTrack drops IDs around partial occlusions and you don't want the cost of ReID.
- Crowd / queue / retail / sports scenes with frequent target-on-target overlap.
- Latency-sensitive deployments. FastTracker has no extra network in the loop.

#### FastTracker-specific arguments

In addition to the shared arguments (`track_high_thresh`, `track_low_thresh`, `new_track_thresh`, `track_buffer`, `match_thresh`, `fuse_score`), FastTracker exposes the following parameters in [`fasttrack.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/fasttrack.yaml):

| **Parameter**               | **Valid Values or Ranges** | **Description**                                                                                                                                       |
| --------------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `reset_velocity_offset_occ` | `>=0`                      | Number of history frames back to restore the Kalman velocity from on occlusion onset.                                                                 |
| `reset_pos_offset_occ`      | `>=0`                      | Number of history frames back to restore the Kalman position from on occlusion onset.                                                                 |
| `enlarge_bbox_occ`          | `>=1.0`                    | One-shot height scaling applied to the predicted bbox while occluded so the search region widens (width scales proportionally via XYAH aspect ratio). |
| `dampen_motion_occ`         | `0.0-1.0`                  | Multiplicative factor applied to velocity while occluded. Lower values make the track "slow down" through the occlusion.                              |
| `active_occ_to_lost_thresh` | `>=1`                      | Maximum consecutive occluded frames before an active track is moved to the lost pool.                                                                 |
| `occ_cover_thresh`          | `0.0-1.0`                  | Fraction of a track's area that must be covered by another active track to declare occlusion.                                                         |
| `occ_reappear_window`       | `>=0`                      | Frames a recently-occluded lost track stays preferentially re-findable. Independent of `track_buffer`.                                                |
| `init_iou_suppress`         | `0.0-1.0`                  | Suppress new-track initialization if its IoU with any active track exceeds this. Set to `1.0` to disable suppression.                                 |

#### Example: running FastTracker

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        results = model.track(
            source="https://youtu.be/LNwODJXcvt4",
            tracker="fasttrack.yaml",
            show=True,
        )
        ```

    === "CLI"

        ```bash
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4" tracker="fasttrack.yaml"
        ```

#### Tuning tips

- **Frequent partial occlusions:** lower `occ_cover_thresh` (e.g. `0.5-0.6`) so occlusion is detected earlier and rollback kicks in sooner.
- **Duplicate IDs around overlap:** lower `init_iou_suppress` (e.g. `0.5`) to be stricter about spawning a new track on top of an existing one.
- **Long occlusions:** raise `occ_reappear_window` and `track_buffer` together so recently-occluded tracks survive longer.
- **Fast-moving targets:** raise `dampen_motion_occ` (closer to `1.0`) and lower `enlarge_bbox_occ` to keep the search region tight.

### TrackTrack

[TrackTrack](https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf) (Shim et al., CVPR 2025) is a track-focused online multi-object tracker. Unlike ByteTrack and BoT-SORT which primarily look at matching from the detection side, TrackTrack reasons from each track's perspective and combines multiple cues in a single cost matrix. It introduces two main ideas on top of standard tracking-by-detection:

- **Track-Perspective-Based Association (TPA):** a multi-cue cost combining HMIoU (IoU modulated by vertical overlap), cosine ReID distance, confidence-projection distance, and corner-angle distance. The assignment is solved iteratively with a threshold that relaxes at each iteration, so high-quality matches are locked in first and harder pairs are revisited with looser gates. Low-confidence detections and detections recovered from a looser secondary NMS pass (`D_del`) are added to the cost with penalty terms so they can still rebind lost tracks without stealing strong matches.
- **Track-Aware Initialization (TAI):** before spawning a new track, candidate detections are suppressed if they heavily overlap an existing track or a stronger detection. This greatly reduces duplicate IDs in crowded scenes.

TrackTrack also uses NSA-Kalman updates (measurement-noise scaled by detection confidence) and score-adaptive EMA smoothing of ReID features. Global motion compensation (GMC) is enabled by default via sparse optical flow and can be tuned or skipped for speed.

#### When to use TrackTrack

- Crowded scenes with frequent occlusion (pedestrians, retail, sports) where duplicate IDs are a problem.
- Scenarios where you want good performance without necessarily enabling ReID. The multi-cue cost already uses HMIoU, confidence projection, and corner-angle cues.
- Moving-camera footage where GMC helps, but you also want one-knob control over the speed/accuracy trade-off via `gmc_downscale`, `gmc_max_corners`, and `gmc_skip_frames`.

#### TrackTrack-specific arguments

In addition to the shared arguments (`track_high_thresh`, `track_low_thresh`, `new_track_thresh`, `track_buffer`, `match_thresh`, `gmc_method`, `with_reid`, `model`), TrackTrack exposes the following parameters in [`tracktrack.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/tracktrack.yaml):

| **Parameter**     | **Valid Values or Ranges** | **Description**                                                                                                         |
| ----------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `iou_weight`      | `0.0-1.0`                  | Weight of the HMIoU distance term in the multi-cue cost matrix.                                                         |
| `reid_weight`     | `0.0-1.0`                  | Weight of the cosine ReID distance. If ReID is disabled, HMIoU is used as a fallback so this weight still applies.      |
| `conf_weight`     | `0.0-1.0`                  | Weight of the confidence-projection distance (tracks extrapolate their score and are compared to detection confidence). |
| `angle_weight`    | `0.0-1.0`                  | Weight of the corner-angle distance between each track's per-corner velocity and the track-to-detection direction.      |
| `det_thr`         | `0.0-1.0`                  | Detection threshold that separates high- from low-confidence detections for the penalty logic.                          |
| `penalty_p`       | `0.0-1.0`                  | Cost penalty added to low-confidence detections (`D_low`) during association.                                           |
| `penalty_q`       | `0.0-1.0`                  | Cost penalty added to detections recovered by the looser secondary NMS (`D_del`).                                       |
| `reduce_step`     | `0.0-1.0`                  | Amount by which the match threshold is relaxed at each iteration of the iterative assignment.                           |
| `tai_thr`         | `0.0-1.0`                  | IoU threshold used by Track-Aware Initialization NMS to suppress duplicate-looking spawn candidates.                    |
| `init_thr`        | `0.0-1.0`                  | Minimum detection score required to initialize a new track from TAI.                                                    |
| `min_track_len`   | `>=0`                      | Minimum number of successful updates before a newly spawned track is confirmed (promoted to `Tracked`).                 |
| `gmc_downscale`   | `>=1`                      | Downscale factor for the GMC input image. Higher values are faster but less accurate.                                   |
| `gmc_max_corners` | `>=1`                      | Max keypoints for `sparseOptFlow`. 200 is typically enough for an affine warp.                                          |
| `gmc_skip_frames` | `>=0`                      | Skip `N` frames between GMC updates and reuse the cached warp. `0` = recompute every frame.                             |

#### Example: running TrackTrack

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a detection, segmentation, or pose model
        model = YOLO("yolo26n.pt")

        # Run tracking with TrackTrack
        results = model.track(
            source="https://youtu.be/LNwODJXcvt4",
            tracker="tracktrack.yaml",
            show=True,
        )
        ```

    === "CLI"

        ```bash
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4" tracker="tracktrack.yaml"
        ```

#### Example: TrackTrack with ReID

ReID is optional in TrackTrack. When it's disabled, the `reid_weight` falls back to HMIoU. Enabling it gives appearance-based rebinding across longer occlusions, at some additional cost:

!!! example "TrackTrack + ReID"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Copy ultralytics/cfg/trackers/tracktrack.yaml and set:
        #   with_reid: True
        #   model: auto       # or path to an exported ReID model (.torchscript / .onnx / .engine)
        model = YOLO("yolo26n.pt")
        results = model.track(
            source="path/to/crowded_scene.mp4",
            tracker="custom_tracktrack.yaml",
            persist=True,
            show=True,
        )
        ```

For a per-frame loop with track IDs, see [Persisting Tracks Loop](#persisting-tracks-loop) under [Python Examples](#python-examples). That pattern works for any tracker, just swap the `tracker=` argument.

#### Tuning tips

- **Crowded pedestrians:** keep the defaults, but consider lowering `tai_thr` (e.g. `0.45`) to suppress more duplicate spawns, and raising `track_buffer` so lost tracks survive longer occlusions.
- **Fast camera motion:** keep `gmc_method: sparseOptFlow`, lower `gmc_downscale` (e.g. `2`) and/or raise `gmc_max_corners`. If GMC becomes a bottleneck, set `gmc_skip_frames: 1` or `2`.
- **Small/fast objects:** raise `angle_weight` slightly so direction-of-motion contributes more, and lower `min_track_len` to confirm tracks faster.
- **Enable ReID only when needed:** it adds inference cost; for scenes with short occlusions, the default multi-cue cost is usually sufficient.

## Tracking

To run the tracker on video streams, use a trained Detect, Segment, or Pose model such as YOLO26n, YOLO26n-seg, or YOLO26n-pose. You can train custom models locally or on cloud GPUs through [Ultralytics Platform](https://platform.ultralytics.com).

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load an official or custom model
        model = YOLO("yolo26n.pt")  # Load an official Detect model
        model = YOLO("yolo26n-seg.pt")  # Load an official Segment model
        model = YOLO("yolo26n-pose.pt")  # Load an official Pose model
        model = YOLO("path/to/best.pt")  # Load a custom-trained model

        # Perform tracking with the model
        results = model.track("https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
        results = model.track("https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # with ByteTrack
        ```

    === "CLI"

        ```bash
        # Perform tracking with various models using the command line interface
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4"      # Official Detect model
        yolo track model=yolo26n-seg.pt source="https://youtu.be/LNwODJXcvt4"  # Official Segment model
        yolo track model=yolo26n-pose.pt source="https://youtu.be/LNwODJXcvt4" # Official Pose model
        yolo track model=path/to/best.pt source="https://youtu.be/LNwODJXcvt4" # Custom trained model

        # Track using ByteTrack tracker
        yolo track model=path/to/best.pt source="https://youtu.be/LNwODJXcvt4" tracker="bytetrack.yaml"
        ```

As can be seen in the above usage, tracking is available for all Detect, Segment, and Pose models run on videos or streaming sources.

## Configuration

### Tracking Arguments

Tracking configuration shares properties with Predict mode, such as `conf`, `iou`, and `show`. For further configurations, refer to the [Predict](../modes/predict.md#inference-arguments) model page.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Configure the tracking parameters and run the tracker
        model = YOLO("yolo26n.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.1, iou=0.7, show=True)
        ```

    === "CLI"

        ```bash
        # Configure tracking parameters and run the tracker using the command line interface
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.1 iou=0.7 show
        ```

### Tracker Selection

Ultralytics also allows you to use a modified tracker configuration file. To do this, simply make a copy of a tracker config file (for example, `custom_tracker.yaml`) from [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) and modify any configurations (except the `tracker_type`) as per your needs.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the model and run the tracker with a custom configuration file
        model = YOLO("yolo26n.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker="custom_tracker.yaml")
        ```

    === "CLI"

        ```bash
        # Load the model and run the tracker with a custom configuration file using the command line interface
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
        ```

Refer to [Tracker Arguments](#tracker-arguments) section for a detailed description of each parameter.

### Tracker Arguments

Some tracking behaviors can be fine-tuned by editing the YAML configuration files specific to each tracking algorithm. These files define parameters like thresholds, buffers, and matching logic:

- [`botsort.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/botsort.yaml)
- [`bytetrack.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml)
- [`ocsort.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/ocsort.yaml)
- [`deepocsort.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/deepocsort.yaml)
- [`fasttrack.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/fasttrack.yaml)
- [`tracktrack.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/tracktrack.yaml)

The following table provides a description of each parameter:

!!! warning "Tracker Threshold Information"

    If a detection's confidence score falls below [`track_high_thresh`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml#L5), the tracker will not update that object, resulting in no active tracks.

| **Parameter**       | **Valid Values or Ranges**                                                | **Description**                                                                                                                                                                   |
| ------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tracker_type`      | `botsort`, `bytetrack`, `ocsort`, `deepocsort`, `fasttrack`, `tracktrack` | Specifies the tracker type. One of `botsort`, `bytetrack`, `ocsort`, `deepocsort`, `fasttrack`, or `tracktrack`.                                                                  |
| `track_high_thresh` | `0.0-1.0`                                                                 | Threshold used for the first association during tracking. Affects how confidently a detection is matched to an existing track.                                                    |
| `track_low_thresh`  | `0.0-1.0`                                                                 | Threshold for the second association during tracking. Used when the first association fails, with more lenient criteria.                                                          |
| `new_track_thresh`  | `0.0-1.0`                                                                 | Threshold to initialize a new track if the detection does not match any existing tracks. Controls when a new object is considered to appear.                                      |
| `track_buffer`      | `>=0`                                                                     | Buffer used to indicate the number of frames lost tracks should be kept alive before getting removed. Higher value means more tolerance for occlusion.                            |
| `match_thresh`      | `0.0-1.0`                                                                 | Threshold for matching tracks. Higher values make the matching more lenient.                                                                                                      |
| `fuse_score`        | `True`, `False`                                                           | Determines whether to fuse confidence scores with IoU distances before matching. Helps balance spatial and confidence information when associating.                               |
| `gmc_method`        | `sparseOptFlow`, `orb`, `sift`, `ecc`, `none`                             | Method used for global motion compensation. Helps account for camera movement to improve tracking.                                                                                |
| `proximity_thresh`  | `0.0-1.0`                                                                 | Minimum IoU required for a valid match with ReID (Re-identification). Ensures spatial closeness before using appearance cues.                                                     |
| `appearance_thresh` | `0.0-1.0`                                                                 | Minimum appearance similarity required for ReID. Sets how visually similar two detections must be to be linked.                                                                   |
| `with_reid`         | `True`, `False`                                                           | Indicates whether to use ReID. Enables appearance-based matching for better tracking across occlusions. Supported by BoT-SORT, Deep OC-SORT, and TrackTrack.                      |
| `model`             | `auto` or path to an exported file                                        | ReID model. `auto` uses native YOLO features (falls back to `yolo26n-cls.pt`); otherwise pass a `.torchscript`, `.onnx`, `.engine`, `.openvino`, … file loaded via `AutoBackend`. |

#### Tracker-specific arguments

Each algorithm exposes additional knobs on top of the shared parameters above. See the per-tracker sections below for the full description and tuning advice:

| Tracker      | Config file       | Specific arguments                                                                                                                                                                                                          | Section                                                             |
| ------------ | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| BoT-SORT     | `botsort.yaml`    | `gmc_method`, `proximity_thresh`, `appearance_thresh`, `with_reid`, `model`                                                                                                                                                 | [BoT-SORT-specific arguments](#bot-sort-specific-arguments)         |
| ByteTrack    | `bytetrack.yaml`  | None uses only the shared arguments.                                                                                                                                                                                        | [ByteTrack-specific arguments](#bytetrack-specific-arguments)       |
| OC-SORT      | `ocsort.yaml`     | `delta_t`, `inertia`, `use_byte`                                                                                                                                                                                            | [OC-SORT-specific arguments](#oc-sort-specific-arguments)           |
| Deep OC-SORT | `deepocsort.yaml` | OC-SORT args + `with_reid`, `model`, `proximity_thresh`, `appearance_thresh`, `alpha_fixed_emb`, `gmc_method`                                                                                                               | [Deep OC-SORT-specific arguments](#deep-oc-sort-specific-arguments) |
| FastTracker  | `fasttrack.yaml`  | `reset_velocity_offset_occ`, `reset_pos_offset_occ`, `enlarge_bbox_occ`, `dampen_motion_occ`, `active_occ_to_lost_thresh`, `occ_cover_thresh`, `occ_reappear_window`, `init_iou_suppress`                                   | [FastTracker-specific arguments](#fasttracker-specific-arguments)   |
| TrackTrack   | `tracktrack.yaml` | `iou_weight`, `reid_weight`, `conf_weight`, `angle_weight`, `det_thr`, `penalty_p`, `penalty_q`, `reduce_step`, `tai_thr`, `init_thr`, `min_track_len`, `gmc_method`, `gmc_downscale`, `gmc_max_corners`, `gmc_skip_frames` | [TrackTrack-specific arguments](#tracktrack-specific-arguments)     |

### Enabling Re-Identification (ReID)

By default, ReID is turned off to minimize performance overhead. Enabling it is simple—just set `with_reid: True` in the [tracker configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/botsort.yaml). You can also customize the `model` used for ReID, allowing you to trade off accuracy and speed depending on your use case:

- **Native features (`model: auto`)**: This leverages features directly from the YOLO detector for ReID, adding minimal overhead. It's ideal when you need some level of ReID without significantly impacting performance. If the detector doesn't support native features, it automatically falls back to using `yolo26n-cls.pt`.
- **Exported ReID model**: Point `model:` at an exported model file (`.torchscript`, `.onnx`, `.engine`, `.openvino`, etc.) that outputs an embedding tensor. The encoder is loaded via `AutoBackend`, so any export format Ultralytics supports works without code changes. This produces more discriminative embeddings than native features at the cost of an extra forward pass per crop. Set the file path as `model:` in your tracker config to use it.

For better performance, especially when using a separate classification model for ReID, you can export it to a faster backend like TensorRT:

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

Once exported, you can point to the TensorRT model path in your tracker config, and it will be used for ReID during tracking.

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

Here is a Python script using [OpenCV](https://www.ultralytics.com/glossary/opencv) (`cv2`) and YOLO26 to run object tracking on video frames. This script assumes the necessary packages (`opencv-python` and `ultralytics`) are already installed. The `persist=True` argument tells the tracker that the current image or frame is the next in a sequence and to expect tracks from the previous image in the current image.

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

Please note the change from `model(frame)` to `model.track(frame)`, which enables object tracking instead of simple detection. This modified script will run the tracker on each frame of the video, visualize the results, and display them in a window. The loop can be exited by pressing 'q'.

### Plotting Tracks Over Time

Visualizing object tracks over consecutive frames can provide valuable insights into the movement patterns and behavior of detected objects within a video. With Ultralytics YOLO26, plotting these tracks is a seamless and efficient process.

In the following example, we demonstrate how to utilize YOLO26's tracking capabilities to plot the movement of detected objects across multiple video frames. This script involves opening a video file, reading it frame by frame, and utilizing the YOLO model to identify and track various objects. By retaining the center points of the detected bounding boxes and connecting them, we can draw lines that represent the paths followed by the tracked objects.

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

Multithreaded tracking provides the capability to run object tracking on multiple video streams simultaneously. This is particularly useful when handling multiple video inputs, such as from multiple surveillance cameras, where concurrent processing can greatly enhance efficiency and performance.

In the provided Python script, we make use of Python's `threading` module to run multiple instances of the tracker concurrently. Each thread is responsible for running the tracker on one video file, and all the threads run simultaneously in the background.

To ensure that each thread receives the correct parameters (the video file, the model to use and the file index), we define a function `run_tracker_in_thread` that accepts these parameters and contains the main tracking loop. This function reads the video frame by frame, runs the tracker, and displays the results.

Two different models are used in this example: `yolo26n.pt` and `yolo26n-seg.pt`, each tracking objects in a different video file. The video files are specified in `SOURCES`.

The `daemon=True` parameter in `threading.Thread` means that these threads will be closed as soon as the main program finishes. We then start the threads with `start()` and use `join()` to make the main thread wait until both tracker threads have finished.

Finally, after all threads have completed their task, the windows displaying the results are closed using `cv2.destroyAllWindows()`.

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

This example can easily be extended to handle more video files and models by creating more threads and applying the same methodology.

## Contribute New Trackers

Are you proficient in multi-object tracking and have successfully implemented or adapted a tracking algorithm with Ultralytics YOLO? We invite you to contribute to our Trackers section in [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)! Your real-world applications and solutions could be invaluable for users working on tracking tasks.

By contributing to this section, you help expand the scope of tracking solutions available within the Ultralytics YOLO framework, adding another layer of functionality and utility for the community.

To initiate your contribution, please refer to our [Contributing Guide](../help/contributing.md) for comprehensive instructions on submitting a Pull Request (PR) 🛠️. We are excited to see what you bring to the table!

Together, let's enhance the tracking capabilities of the Ultralytics YOLO ecosystem 🙏!

[fish track]: https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/fish-tracking.avif
[people track]: https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/people-tracking.avif
[vehicle track]: https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/vehicle-tracking.avif

## FAQ

### What is Multi-Object Tracking and how does Ultralytics YOLO support it?

Multi-object tracking in video analytics involves both identifying objects and maintaining a unique ID for each detected object across video frames. Ultralytics YOLO supports this by providing real-time tracking along with object IDs, facilitating tasks such as security surveillance and sports analytics. The system ships with six built-in trackers [BoT-SORT](https://github.com/NirAharon/BoT-SORT), [ByteTrack](https://github.com/FoundationVision/ByteTrack), [OC-SORT](https://arxiv.org/abs/2203.14360), [Deep OC-SORT](https://arxiv.org/abs/2302.11813), [FastTracker](https://arxiv.org/abs/2508.14370), and [TrackTrack](https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf) each configurable via YAML. See the [Available Trackers](#available-trackers) comparison for picking one.

### How do I configure a custom tracker for Ultralytics YOLO?

You can configure a custom tracker by copying an existing tracker configuration file (e.g., `custom_tracker.yaml`) from the [Ultralytics tracker configuration directory](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) and modifying parameters as needed, except for the `tracker_type`. Use this file in your tracking model like so:

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

### How can I run object tracking on multiple video streams simultaneously?

To run object tracking on multiple video streams simultaneously, you can use Python's `threading` module. Each thread will handle a separate video stream. Here's an example of how you can set this up:

!!! example "Multithreaded Tracking"

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

### When should I use TrackTrack instead of BoT-SORT or ByteTrack?

TrackTrack is a good choice when identity preservation is critical, especially in crowded scenes with frequent occlusion. It combines multiple cues (HMIoU, ReID, confidence projection, corner-angle distance) from each track's perspective and solves the association with an iterative threshold, which helps high-quality matches lock in first. Track-Aware Initialization (TAI) also suppresses duplicate spawns before a new ID is created, reducing ID fragmentation. ByteTrack remains the fastest option for simple scenes, BoT-SORT is a strong general-purpose baseline with optional ReID, and TrackTrack trades a bit of overhead for better behavior under occlusion and crowding. Enable it with `tracker="tracktrack.yaml"`:

!!! example

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    results = model.track(source="path/to/video.mp4", tracker="tracktrack.yaml")
    ```

### What are the real-world applications of multi-object tracking with Ultralytics YOLO?

Multi-object tracking with Ultralytics YOLO has numerous applications, including:

- **Transportation:** Vehicle tracking for traffic management and [autonomous driving](https://www.ultralytics.com/blog/ai-in-self-driving-cars).
- **Retail:** People tracking for in-store analytics and security.
- **Aquaculture:** Fish tracking for monitoring aquatic environments.
- **Sports Analytics:** Tracking players and equipment for performance analysis.
- **Security Systems:** [Monitoring suspicious activities](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8) and creating [security alarms](https://docs.ultralytics.com/guides/security-alarm-system).

These applications benefit from Ultralytics YOLO's ability to process high-frame-rate videos in real time with exceptional accuracy.

### How can I visualize object tracks over multiple video frames with Ultralytics YOLO?

To visualize object tracks over multiple video frames, you can use the YOLO model's tracking features along with OpenCV to draw the paths of detected objects. Here's an example script that demonstrates this:

!!! example "Plotting tracks over multiple video frames"

    ```python
    from collections import defaultdict

    import cv2
    import numpy as np

    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)
    track_history = defaultdict(lambda: [])

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True)
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            annotated_frame = results[0].plot()
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            cv2.imshow("YOLO26 Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    ```

This script will plot the tracking lines showing the movement paths of the tracked objects over time, providing valuable insights into object behavior and patterns.
