---
title: Real-Time Action Recognition with YOLO26
comments: true
description: Recognize human actions in real-time video by pairing Ultralytics YOLO26 tracking with TorchVision video classifiers pretrained on Kinetics-400.
keywords: action recognition, Ultralytics YOLO26, video classification, TorchVision, Kinetics-400, real-time, computer vision, ActionRecognition, yolo solutions action
---

# Action Recognition using Ultralytics YOLO26

The [ActionRecognition solution](../reference/solutions/action_recognition.md) combines [Ultralytics YOLO26](../models/yolo26.md) detection and [object tracking](../modes/track.md) with TorchVision video classification models to identify human actions in real-time video streams. It follows each person across frames and classifies their activity using models pretrained on [Kinetics-400](https://arxiv.org/abs/1705.06950).

<p align="center">
  <video width="1024" src="https://cdn.ul.run/v/10c887c2e5324dce8611c595673a56c0.mp4" autoplay loop muted playsinline aria-label="Action recognition classifying a tracked person dribbling a basketball"></video>
</p>

## Advantages of Action Recognition

- **Real-time Analysis:** Process video streams and identify actions as they happen.
- **Pretrained Models:** Uses TorchVision models trained on Kinetics-400 (400 action classes).
- **Multiple Architectures:** Supports S3D, R3D, Swin3D, and MViT models.
- **Integrated Tracking:** Combines YOLO detection with per-person action classification.

## Recognize Actions with YOLO26

!!! example "Action Recognition using Ultralytics YOLO"

    === "CLI"

        ```bash
        # Run an action recognition example
        yolo solutions action show=True

        # Pass a source video and pick a classifier backbone
        yolo solutions action source="path/to/video.mp4" video_classifier_model="swin3d_t"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("action_recognition.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        action = solutions.ActionRecognition(
            show=True,
            model="yolo26n.pt",
            video_classifier_model="s3d",  # TorchVision model
        )

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break
            results = action(im0)
            video_writer.write(results.plot_im)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

        For a live camera feed, pass the device index instead of a file path: `cv2.VideoCapture(0)`, and set `fps` manually if the camera reports 0.

### Video Classifier Models

All backbones are pretrained on Kinetics-400. Top-1 accuracy and parameter counts below are the values TorchVision publishes in its weight metadata:

| Model       | Top-1 | Parameters | Notes                                                                  |
| ----------- | ----- | ---------- | ---------------------------------------------------------------------- |
| `s3d`       | 68.4  | 8.3 M      | Default; smallest of the six                                           |
| `swin3d_t`  | 77.7  | 28.2 M     | Swin Transformer 3D (tiny)                                             |
| `r3d_18`    | 63.2  | 33.4 M     | Oldest backbone; available on the widest range of TorchVision releases |
| `mvit_v2_s` | 80.8  | 34.5 M     | Most accurate of the six                                               |
| `mvit_v1_b` | 78.5  | 36.6 M     | MViT v1 (base)                                                         |
| `swin3d_b`  | 79.4  | 88.0 M     | Swin Transformer 3D (base); heaviest                                   |

!!! note "TorchVision availability"

    `torchvision` installs alongside `ultralytics`, but the usable set of `video_classifier_model` values depends on
    your build: `ActionRecognition` offers only the backbones whose pretrained video weights are present and raises for
    the rest. `s3d` requires TorchVision 0.14 or newer, while `r3d_18` works from 0.13, so switch to `r3d_18` if
    an older build rejects the default, or upgrade with `pip install -U torchvision`.

### `ActionRecognition` Arguments

Here's a table with the `ActionRecognition` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "video_classifier_model", "crop_margin_percentage", "num_video_sequence_samples", "skip_frame", "video_cls_overlap_ratio", "line_width", "verbose"]) }}

The `ActionRecognition` solution also supports `track` arguments:

{% from "macros/solutions-track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "device"]) }}

Moreover, the following visualization options are available:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "show_conf", "show_labels"]) }}

`ActionRecognition` overrides `classes` to `[0]`, so it tracks and classifies people only; pass a different list to act on other classes. `device` is shared, selecting the device for both the tracker and the video classifier.

## FAQ

### What actions can Ultralytics YOLO26 action recognition detect?

The solution recognizes the 400 human actions in the Kinetics-400 label set, including `playing guitar`, `jogging`, `tap dancing`, and `cooking egg`. Class names are specific rather than generic, so check the [full class list](https://github.com/google-deepmind/kinetics-i3d/blob/master/data/label_map.txt) for the ones that match your footage.

### Why do I see "detecting..." instead of action labels?

A tracked box reads `2 person 0.89 | detecting...` until `ActionRecognition` has collected `num_video_sequence_samples` crops for it, which spans `num_video_sequence_samples * skip_frame` video frames, or 32 with the defaults. The placeholder is then replaced by the prediction and its confidence, as in `2 person 0.89 | jogging (0.97)`. Lowering `skip_frame` shortens the wait but narrows the time window the classifier sees, which can reduce accuracy.

### Which video classifier should I use?

Start with the default `s3d`, which is the smallest of the six backbones at 8.3 M parameters and still more accurate than `r3d_18`. Move to `mvit_v2_s` for the best accuracy, or `swin3d_t` for most of it at about a fifth fewer parameters. Swap in a detection model trained on [Ultralytics Platform](../platform/index.md) when your subjects are not covered by the default person class.

### How do I read the predicted actions in code?

`SolutionResults` exposes `action_labels` and `action_confs`, both keyed by track ID:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
action = solutions.ActionRecognition(show=False)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = action(im0)
    for track_id, label in results.action_labels.items():
        print(f"track {track_id}: {label} ({results.action_confs[track_id]:.2f})")

cap.release()
```
