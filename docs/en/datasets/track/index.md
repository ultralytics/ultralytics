---
comments: true
description: Learn how to use Multi-Object Tracking with YOLO. Explore dataset formats, tracking algorithms, and implementation examples using Python or CLI for real-time object tracking.
keywords: YOLO, Multi-Object Tracking, Tracking Datasets, Python Tracking Example, CLI Tracking Example, Object Detection, Ultralytics, AI, Machine Learning, BoT-SORT, ByteTrack, OC-SORT, Deep OC-SORT, FastTracker, TrackTrack
title: Multi-Object Tracking Datasets
---

# Multi-object Tracking Datasets Overview

Multi-object tracking is a critical component in video analytics that identifies objects and maintains unique IDs for each detected object across video frames. Ultralytics YOLO provides powerful tracking capabilities that can be applied to various domains including surveillance, sports analytics, and traffic monitoring.

## Dataset Format

Track mode does not train a tracker or require a tracker-specific dataset. Train a detection, segmentation, pose, or OBB model with that task's standard dataset format, then use the resulting weights with `model.track()` or `yolo track`. The selected tracker associates the model's predictions across frames at inference time.

## Available Trackers

Ultralytics YOLO supports the following tracking algorithms:

- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - Use `botsort.yaml` to enable this tracker
- [ByteTrack](https://github.com/FoundationVision/ByteTrack) - Use `bytetrack.yaml` to enable this tracker
- [OC-SORT](https://arxiv.org/abs/2203.14360) - Use `ocsort.yaml` to enable this tracker
- [Deep OC-SORT](https://arxiv.org/abs/2302.11813) - Use `deepocsort.yaml` to enable this tracker
- [FastTracker](https://arxiv.org/abs/2508.14370) - Use `fasttrack.yaml` to enable this tracker
- [TrackTrack](https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf) - Use `tracktrack.yaml` to enable this tracker (default)

## Usage

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.1, iou=0.7, show=True)
        ```

    === "CLI"

        ```bash
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.1 iou=0.7 show=True
        ```

## Persisting Tracks Between Frames

For continuous tracking across video frames, you can use the `persist=True` parameter:

!!! example

    === "Python"

        ```python
        import cv2

        from ultralytics import YOLO

        # Load the YOLO model
        model = YOLO("yolo26n.pt")

        # Open the video file
        cap = cv2.VideoCapture("path/to/video.mp4")

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                # Run tracking with persistence between frames
                results = model.track(frame, persist=True)

                # Visualize the results
                annotated_frame = results[0].plot()
                cv2.imshow("Tracking", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
        ```

## FAQ

### How do I use Multi-Object Tracking with Ultralytics YOLO?

To use Multi-Object Tracking with Ultralytics YOLO, you can start by using the Python or CLI examples provided. Here is how you can get started:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")  # Load the YOLO26 model
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.1, iou=0.7, show=True)
        ```

    === "CLI"

        ```bash
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.1 iou=0.7 show=True
        ```

These commands load the YOLO26 model and use it for tracking objects in the given video source with specific confidence (`conf`) and [Intersection over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (`iou`) thresholds. For more details, refer to the [track mode documentation](../../modes/track.md).

### Why should I use Ultralytics YOLO for multi-object tracking?

Ultralytics YOLO is a state-of-the-art [object detection](https://www.ultralytics.com/glossary/object-detection) model known for its real-time performance and high [accuracy](https://www.ultralytics.com/glossary/accuracy). Using YOLO for multi-object tracking provides several advantages:

- **Real-time tracking:** Achieve efficient and high-speed tracking ideal for dynamic environments.
- **Flexibility with pretrained models:** No need to train from scratch; simply use pretrained detection, segmentation, pose, or OBB models.
- **Ease of use:** Simple API integration with both Python and CLI makes setting up tracking pipelines straightforward.
- **Extensive documentation and community support:** Ultralytics provides comprehensive documentation and an active community forum to troubleshoot issues and enhance your tracking models.

For more details on setting up and using YOLO for tracking, visit our [track usage guide](../../modes/track.md).

### Can I use custom datasets for multi-object tracking with Ultralytics YOLO?

Yes. Train a detection, segmentation, pose, or OBB model on your custom dataset, then pass its weights to `model.track()`. The tracker itself runs during inference and does not require separate training data.

### How do I interpret the results from the Ultralytics YOLO tracking model?

After running a tracking job with Ultralytics YOLO, the results include various data points such as tracked object IDs, their bounding boxes, and the confidence scores. Here's a brief overview of how to interpret these results:

- **Tracked IDs:** Each object is assigned a unique ID, which helps in tracking it across frames.
- **Bounding boxes:** These indicate the location of tracked objects within the frame.
- **Confidence scores:** These reflect the model's confidence in detecting the tracked object.

For detailed guidance on interpreting and visualizing these results, refer to the [results handling guide](../../reference/engine/results.md).

### How can I customize the tracker configuration?

You can customize the tracker by creating a modified version of the tracker configuration file. Copy an existing tracker config file from [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers), modify the parameters as needed, and specify this file when running the tracker:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.track(source="video.mp4", tracker="custom_tracker.yaml")
```
