---
comments: true
description: Learn how to use Ultralytics YOLO for object tracking in video streams. Guides to use different trackers and customise tracker configurations.
keywords: Ultralytics, YOLO, object tracking, video streams, BoT-SORT, ByteTrack, Python guide, CLI guide
---

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418637-1d6250fd-1515-4c10-a844-a32818ae6d46.png">

Object tracking is a task that involves identifying the location and class of objects, then assigning a unique ID to that detection in video streams.

The output of tracker is the same as detection with an added object ID.

## Available Trackers

Ultralytics YOLO supports the following tracking algorithms. They can be enabled by passing the relevant YAML configuration file such as `tracker=tracker_type.yaml`:

* [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - Use `botsort.yaml` to enable this tracker.
* [ByteTrack](https://github.com/ifzhang/ByteTrack) - Use `bytetrack.yaml` to enable this tracker.

The default tracker is BoT-SORT.

## Tracking

To run the tracker on video streams, use a trained Detect, Segment or Pose model such as YOLOv8n, YOLOv8n-seg and YOLOv8n-pose.

!!! example ""

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load an official or custom model
        model = YOLO('yolov8n.pt')  # Load an official Detect model
        model = YOLO('yolov8n-seg.pt')  # Load an official Segment model
        model = YOLO('yolov8n-pose.pt')  # Load an official Pose model
        model = YOLO('path/to/best.pt')  # Load a custom trained model

        # Perform tracking with the model
        results = model.track(source="https://youtu.be/Zgi9g1ksQHc", show=True)  # Tracking with default tracker
        results = model.track(source="https://youtu.be/Zgi9g1ksQHc", show=True, tracker="bytetrack.yaml")  # Tracking with ByteTrack tracker
        ```

    === "CLI"

        ```bash
        # Perform tracking with various models using the command line interface
        yolo track model=yolov8n.pt source="https://youtu.be/Zgi9g1ksQHc"  # Official Detect model
        yolo track model=yolov8n-seg.pt source="https://youtu.be/Zgi9g1ksQHc"  # Official Segment model
        yolo track model=yolov8n-pose.pt source="https://youtu.be/Zgi9g1ksQHc"  # Official Pose model
        yolo track model=path/to/best.pt source="https://youtu.be/Zgi9g1ksQHc"  # Custom trained model

        # Track using ByteTrack tracker
        yolo track model=path/to/best.pt tracker="bytetrack.yaml" 
        ```

As can be seen in the above usage, tracking is available for all Detect, Segment and Pose models run on videos or streaming sources.

## Configuration

### Tracking Arguments

Tracking configuration shares properties with Predict mode, such as `conf`, `iou`, and `show`. For further configurations, refer to the [Predict](https://docs.ultralytics.com/modes/predict/) model page.

!!! example ""

    === "Python"

        ```python
        from ultralytics import YOLO

        # Configure the tracking parameters and run the tracker
        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/Zgi9g1ksQHc", conf=0.3, iou=0.5, show=True)
        ```

    === "CLI"

        ```bash
        # Configure tracking parameters and run the tracker using the command line interface
        yolo track model=yolov8n.pt source="https://youtu.be/Zgi9g1ksQHc" conf=0.3, iou=0.5 show
        ```

### Tracker Selection

Ultralytics also allows you to use a modified tracker configuration file. To do this, simply make a copy of a tracker config file (for example, `custom_tracker.yaml`) from [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) and modify any configurations (except the `tracker_type`) as per your needs.

!!! example ""

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the model and run the tracker with a custom configuration file
        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/Zgi9g1ksQHc", tracker='custom_tracker.yaml')
        ```

    === "CLI"

        ```bash
        # Load the model and run the tracker with a custom configuration file using the command line interface
        yolo track model=yolov8n.pt source="https://youtu.be/Zgi9g1ksQHc" tracker='custom_tracker.yaml'
        ```

For a comprehensive list of tracking arguments, refer to the [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) page.