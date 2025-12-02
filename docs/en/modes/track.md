---
comments: true
description: Discover efficient, flexible, and customizable multi-object tracking with Ultralytics YOLO. Learn to track real-time video streams with ease.
keywords: multi-object tracking, Ultralytics YOLO, video analytics, real-time tracking, object detection, AI, machine learning
---

# Multi-Object Tracking with Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/multi-object-tracking-examples.avif" alt="Multi-object tracking examples">

Object tracking in the realm of video analytics is a critical task that not only identifies the location and class of objects within the frame but also maintains a unique ID for each detected object as the video progresses. The applications are limitless—ranging from surveillance and security to real-time sports analytics.

## Why Choose Ultralytics YOLO for Object Tracking?

The output from Ultralytics trackers is consistent with standard [object detection](https://www.ultralytics.com/glossary/object-detection) but has the added value of object IDs. This makes it easy to track objects in video streams and perform subsequent analytics. Here's why you should consider using Ultralytics YOLO for your object tracking needs:

- **Efficiency:** Process video streams in real-time without compromising [accuracy](https://www.ultralytics.com/glossary/accuracy).
- **Flexibility:** Supports multiple tracking algorithms and configurations.
- **Ease of Use:** Simple Python API and CLI options for quick integration and deployment.
- **Customizability:** Easy to use with custom trained YOLO models, allowing integration into domain-specific applications.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/hHyHmOtmEgs?si=VNZtXmm45Nb9s-N-"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Object Detection and Tracking with Ultralytics YOLO.
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

- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - Use `botsort.yaml` to enable this tracker.
- [ByteTrack](https://github.com/FoundationVision/ByteTrack) - Use `bytetrack.yaml` to enable this tracker.

The default tracker is BoT-SORT.

## Tracking

To run the tracker on video streams, use a trained Detect, Segment, or Pose model such as YOLO11n, YOLO11n-seg, or YOLO11n-pose.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load an official or custom model
        model = YOLO("yolo11n.pt")  # Load an official Detect model
        model = YOLO("yolo11n-seg.pt")  # Load an official Segment model
        model = YOLO("yolo11n-pose.pt")  # Load an official Pose model
        model = YOLO("path/to/best.pt")  # Load a custom trained model

        # Perform tracking with the model
        results = model.track("https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
        results = model.track("https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # with ByteTrack
        ```

    === "CLI"

        ```bash
        # Perform tracking with various models using the command line interface
        yolo track model=yolo11n.pt source="https://youtu.be/LNwODJXcvt4"      # Official Detect model
        yolo track model=yolo11n-seg.pt source="https://youtu.be/LNwODJXcvt4"  # Official Segment model
        yolo track model=yolo11n-pose.pt source="https://youtu.be/LNwODJXcvt4" # Official Pose model
        yolo track model=path/to/best.pt source="https://youtu.be/LNwODJXcvt4" # Custom trained model

        # Track using ByteTrack tracker
        yolo track model=path/to/best.pt tracker="bytetrack.yaml"
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
        model = YOLO("yolo11n.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
        ```

    === "CLI"

        ```bash
        # Configure tracking parameters and run the tracker using the command line interface
        yolo track model=yolo11n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.3, iou=0.5 show
        ```

### Tracker Selection

Ultralytics also allows you to use a modified tracker configuration file. To do this, simply make a copy of a tracker config file (for example, `custom_tracker.yaml`) from [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) and modify any configurations (except the `tracker_type`) as per your needs.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the model and run the tracker with a custom configuration file
        model = YOLO("yolo11n.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker="custom_tracker.yaml")
        ```

    === "CLI"

        ```bash
        # Load the model and run the tracker with a custom configuration file using the command line interface
        yolo track model=yolo11n.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
        ```

Refer to [Tracker Arguments](#tracker-arguments) section for a detailed description of each parameter.

### Tracker Arguments

Some tracking behaviors can be fine-tuned by editing the YAML configuration files specific to each tracking algorithm. These files define parameters like thresholds, buffers, and matching logic:

- [`botsort.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/botsort.yaml)
- [`bytetrack.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml)

The following table provides a description of each parameter:

!!! warning "Tracker Threshold Information"

    If a detection's confidence score falls below [`track_high_thresh`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml#L5), the tracker will not update that object, resulting in no active tracks.

| **Parameter**       | **Valid Values or Ranges**                    | **Description**                                                                                                                                        |
| ------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tracker_type`      | `botsort`, `bytetrack`                        | Specifies the tracker type. Options are `botsort` or `bytetrack`.                                                                                      |
| `track_high_thresh` | `0.0-1.0`                                     | Threshold for the first association during tracking used. Affects how confidently a detection is matched to an existing track.                         |
| `track_low_thresh`  | `0.0-1.0`                                     | Threshold for the second association during tracking. Used when the first association fails, with more lenient criteria.                               |
| `new_track_thresh`  | `0.0-1.0`                                     | Threshold to initialize a new track if the detection does not match any existing tracks. Controls when a new object is considered to appear.           |
| `track_buffer`      | `>=0`                                         | Buffer used to indicate the number of frames lost tracks should be kept alive before getting removed. Higher value means more tolerance for occlusion. |
| `match_thresh`      | `0.0-1.0`                                     | Threshold for matching tracks. Higher values makes the matching more lenient.                                                                          |
| `fuse_score`        | `True`, `False`                               | Determines whether to fuse confidence scores with IoU distances before matching. Helps balance spatial and confidence information when associating.    |
| `gmc_method`        | `orb`, `sift`, `ecc`, `sparseOptFlow`, `None` | Method used for global motion compensation. Helps account for camera movement to improve tracking.                                                     |
| `proximity_thresh`  | `0.0-1.0`                                     | Minimum IoU required for a valid match with ReID (Re-identification). Ensures spatial closeness before using appearance cues.                          |
| `appearance_thresh` | `0.0-1.0`                                     | Minimum appearance similarity required for ReID. Sets how visually similar two detections must be to be linked.                                        |
| `with_reid`         | `True`, `False`                               | Indicates whether to use ReID. Enables appearance-based matching for better tracking across occlusions. Only supported by BoTSORT.                     |
| `model`             | `auto`, `yolo11[nsmlx]-cls.pt`                | Specifies the model to use. Defaults to `auto`, which uses native features if the detector is YOLO, otherwise uses `yolo11n-cls.pt`.                   |

### Enabling Re-Identification (ReID)

By default, ReID is turned off to minimize performance overhead. Enabling it is simple—just set `with_reid: True` in the [tracker configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/botsort.yaml). You can also customize the `model` used for ReID, allowing you to trade off accuracy and speed depending on your use case:

- **Native features (`model: auto`)**: This leverages features directly from the YOLO detector for ReID, adding minimal overhead. It's ideal when you need some level of ReID without significantly impacting performance. If the detector doesn't support native features, it automatically falls back to using `yolo11n-cls.pt`.
- **YOLO classification models**: You can explicitly set a classification model (e.g. `yolo11n-cls.pt`) for ReID feature extraction. This provides more discriminative embeddings, but introduces additional latency due to the extra inference step.

For better performance, especially when using a separate classification model for ReID, you can export it to a faster backend like TensorRT:

!!! example "Exporting a ReID model to TensorRT"

    ```python
    from torch import nn

    from ultralytics import YOLO

    # Load the classification model
    model = YOLO("yolo11n-cls.pt")

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

Here is a Python script using [OpenCV](https://www.ultralytics.com/glossary/opencv) (`cv2`) and YOLO11 to run object tracking on video frames. This script assumes the necessary packages (`opencv-python` and `ultralytics`) are already installed. The `persist=True` argument tells the tracker that the current image or frame is the next in a sequence and to expect tracks from the previous image in the current image.

!!! example "Streaming for-loop with tracking"

    ```python
    import cv2

    from ultralytics import YOLO

    # Load the YOLO11 model
    model = YOLO("yolo11n.pt")

    # Open the video file
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO11 Tracking", annotated_frame)

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

Visualizing object tracks over consecutive frames can provide valuable insights into the movement patterns and behavior of detected objects within a video. With Ultralytics YOLO11, plotting these tracks is a seamless and efficient process.

In the following example, we demonstrate how to utilize YOLO11's tracking capabilities to plot the movement of detected objects across multiple video frames. This script involves opening a video file, reading it frame by frame, and utilizing the YOLO model to identify and track various objects. By retaining the center points of the detected bounding boxes and connecting them, we can draw lines that represent the paths followed by the tracked objects.

!!! example "Plotting tracks over multiple video frames"

    ```python
    from collections import defaultdict

    import cv2
    import numpy as np

    from ultralytics import YOLO

    # Load the YOLO11 model
    model = YOLO("yolo11n.pt")

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
            # Run YOLO11 tracking on the frame, persisting tracks between frames
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
            cv2.imshow("YOLO11 Tracking", frame)

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

Two different models are used in this example: `yolo11n.pt` and `yolo11n-seg.pt`, each tracking objects in a different video file. The video files are specified in `SOURCES`.

The `daemon=True` parameter in `threading.Thread` means that these threads will be closed as soon as the main program finishes. We then start the threads with `start()` and use `join()` to make the main thread wait until both tracker threads have finished.

Finally, after all threads have completed their task, the windows displaying the results are closed using `cv2.destroyAllWindows()`.

!!! example "Multithreaded tracking implementation"

    ```python
    import threading

    import cv2

    from ultralytics import YOLO

    # Define model names and video sources
    MODEL_NAMES = ["yolo11n.pt", "yolo11n-seg.pt"]
    SOURCES = ["path/to/video.mp4", "0"]  # local video, 0 for webcam


    def run_tracker_in_thread(model_name, filename):
        """Run YOLO tracker in its own thread for concurrent processing.

        Args:
            model_name (str): The YOLO11 model object.
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

[fish track]: https://github.com/ultralytics/docs/releases/download/0/fish-tracking.avif
[people track]: https://github.com/ultralytics/docs/releases/download/0/people-tracking.avif
[vehicle track]: https://github.com/ultralytics/docs/releases/download/0/vehicle-tracking.avif

## FAQ

### What is Multi-Object Tracking and how does Ultralytics YOLO support it?

Multi-object tracking in video analytics involves both identifying objects and maintaining a unique ID for each detected object across video frames. Ultralytics YOLO supports this by providing real-time tracking along with object IDs, facilitating tasks such as security surveillance and sports analytics. The system uses trackers like [BoT-SORT](https://github.com/NirAharon/BoT-SORT) and [ByteTrack](https://github.com/FoundationVision/ByteTrack), which can be configured via YAML files.

### How do I configure a custom tracker for Ultralytics YOLO?

You can configure a custom tracker by copying an existing tracker configuration file (e.g., `custom_tracker.yaml`) from the [Ultralytics tracker configuration directory](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) and modifying parameters as needed, except for the `tracker_type`. Use this file in your tracking model like so:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker="custom_tracker.yaml")
        ```

    === "CLI"

        ```bash
        yolo track model=yolo11n.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
        ```

### How can I run object tracking on multiple video streams simultaneously?

To run object tracking on multiple video streams simultaneously, you can use Python's `threading` module. Each thread will handle a separate video stream. Here's an example of how you can set this up:

!!! example "Multithreaded Tracking"

    ```python
    import threading

    import cv2

    from ultralytics import YOLO

    # Define model names and video sources
    MODEL_NAMES = ["yolo11n.pt", "yolo11n-seg.pt"]
    SOURCES = ["path/to/video.mp4", "0"]  # local video, 0 for webcam


    def run_tracker_in_thread(model_name, filename):
        """Run YOLO tracker in its own thread for concurrent processing.

        Args:
            model_name (str): The YOLO11 model object.
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

### What are the real-world applications of multi-object tracking with Ultralytics YOLO?

Multi-object tracking with Ultralytics YOLO has numerous applications, including:

- **Transportation:** Vehicle tracking for traffic management and [autonomous driving](https://www.ultralytics.com/blog/ai-in-self-driving-cars).
- **Retail:** People tracking for in-store analytics and security.
- **Aquaculture:** Fish tracking for monitoring aquatic environments.
- **Sports Analytics:** Tracking players and equipment for performance analysis.
- **Security Systems:** [Monitoring suspicious activities](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8) and creating [security alarms](https://docs.ultralytics.com/guides/security-alarm-system/).

These applications benefit from Ultralytics YOLO's ability to process high-frame-rate videos in real time with exceptional accuracy.

### How can I visualize object tracks over multiple video frames with Ultralytics YOLO?

To visualize object tracks over multiple video frames, you can use the YOLO model's tracking features along with OpenCV to draw the paths of detected objects. Here's an example script that demonstrates this:

!!! example "Plotting tracks over multiple video frames"

    ```python
    from collections import defaultdict

    import cv2
    import numpy as np

    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
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
            cv2.imshow("YOLO11 Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    ```

This script will plot the tracking lines showing the movement paths of the tracked objects over time, providing valuable insights into object behavior and patterns.
