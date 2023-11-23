---
comments: true
description: Learn how to use Ultralytics YOLO for object tracking in video streams. Guides to use different trackers and customise tracker configurations.
keywords: Ultralytics, YOLO, object tracking, video streams, BoT-SORT, ByteTrack, Python guide, CLI guide
---

# Multi-Object Tracking with Ultralytics YOLO

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418637-1d6250fd-1515-4c10-a844-a32818ae6d46.png" alt="Multi-object tracking examples">

Object tracking in the realm of video analytics is a critical task that not only identifies the location and class of objects within the frame but also maintains a unique ID for each detected object as the video progresses. The applications are limitless—ranging from surveillance and security to real-time sports analytics.

## Why Choose Ultralytics YOLO for Object Tracking?

The output from Ultralytics trackers is consistent with standard object detection but has the added value of object IDs. This makes it easy to track objects in video streams and perform subsequent analytics. Here's why you should consider using Ultralytics YOLO for your object tracking needs:

- **Efficiency:** Process video streams in real-time without compromising accuracy.
- **Flexibility:** Supports multiple tracking algorithms and configurations.
- **Ease of Use:** Simple Python API and CLI options for quick integration and deployment.
- **Customizability:** Easy to use with custom trained YOLO models, allowing integration into domain-specific applications.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/hHyHmOtmEgs?si=VNZtXmm45Nb9s-N-"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Object Detection and Tracking with Ultralytics YOLOv8.
</p>

## Real-world Applications

|           Transportation           |              Retail              |         Aquaculture          |
|:----------------------------------:|:--------------------------------:|:----------------------------:|
| ![Vehicle Tracking][vehicle track] | ![People Tracking][people track] | ![Fish Tracking][fish track] |
|          Vehicle Tracking          |         People Tracking          |        Fish Tracking         |

## Features at a Glance

Ultralytics YOLO extends its object detection features to provide robust and versatile object tracking:

- **Real-Time Tracking:** Seamlessly track objects in high-frame-rate videos.
- **Multiple Tracker Support:** Choose from a variety of established tracking algorithms.
- **Customizable Tracker Configurations:** Tailor the tracking algorithm to meet specific requirements by adjusting various parameters.

## Available Trackers

Ultralytics YOLO supports the following tracking algorithms. They can be enabled by passing the relevant YAML configuration file such as `tracker=tracker_type.yaml`:

* [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - Use `botsort.yaml` to enable this tracker.
* [ByteTrack](https://github.com/ifzhang/ByteTrack) - Use `bytetrack.yaml` to enable this tracker.

The default tracker is BoT-SORT.

## Tracking

To run the tracker on video streams, use a trained Detect, Segment or Pose model such as YOLOv8n, YOLOv8n-seg and YOLOv8n-pose.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load an official or custom model
        model = YOLO('yolov8n.pt')  # Load an official Detect model
        model = YOLO('yolov8n-seg.pt')  # Load an official Segment model
        model = YOLO('yolov8n-pose.pt')  # Load an official Pose model
        model = YOLO('path/to/best.pt')  # Load a custom trained model

        # Perform tracking with the model
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # Tracking with ByteTrack tracker
        ```

    === "CLI"

        ```bash
        # Perform tracking with various models using the command line interface
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4"  # Official Detect model
        yolo track model=yolov8n-seg.pt source="https://youtu.be/LNwODJXcvt4"  # Official Segment model
        yolo track model=yolov8n-pose.pt source="https://youtu.be/LNwODJXcvt4"  # Official Pose model
        yolo track model=path/to/best.pt source="https://youtu.be/LNwODJXcvt4"  # Custom trained model

        # Track using ByteTrack tracker
        yolo track model=path/to/best.pt tracker="bytetrack.yaml"
        ```

As can be seen in the above usage, tracking is available for all Detect, Segment and Pose models run on videos or streaming sources.

## Configuration

### Tracking Arguments

Tracking configuration shares properties with Predict mode, such as `conf`, `iou`, and `show`. For further configurations, refer to the [Predict](../modes/predict.md#inference-arguments) model page.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Configure the tracking parameters and run the tracker
        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
        ```

    === "CLI"

        ```bash
        # Configure tracking parameters and run the tracker using the command line interface
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.3, iou=0.5 show
        ```

### Tracker Selection

Ultralytics also allows you to use a modified tracker configuration file. To do this, simply make a copy of a tracker config file (for example, `custom_tracker.yaml`) from [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) and modify any configurations (except the `tracker_type`) as per your needs.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the model and run the tracker with a custom configuration file
        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker='custom_tracker.yaml')
        ```

    === "CLI"

        ```bash
        # Load the model and run the tracker with a custom configuration file using the command line interface
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
        ```

For a comprehensive list of tracking arguments, refer to the [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) page.

## Python Examples

### Persisting Tracks Loop

Here is a Python script using OpenCV (`cv2`) and YOLOv8 to run object tracking on video frames. This script still assumes you have already installed the necessary packages (`opencv-python` and `ultralytics`). The `persist=True` argument tells the tracker that the current image or frame is the next in a sequence and to expect tracks from the previous image in the current image.

!!! Example "Streaming for-loop with tracking"

    ```python
    import cv2
    from ultralytics import YOLO

    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Open the video file
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

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

Visualizing object tracks over consecutive frames can provide valuable insights into the movement patterns and behavior of detected objects within a video. With Ultralytics YOLOv8, plotting these tracks is a seamless and efficient process.

In the following example, we demonstrate how to utilize YOLOv8's tracking capabilities to plot the movement of detected objects across multiple video frames. This script involves opening a video file, reading it frame by frame, and utilizing the YOLO model to identify and track various objects. By retaining the center points of the detected bounding boxes and connecting them, we can draw lines that represent the paths followed by the tracked objects.

!!! Example "Plotting tracks over multiple video frames"

    ```python
    from collections import defaultdict

    import cv2
    import numpy as np

    from ultralytics import YOLO

    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

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
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

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

Two different models are used in this example: `yolov8n.pt` and `yolov8n-seg.pt`, each tracking objects in a different video file. The video files are specified in `video_file1` and `video_file2`.

The `daemon=True` parameter in `threading.Thread` means that these threads will be closed as soon as the main program finishes. We then start the threads with `start()` and use `join()` to make the main thread wait until both tracker threads have finished.

Finally, after all threads have completed their task, the windows displaying the results are closed using `cv2.destroyAllWindows()`.

!!! Example "Streaming for-loop with tracking"

    ```python
    import threading
    import cv2
    from ultralytics import YOLO


    def run_tracker_in_thread(filename, model, file_index):
        """
        Runs a video file or webcam stream concurrently with the YOLOv8 model using threading.

        This function captures video frames from a given file or camera source and utilizes the YOLOv8 model for object
        tracking. The function runs in its own thread for concurrent processing.

        Args:
            filename (str): The path to the video file or the identifier for the webcam/external camera source.
            model (obj): The YOLOv8 model object.
            file_index (int): An index to uniquely identify the file being processed, used for display purposes.

        Note:
            Press 'q' to quit the video display window.
        """
        video = cv2.VideoCapture(filename)  # Read the video file

        while True:
            ret, frame = video.read()  # Read the video frames

            # Exit the loop if no more frames in either video
            if not ret:
                break

            # Track objects in frames if available
            results = model.track(frame, persist=True)
            res_plotted = results[0].plot()
            cv2.imshow(f"Tracking_Stream_{file_index}", res_plotted)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        # Release video sources
        video.release()


    # Load the models
    model1 = YOLO('yolov8n.pt')
    model2 = YOLO('yolov8n-seg.pt')

    # Define the video files for the trackers
    video_file1 = "path/to/video1.mp4"  # Path to video file, 0 for webcam
    video_file2 = 0  # Path to video file, 0 for webcam, 1 for external camera

    # Create the tracker threads
    tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(video_file1, model1, 1), daemon=True)
    tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, model2, 2), daemon=True)

    # Start the tracker threads
    tracker_thread1.start()
    tracker_thread2.start()

    # Wait for the tracker threads to finish
    tracker_thread1.join()
    tracker_thread2.join()

    # Clean up and close windows
    cv2.destroyAllWindows()
    ```

This example can easily be extended to handle more video files and models by creating more threads and applying the same methodology.

## Contribute New Trackers

Are you proficient in multi-object tracking and have successfully implemented or adapted a tracking algorithm with Ultralytics YOLO? We invite you to contribute to our Trackers section in [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)! Your real-world applications and solutions could be invaluable for users working on tracking tasks.

By contributing to this section, you help expand the scope of tracking solutions available within the Ultralytics YOLO framework, adding another layer of functionality and utility for the community.

To initiate your contribution, please refer to our [Contributing Guide](https://docs.ultralytics.com/help/contributing) for comprehensive instructions on submitting a Pull Request (PR) 🛠️. We are excited to see what you bring to the table!

Together, let's enhance the tracking capabilities of the Ultralytics YOLO ecosystem 🙏!

[vehicle track]: https://github.com/RizwanMunawar/ultralytics/assets/62513924/ee6e6038-383b-4f21-ac29-b2a1c7d386ab

[people track]:  https://github.com/RizwanMunawar/ultralytics/assets/62513924/93bb4ee2-77a0-4e4e-8eb6-eb8f527f0527

[fish track]:    https://github.com/RizwanMunawar/ultralytics/assets/62513924/a5146d0f-bfa8-4e0a-b7df-3c1446cd8142
