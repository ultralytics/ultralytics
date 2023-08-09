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

## Python Examples

### Persisting Tracks Loop

Here is a Python script using OpenCV (`cv2`) and YOLOv8 to run object tracking on video frames. This script still assumes you have already installed the necessary packages (`opencv-python` and `ultralytics`).

!!! example "Streaming for-loop with tracking"

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

!!! example "Plotting tracks over multiple video frames"

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

To ensure that each thread receives the correct parameters (the video file and the model to use), we define a function `run_tracker_in_thread` that accepts these parameters and contains the main tracking loop. This function reads the video frame by frame, runs the tracker, and displays the results.

Two different models are used in this example: `yolov8n.pt` and `yolov8n-seg.pt`, each tracking objects in a different video file. The video files are specified in `video_file1` and `video_file2`.

The `daemon=True` parameter in `threading.Thread` means that these threads will be closed as soon as the main program finishes. We then start the threads with `start()` and use `join()` to make the main thread wait until both tracker threads have finished.

Finally, after all threads have completed their task, the windows displaying the results are closed using `cv2.destroyAllWindows()`.

!!! example "Streaming for-loop with tracking"

    ```python
    import threading
    
    import cv2
    from ultralytics import YOLO
    
    
    def run_tracker_in_thread(filename, model):
        video = cv2.VideoCapture(filename)
        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in range(frames):
            ret, frame = video.read()
            if ret:
                results = model.track(source=frame, persist=True)
                res_plotted = results[0].plot()
                cv2.imshow('p', res_plotted)
                if cv2.waitKey(1) == ord('q'):
                    break
    
    
    # Load the models
    model1 = YOLO('yolov8n.pt')
    model2 = YOLO('yolov8n-seg.pt')
    
    # Define the video files for the trackers
    video_file1 = 'path/to/video1.mp4'
    video_file2 = 'path/to/video2.mp4'
    
    # Create the tracker threads
    tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(video_file1, model1), daemon=True)
    tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, model2), daemon=True)
    
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
