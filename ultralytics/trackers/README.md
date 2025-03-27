# Multi-Object Tracking with Ultralytics YOLO

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418637-1d6250fd-1515-4c10-a844-a32818ae6d46.png" alt="Ultralytics YOLO trackers visualization">

[Object tracking](https://www.ultralytics.com/glossary/object-tracking) in the realm of [video analytics](https://www.ultralytics.com/glossary/computer-vision-cv) is a critical task that not only identifies the location and class of objects within the frame but also maintains a unique ID for each detected object as the video progresses. The applications are limitless—ranging from surveillance and security to [real-time](https://www.ultralytics.com/glossary/real-time-inference) sports analytics.

## 🎯 Why Choose Ultralytics YOLO for Object Tracking?

The output from Ultralytics trackers is consistent with standard [object detection](https://docs.ultralytics.com/tasks/detect/) but has the added value of object IDs. This makes it easy to track objects in video streams and perform subsequent analytics. Here's why you should consider using Ultralytics YOLO for your object tracking needs:

- **Efficiency:** Process video streams in real-time without compromising accuracy.
- **Flexibility:** Supports multiple tracking algorithms and configurations.
- **Ease of Use:** Simple [Python API](https://docs.ultralytics.com/usage/python/) and [CLI](https://docs.ultralytics.com/usage/cli/) options for quick integration and deployment.
- **Customizability:** Easy to use with [custom trained YOLO models](https://docs.ultralytics.com/modes/train/), allowing integration into domain-specific applications.

**Watch:** Object Detection and Tracking with Ultralytics YOLO.

[![Watch the video](https://user-images.githubusercontent.com/26833433/244171528-66a4a68d-cb85-466a-984a-34301616b7a3.png)](https://www.youtube.com/watch?v=hHyHmOtmEgs)

## ✨ Features at a Glance

Ultralytics YOLO extends its object detection features to provide robust and versatile object tracking:

- **Real-Time Tracking:** Seamlessly track objects in high-frame-rate videos.
- **Multiple Tracker Support:** Choose from a variety of established tracking algorithms.
- **Customizable Tracker Configurations:** Tailor the tracking algorithm to meet specific requirements by adjusting various parameters.

## 🛠️ Available Trackers

Ultralytics YOLO supports the following tracking algorithms. They can be enabled by passing the relevant YAML configuration file such as `tracker=tracker_type.yaml`:

- **BoT-SORT** - Use `botsort.yaml` to enable this tracker. Based on the [BoT-SORT paper](https://arxiv.org/abs/2206.14651) and [code](https://github.com/NirAharon/BoT-SORT).
- **ByteTrack** - Use `bytetrack.yaml` to enable this tracker. Based on the [ByteTrack paper](https://arxiv.org/abs/2110.06864) and [code](https://github.com/ifzhang/ByteTrack).

The default tracker is BoT-SORT.

## ⚙️ Usage

To run the tracker on video streams, use a trained Detect, Segment or Pose model such as [Ultralytics YOLO11n](https://docs.ultralytics.com/models/yolo11/), YOLO11n-seg, and YOLO11n-pose.

```python
# Python
from ultralytics import YOLO

# Load an official or custom model
model = YOLO("yolo11n.pt")  # Load an official Detect model
# model = YOLO("yolo11n-seg.pt")  # Load an official Segment model
# model = YOLO("yolo11n-pose.pt")  # Load an official Pose model
# model = YOLO("path/to/best.pt")  # Load a custom trained model

# Perform tracking with the model
results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # Tracking with ByteTrack tracker
```

```bash
# CLI
# Perform tracking with various models using the command line interface
yolo track model=yolo11n.pt source="https://youtu.be/LNwODJXcvt4" # Official Detect model
# yolo track model=yolo11n-seg.pt source="https://youtu.be/LNwODJXcvt4"  # Official Segment model
# yolo track model=yolo11n-pose.pt source="https://youtu.be/LNwODJXcvt4" # Official Pose model
# yolo track model=path/to/best.pt source="https://youtu.be/LNwODJXcvt4" # Custom trained model

# Track using ByteTrack tracker
# yolo track model=path/to/best.pt tracker="bytetrack.yaml"
```

As can be seen in the above usage, tracking is available for all [Detect](https://docs.ultralytics.com/tasks/detect/), [Segment](https://docs.ultralytics.com/tasks/segment/) and [Pose](https://docs.ultralytics.com/tasks/pose/) models run on videos or streaming sources.

## 🔧 Configuration

### Tracking Arguments

Tracking configuration shares properties with Predict mode, such as `conf`, `iou`, and `show`. For further configurations, refer to the [Predict mode](https://docs.ultralytics.com/modes/predict/) documentation.

```python
# Python
from ultralytics import YOLO

# Configure the tracking parameters and run the tracker
model = YOLO("yolo11n.pt")
results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
```

```bash
# CLI
# Configure tracking parameters and run the tracker using the command line interface
yolo track model=yolo11n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.3 iou=0.5 show
```

### Tracker Selection

Ultralytics also allows you to use a modified tracker configuration file. To do this, simply make a copy of a tracker config file (for example, `custom_tracker.yaml`) from [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) and modify any configurations (except the `tracker_type`) as per your needs.

```python
# Python
from ultralytics import YOLO

# Load the model and run the tracker with a custom configuration file
model = YOLO("yolo11n.pt")
results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker="custom_tracker.yaml")
```

```bash
# CLI
# Load the model and run the tracker with a custom configuration file using the command line interface
yolo track model=yolo11n.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
```

For a comprehensive list of tracking arguments, refer to the [Tracking Configuration](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) documentation.

## 🐍 Python Examples

### Persisting Tracks Loop

Here is a Python script using [OpenCV (`cv2`)](https://opencv.org/) and Ultralytics YOLO11 to run object tracking on video frames. This script assumes you have already installed the necessary packages (`opencv-python` and `ultralytics`). The [`persist=True`](https://docs.ultralytics.com/modes/predict/#tracking) argument tells the tracker that the current image or frame is the next in a sequence and to expect tracks from the previous image in the current image.

```python
# Python
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

```python
# Python
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
            if len(track) > 30:  # retain 30 tracks for 30 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

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

### Multithreaded Tracking

Multithreaded tracking provides the capability to run object tracking on multiple video streams simultaneously. This is particularly useful when handling multiple video inputs, such as from multiple surveillance cameras, where concurrent processing can greatly enhance efficiency and performance.

In the provided Python script, we make use of Python's [`threading`](https://docs.python.org/3/library/threading.html) module to run multiple instances of the tracker concurrently. Each thread is responsible for running the tracker on one video file.

To ensure that each thread receives the correct parameters (video file path, model, and window name), we define a function `run_tracker_in_thread` that accepts these parameters and contains the main tracking loop. This function reads the video frame by frame, runs the tracker, and displays the results in a unique window.

Two different models are used in this example: `yolo11n.pt` and `yolo11n-seg.pt`, each tracking objects in a different video file specified by `video_file1` and `video_file2`.

The `daemon=True` parameter in `threading.Thread` means that these threads will exit automatically when the main program finishes. We start the threads with `start()` and use `join()` to make the main thread wait until both tracker threads have completed.

Finally, after all threads have finished, the script closes all OpenCV windows using `cv2.destroyAllWindows()`.

```python
# Python
import threading

import cv2

from ultralytics import YOLO


def run_tracker_in_thread(filename, model, file_index):
    """
    Runs tracking on a video file using the specified model.

    Args:
        filename (str): The path to the video file.
        model (YOLO): The YOLO model to use for tracking.
        file_index (int): An index to identify the video thread.
    """
    video = cv2.VideoCapture(filename)  # Read the video file
    while True:
        ret, frame = video.read()  # Read the video frames
        if not ret:
            break  # Exit the loop if no more frames are available
        results = model.track(frame, persist=True)  # Track objects in the frame
        res_plotted = results[0].plot()  # Visualize the results
        cv2.imshow(f"Tracking_Stream_{file_index}", res_plotted)  # Display the results

        key = cv2.waitKey(1)  # Wait for a key event
        if key == ord("q"):  # Check if the 'q' key was pressed
            break  # Exit the loop if 'q' is pressed

    video.release()  # Release the video capture object


# Load the models
model1 = YOLO("yolo11n.pt")
model2 = YOLO("yolo11n-seg.pt")

# Define the video files for the trackers
video_file1 = "path/to/video1.mp4"  # Path to video file 1
video_file2 = "path/to/video2.mp4"  # Path to video file 2

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

## 🤝 Contribute New Trackers

Are you proficient in multi-object tracking and have successfully implemented or adapted a tracking algorithm with Ultralytics YOLO? We invite you to contribute to our Trackers section in [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)! Your real-world applications and solutions could be invaluable for users working on tracking tasks.

By contributing to this section, you help expand the scope of tracking solutions available within the Ultralytics YOLO framework, adding another layer of functionality and utility for the community.

To initiate your contribution, please refer to our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for comprehensive instructions on submitting a [Pull Request (PR)](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) 🛠️. We are excited to see what you bring to the table!

Together, let's enhance the tracking capabilities of the Ultralytics YOLO ecosystem 🙏!
