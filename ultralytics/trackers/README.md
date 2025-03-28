<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Multi-Object Tracking with Ultralytics YOLO

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418637-1d6250fd-1515-4c10-a844-a32818ae6d46.png" alt="Ultralytics YOLO trackers visualization">

[Object tracking](https://www.ultralytics.com/glossary/object-tracking), a key aspect of [video analytics](https://en.wikipedia.org/wiki/Video_content_analysis), involves identifying the location and class of objects within video frames and assigning a unique ID to each detected object as it moves. This capability enables a wide range of applications, from surveillance and security systems to [real-time](https://www.ultralytics.com/glossary/real-time-inference) sports analysis and autonomous vehicle navigation. Learn more about tracking on our [tracking documentation page](https://docs.ultralytics.com/modes/track/).

## 🎯 Why Choose Ultralytics YOLO for Object Tracking?

Ultralytics YOLO trackers provide output consistent with standard [object detection](https://docs.ultralytics.com/tasks/detect/) but add persistent object IDs. This simplifies the process of tracking objects in video streams and performing subsequent analyses. Here’s why Ultralytics YOLO is an excellent choice for your object tracking needs:

- **Efficiency:** Process video streams in real-time without sacrificing accuracy.
- **Flexibility:** Supports multiple robust tracking algorithms and configurations.
- **Ease of Use:** Offers straightforward [Python API](https://docs.ultralytics.com/usage/python/) and [CLI](https://docs.ultralytics.com/usage/cli/) options for rapid integration and deployment.
- **Customizability:** Easily integrates with [custom-trained YOLO models](https://docs.ultralytics.com/modes/train/), enabling deployment in specialized, domain-specific applications.

**Watch:** Object Detection and Tracking with Ultralytics YOLOv8.

[![Watch the video](https://user-images.githubusercontent.com/26833433/244171528-66a4a68d-cb85-466a-984a-34301616b7a3.png)](https://www.youtube.com/watch?v=hHyHmOtmEgs)

## ✨ Features at a Glance

Ultralytics YOLO extends its powerful object detection features to deliver robust and versatile object tracking:

- **Real-Time Tracking:** Seamlessly track objects in high-frame-rate videos.
- **Multiple Tracker Support:** Choose from a selection of established tracking algorithms.
- **Customizable Tracker Configurations:** Adapt the tracking algorithm to specific requirements by adjusting various parameters.

## 🛠️ Available Trackers

Ultralytics YOLO supports the following tracking algorithms. Enable them by passing the relevant YAML configuration file, such as `tracker=tracker_type.yaml`:

- **BoT-SORT:** Use [`botsort.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/botsort.yaml) to enable this tracker. Based on the [BoT-SORT paper](https://arxiv.org/abs/2206.14651) and its official [code implementation](https://github.com/NirAharon/BoT-SORT).
- **ByteTrack:** Use [`bytetrack.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml) to enable this tracker. Based on the [ByteTrack paper](https://arxiv.org/abs/2110.06864) and its official [code implementation](https://github.com/ifzhang/ByteTrack).

The default tracker is **BoT-SORT**.

## ⚙️ Usage

To run the tracker on video streams, use a trained Detect, Segment, or Pose model like [Ultralytics YOLO11n](https://docs.ultralytics.com/models/yolo11/), YOLO11n-seg, or YOLO11n-pose.

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

As shown above, tracking is available for all [Detect](https://docs.ultralytics.com/tasks/detect/), [Segment](https://docs.ultralytics.com/tasks/segment/), and [Pose](https://docs.ultralytics.com/tasks/pose/) models when run on videos or streaming sources.

## 🔧 Configuration

### Tracking Arguments

Tracking configuration shares properties with the Predict mode, such as `conf` (confidence threshold), `iou` ([Intersection over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) threshold), and `show` (display results). For additional configurations, refer to the [Predict mode documentation](https://docs.ultralytics.com/modes/predict/).

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

Ultralytics allows you to use a modified tracker configuration file. Create a copy of a tracker config file (e.g., `custom_tracker.yaml`) from [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) and adjust any configurations (except `tracker_type`) according to your needs.

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

For a comprehensive list of tracking arguments, consult the [Tracking Configuration files](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) in the repository.

## 🐍 Python Examples

### Persisting Tracks Loop

This Python script uses [OpenCV (`cv2`)](https://opencv.org/) and Ultralytics YOLO11 to perform object tracking on video frames. Ensure you have installed the necessary packages (`opencv-python` and `ultralytics`). The [`persist=True`](https://docs.ultralytics.com/modes/predict/#tracking) argument indicates that the current frame is the next in a sequence, allowing the tracker to maintain track continuity from the previous frame.

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

Note the use of `model.track(frame)` instead of `model(frame)`, which specifically enables object tracking. This script processes each video frame, visualizes the tracking results, and displays them. Press 'q' to exit the loop.

### Plotting Tracks Over Time

Visualizing object tracks across consecutive frames offers valuable insights into movement patterns within a video. Ultralytics YOLO11 makes plotting these tracks efficient.

The following example demonstrates how to use YOLO11's tracking capabilities to plot the movement of detected objects. The script opens a video, reads it frame by frame, and uses the YOLO model built on [PyTorch](https://pytorch.org/) to identify and track objects. By storing the center points of the detected [bounding boxes](https://www.ultralytics.com/glossary/bounding-box) and connecting them, we can draw lines representing the paths of tracked objects using [NumPy](https://numpy.org/) for numerical operations.

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
        result = model.track(frame, persist=True)[0]

        # Get the boxes and track IDs
        if result.boxes and result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()
        else:
            pass

        # Visualize the result on the frame
        annotated_frame = result.plot()

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

Multithreaded tracking allows running object tracking on multiple video streams simultaneously, which is highly beneficial for systems handling inputs from several cameras, improving efficiency through concurrent processing.

This Python script utilizes Python's [`threading`](https://docs.python.org/3/library/threading.html) module for concurrent tracker execution. Each thread manages tracking for a single video file.

The `run_tracker_in_thread` function accepts parameters like the video file path, model, and a unique window index. It contains the main tracking loop, reading frames, running the tracker, and displaying results in a dedicated window.

This example uses two models, `yolo11n.pt` and `yolo11n-seg.pt`, tracking objects in `video_file1` and `video_file2`, respectively.

Setting `daemon=True` in `threading.Thread` ensures threads exit when the main program finishes. Threads are started with `start()` and the main thread waits for their completion using `join()`.

Finally, `cv2.destroyAllWindows()` closes all OpenCV windows after the threads finish.

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

This setup can be easily scaled to handle more video streams by creating additional threads following the same pattern. Explore more applications in our [blog post on object tracking](https://www.ultralytics.com/blog/object-detection-and-tracking-with-ultralytics-yolov8).

## 🤝 Contribute New Trackers

Are you experienced in multi-object tracking and have implemented or adapted an algorithm with Ultralytics YOLO? We encourage you to contribute to our Trackers section in [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)! Your contributions can help expand the tracking solutions available within the Ultralytics [ecosystem](https://docs.ultralytics.com/).

To contribute, please review our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for instructions on submitting a [Pull Request (PR)](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) 🛠️. We look forward to your contributions!

Let's work together to enhance the tracking capabilities of Ultralytics YOLO and provide more powerful tools for the [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) and [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) community 🙏!
