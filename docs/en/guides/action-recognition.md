---
comments: true
description: Learn how to recognize actions in real-time using Ultralytics YOLOv8 for applications like surveillance, sports analysis, and more.
keywords: action recognition, YOLOv8, Ultralytics, real-time action detection, AI, deep learning, video classification, surveillance, sports analysis
---

# Action Recognition using Ultralytics YOLOv8

## What is Action Recognition?

Action recognition involves identifying and classifying actions performed by objects (typically humans) in video streams. Using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/), you can achieve real-time action recognition for various applications such as surveillance, sports analysis, and more.

## Advantages of Action Recognition

- **Enhanced Surveillance:** Detect and classify suspicious activities in real-time, improving security measures.
- **Sports Analysis:** Analyze player movements and actions to provide insights and improve performance.
- **Behavior Monitoring:** Monitor and analyze behaviors in various settings, such as retail or healthcare.

## Real World Applications

| Surveillance | Sports Analysis |
| :----------: | :-------------: |
| ![Surveillance](https://github.com/RizwanMunawar/ultralytics/assets/62513924/surveillance.jpg) | ![Sports Analysis](https://github.com/RizwanMunawar/ultralytics/assets/62513924/sports_analysis.jpg) |
| Real-time action recognition for enhanced security | Analyzing player movements and actions |

## How to Use Action Recognition

### Installation

Ensure you have the necessary dependencies installed:

```bash
pip install ultralytics
pip install torch torchvision transformers
```

### Example Usage

Here is an example of how to use the `ActionRecognition` class for real-time action recognition:

```python
import cv2
from ultralytics import YOLO
from ultralytics.solutions.action_recognition import ActionRecognition

# Initialize the YOLO model
model = YOLO("yolov8n.pt")

# Initialize the ActionRecognition class
action_recognition = ActionRecognition(video_classifier_model="microsoft/xclip-base-patch32")

# Open a video file or capture from a camera
cap = cv2.VideoCapture("path/to/video/file.mp4")
assert cap.isOpened(), "Error reading video file"
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    # Perform object detection
    results = model(frame)
    # Perform action recognition
    tracks = results.tracks
    annotated_frame = action_recognition.recognize_actions(frame, tracks)
    # Display the frame
    cv2.imshow("Action Recognition", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
```

### Arguments

- `video_classifier_model`: Name or path of the video classifier model. Defaults to `"microsoft/xclip-base-patch32"`.
- `labels`: List of labels for zero-shot classification. Defaults to a predefined list.
- `fp16`: Whether to use half-precision floating point. Defaults to `False`.
- `crop_margin_percentage`: Percentage of margin to add around detected objects. Defaults to `10`.
- `num_video_sequence_samples`: Number of video frames to use for classification. Defaults to `8`.
- `skip_frame`: Number of frames to skip between detections. Defaults to `2`.
- `video_cls_overlap_ratio`: Overlap ratio between video sequences. Defaults to `0.25`.
- `device`: The device to run the model on. Defaults to `""`.

### Methods

- `recognize_actions(im0, tracks)`: Recognizes actions based on tracking data.
- `process_tracks(tracks)`: Extracts results from the provided tracking data and stores track information.
- `plot_box_and_action(box, pred_label, pred_conf)`: Plots track and bounding box with action label.
- `display_frames()`: Displays the current frame.
- `predict_action(sequences)`: Perform inference on the given sequences.
- `postprocess(outputs)`: Postprocess the model's batch output.

## Conclusion

Using Ultralytics YOLOv8 for action recognition provides a powerful tool for real-time applications in various domains. By following the steps outlined in this guide, you can implement action recognition in your projects and leverage the capabilities of YOLOv8 for enhanced surveillance, sports analysis, and behavior monitoring.
