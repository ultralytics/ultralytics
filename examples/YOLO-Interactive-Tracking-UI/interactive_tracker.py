"""
YOLO Interactive Tracking UI
=============================

A user-friendly, educational object tracking UI using Ultralytics YOLO models and OpenCV.
This script allows users to:
- Detect and track objects in real-time
- Click on objects to initiate tracking
- Display visual overlays such as scope lines, center markers, and bounding boxes
- Output real-time tracking data in the terminal (ID, bbox, center, confidence)
- Tune parameters like confidence, IoU, tracker type, and detection limits

Model loading is handled separately via `add_yolo_model.py`, which downloads and prepares your desired model version (e.g., YOLOv8) in NCNN format. Place models inside the `yolo/` folder and configure path accordingly in this script.

Folder Structure:
-----------------
YOLO-Interactive-Tracking-UI/
├── yolo/                    # Stores NCNN models (.param, .bin)
├── add_yolo_model.py        # Script to download + convert YOLO model to NCNN
├── interactive_tracker.py   # Main tracking demo (this file)
└── README.md                # Instructions and demo info

Usage:
------
1. Run `add_yolo_model.py` to set up your desired model
2. Then run this script:
   python interactive_tracker.py

Controls:
---------
- Click on any object to select it for tracking
- Press 'c' to cancel tracking
- Press 'q' to quit the application

Dependencies:
-------------
- Python 3.8+
- ultralytics >= 8.0.0
- opencv-python

Author:
-------
Alireza Ghaderi  <p30planets@gmail.com>
📅 March 2025  
🔗 LinkedIn: https://www.linkedin.com/in/alireza787b/

License & Disclaimer:
---------------------
This project is provided for **educational and demonstration purposes** only.
The author takes **no responsibility for improper use** or deployment in production systems.
Use at your own discretion. Contributions are welcome!

"""

import time

import cv2
import numpy as np

from ultralytics import YOLO

# ========== USER-CONFIGURABLE PARAMETERS (YOLO & Tracker) ==========

SHOW_FPS = True  # If True, shows current FPS in top-left corner

CONFIDENCE_THRESHOLD = (
    0.3  # Min confidence for object detection (lower = more detections, possibly more false positives)
)
IOU_THRESHOLD = 0.3  # IoU threshold for NMS (higher = less overlap allowed)
MAX_DETECTION = 20  # Maximum objects per frame (increase for crowded scenes)

TRACKER_TYPE = "bytetrack.yaml"  # Tracker config: 'bytetrack.yaml', 'botsort.yaml', etc.
TRACKER_ARGS = {
    "persist": True,  # Keep object ID even if momentarily lost (useful for occlusion)
    "verbose": False,  # Print debug info from tracker
}

# =================== INITIALIZATION ===================
# Explicitly set task to avoid warnings; use task='detect'
model = YOLO("yolo/yolov8s_ncnn_model", task="detect")
cap = cv2.VideoCapture(0)  # Replace with video path if needed

selected_object_id = None
selected_bbox = None
selected_center = None
object_colors = {}


# ========= YOLO-LIKE COLOR GENERATOR =========
def get_yolo_color(index):
    """Generate vivid, consistent YOLO-style color for high visibility."""
    hue = (index * 0.61803398875) % 1.0  # Golden ratio
    hsv = np.array([[[int(hue * 179), 255, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ========= UTILITY FUNCTIONS =========
def get_center(x1, y1, x2, y2):
    """Get the center point of a bounding box."""
    return (x1 + x2) // 2, (y1 + y2) // 2


def extend_line_from_edge(mid_x, mid_y, direction, img_shape):
    """Extend a line from a bbox edge midpoint to screen border."""
    h, w = img_shape[:2]
    if direction == "left":
        return (0, mid_y)
    if direction == "right":
        return (w - 1, mid_y)
    if direction == "up":
        return (mid_x, 0)
    if direction == "down":
        return (mid_x, h - 1)
    return mid_x, mid_y


def draw_tracking_scope(frame, bbox, color):
    """Draw 'scope lines' from bbox edge midpoints outward."""
    x1, y1, x2, y2 = bbox
    mid_top = ((x1 + x2) // 2, y1)
    mid_bottom = ((x1 + x2) // 2, y2)
    mid_left = (x1, (y1 + y2) // 2)
    mid_right = (x2, (y1 + y2) // 2)

    cv2.line(frame, mid_top, extend_line_from_edge(*mid_top, "up", frame.shape), color, 2)
    cv2.line(frame, mid_bottom, extend_line_from_edge(*mid_bottom, "down", frame.shape), color, 2)
    cv2.line(frame, mid_left, extend_line_from_edge(*mid_left, "left", frame.shape), color, 2)
    cv2.line(frame, mid_right, extend_line_from_edge(*mid_right, "right", frame.shape), color, 2)


def click_event(event, x, y, flags, param):
    """Allow user to select an object by clicking on it."""
    global selected_object_id
    if event == cv2.EVENT_LBUTTONDOWN:
        detections = results[0].boxes.data if results[0].boxes is not None else []
        if len(detections) > 0:
            min_area = float("inf")
            best_match = None
            for track in detections:
                track = track.tolist()
                if len(track) >= 6:
                    x1, y1, x2, y2 = map(int, track[:4])
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        area = (x2 - x1) * (y2 - y1)
                        if area < min_area:
                            class_id = int(track[-1])
                            track_id = int(track[4]) if len(track) == 7 else -1
                            min_area = area
                            best_match = (track_id, model.names[class_id])
            if best_match:
                selected_object_id, label = best_match
                print(f"🔵 TRACKING STARTED: {label} (ID {selected_object_id})")


cv2.namedWindow("YOLO Tracking")
cv2.setMouseCallback("YOLO Tracking", click_event)

# ========== MAIN LOOP ==========
fps_counter, fps_timer, fps_display = 0, time.time(), 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(
        frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, max_det=MAX_DETECTION, tracker=TRACKER_TYPE, **TRACKER_ARGS
    )

    frame_overlay = frame.copy()
    detections = results[0].boxes.data if results[0].boxes is not None else []

    detected_objects = []
    tracked_info = ""

    # ========== DETECTION HANDLING ==========
    for track in detections:
        track = track.tolist()
        if len(track) < 6:
            continue  # Skip incomplete data
        x1, y1, x2, y2 = map(int, track[:4])
        conf = float(track[5])
        class_id = int(track[6]) if len(track) >= 7 else int(track[5])
        track_id = int(track[4]) if len(track) == 7 else -1

        label_name = model.names[class_id]
        color = object_colors.setdefault(track_id, get_yolo_color(track_id))
        detected_objects.append(f"{label_name} (ID {track_id}, {conf:.2f})")

        if track_id == selected_object_id:
            selected_bbox = (x1, y1, x2, y2)
            selected_center = get_center(x1, y1, x2, y2)
            tracked_info = f"🔴 TRACKING: {label_name} (ID {track_id}) | BBox: {selected_bbox} | Center: {selected_center} | Confidence: {conf:.2f}"

    # ========== DRAW VISUALIZATION ==========
    for track in detections:
        track = track.tolist()
        if len(track) < 6:
            continue
        x1, y1, x2, y2 = map(int, track[:4])
        conf = float(track[5])
        class_id = int(track[6]) if len(track) >= 7 else int(track[5])
        track_id = int(track[4]) if len(track) == 7 else -1
        color = object_colors.setdefault(track_id, get_yolo_color(track_id))
        label = f"{model.names[class_id]} ID {track_id} ({conf:.2f})"

        if track_id == selected_object_id:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            draw_tracking_scope(frame, (x1, y1, x2, y2), color)
            cv2.circle(frame, get_center(x1, y1, x2, y2), 6, color, -1)
            label = f"*ACTIVE* {label}"
        else:
            # Dashed box
            for i in range(x1, x2, 10):
                cv2.line(frame_overlay, (i, y1), (i + 5, y1), color, 2)
                cv2.line(frame_overlay, (i, y2), (i + 5, y2), color, 2)
            for i in range(y1, y2, 10):
                cv2.line(frame_overlay, (x1, i), (x1, i + 5), color, 2)
                cv2.line(frame_overlay, (x2, i), (x2, i + 5), color, 2)

        cv2.putText(frame, label, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ========== FPS DISPLAY ==========
    if SHOW_FPS:
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()
        cv2.putText(frame, f"FPS: {fps_display}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # ========== SHOW FRAME ==========
    blended_frame = cv2.addWeighted(frame_overlay, 0.5, frame, 0.5, 0)
    cv2.imshow("YOLO Tracking", blended_frame)

    # ========== TERMINAL OUTPUT ==========
    print(f"🟡 DETECTED {len(detections)} OBJECT(S): {' | '.join(detected_objects)}")
    if tracked_info:
        print(tracked_info)

    # ========== KEYBOARD CONTROL ==========
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        print("🟢 TRACKING RESET")
        selected_object_id = None

cap.release()
cv2.destroyAllWindows()
