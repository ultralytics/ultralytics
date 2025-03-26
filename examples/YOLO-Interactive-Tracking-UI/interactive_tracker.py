"""
YOLO Interactive Tracking UI (CPU/GPU Version).
==============================================

This is an educational, beginner-friendly demo that shows how to do real-time object detection
and interactive object tracking using Ultralytics YOLO and OpenCV.

✅ You can choose to run the model on:
   - CPU (optimized for devices like Raspberry Pi using NCNN)
   - GPU (for Jetson Nano or any CUDA-capable desktop)

🧠 Features:
------------
- Real-time object detection and tracking with visual overlays
- Click on any object to start tracking it
- Supports both PyTorch (.pt) and NCNN (.param + .bin) models
- Prints live tracking data (ID, class, confidence, center position)
- Visual scope lines and object highlights

📦 Folder Structure:
--------------------
YOLO-Interactive-Tracking-UI/
├── yolo/                    # Folder for model files
├── interactive_tracker.py   # This script
└── add_yolo_model.py        # Optional model downloader

🔧 Setup Instructions:
-----------------------
1. Install Python packages:
   ```bash
   pip install ultralytics opencv-python
   ```

2. If you want GPU acceleration (on CUDA-capable device):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. Download the right model into the `yolo/` folder:
   - For GPU: [https://github.com/ultralytics/assets/releases](https://github.com/ultralytics/assets/releases)
     - Download: `yolo11n.pt`
   - For CPU (e.g. Raspberry Pi): Use `add_yolo_model.py` to convert to NCNN

🚀 Usage:
---------
Run the script:
    python interactive_tracker.py

Toggle options inside script:
    USE_GPU = True / False

🎮 Controls:
------------
- Left click = select object to track
- Press `c` = cancel/reset tracking
- Press `q` = quit

👨‍💻 Author:
------------
Alireza Ghaderi <p30planets@gmail.com> | LinkedIn: alireza787b
March 2025

🛡️ Disclaimer:
---------------
This is for **learning, demos, and education** only — not for production.
"""

import time

import cv2
import numpy as np

from ultralytics import YOLO

# ================================
# 🔧 USER CONFIGURATION SECTION
# ================================

# 🖥️ Toggle this to True if your machine has a CUDA GPU (like Jetson Nano or desktop GPU)
USE_GPU = False

# 🧠 Select the correct model paths for GPU vs CPU
MODEL_PATH_GPU = "yolo/yolo11n.pt"  # PyTorch model (GPU)
MODEL_PATH_CPU = "yolo/yolo11n_ncnn_model"  # NCNN model (CPU)

# 🎯 Detection and tracking settings
SHOW_FPS = True
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.3
MAX_DETECTION = 20
TRACKER_TYPE = "bytetrack.yaml"  # Options: 'bytetrack.yaml', 'botsort.yaml'
TRACKER_ARGS = {
    "persist": True,
    "verbose": False,
}

# ================================
# 🚀 MODEL INITIALIZATION
# ================================

if USE_GPU:
    print("🚀 Running on GPU using PyTorch (.pt model)...")
    model = YOLO(MODEL_PATH_GPU)  # Load PyTorch model
    model.to("cuda")  # Move to GPU
else:
    print("⚙️ Running on CPU using NCNN (.param + .bin)...")
    model = YOLO(MODEL_PATH_CPU, task="detect")  # Load NCNN model for CPU inference

# ================================
# 🎥 VIDEO SOURCE (Camera or Video)
# ================================
cap = cv2.VideoCapture(0)  # Replace with path to a video file if needed

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
