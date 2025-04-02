"""
YOLO Interactive Tracking UI (CPU & GPU Toggle).
===============================================

An educational demo showcasing real-time object detection and interactive tracking
using Ultralytics YOLO + OpenCV — with a clean UI, click-to-track functionality,
and live terminal feedback.

🎯 Features:
------------
- Real-time object detection and visual tracking
- Click on any detected object to begin tracking it
- Visual overlays: bounding boxes, scope lines, center markers, object IDs
- Live terminal output with object ID, class name, confidence, and center position
- Supports both:
  - ✅ PyTorch (.pt) models — for GPU inference (Jetson, CUDA-enabled PC)
  - ✅ NCNN (.param + .bin) models — for CPU-only environments (Raspberry Pi, ARM)

🧠 CPU vs GPU Support:
----------------------
- Set `USE_GPU = True` for GPU (requires PyTorch and CUDA)
- Set `USE_GPU = False` for CPU-only (NCNN models)
- Toggle easily in the config section at the top of the script

🧩 Requirements:
----------------
- Python 3.8 or higher
- Install core dependencies:
  ```bash
  pip install ultralytics opencv-python
  ```

🔧 For PyTorch Models:
----------------------
If using a `.pt` model (GPU-enabled):
- Visit the official PyTorch install page:
  👉 https://pytorch.org/get-started/locally/
- Follow instructions based on your system (CUDA or CPU)

📦 Model File Examples:
------------------------
Place your model in the same folder or use an absolute path.

Examples:
    - `yolov8n.pt`           → PyTorch model for GPU
    - `yolov8n_ncnn_model`   → NCNN model for CPU (consists of `.param` + `.bin`)

🎮 Controls:
------------
- 🖱️  Click on any object to track it
- 🔄 Press `c` to reset tracking
- ❌ Press `q` to quit the application

🚀 How to Run:
--------------
1. Open the script and configure these lines at the top:

USE_GPU = True  # or False
MODEL_PATH_GPU = "yolov8n.pt"
MODEL_PATH_CPU = "yolov8n_ncnn_model"

2. Then run:

python interactive_tracker.py


💡 Tip:
-------
- Use lightweight models (`yolov8n`, `yolov8s`) for smoother real-time performance
- NCNN models are best for Raspberry Pi and ARM-based devices

🧑‍💻 Author:
------------
Alireza Ghaderi
📅 March 2025
🔗 https://linkedin.com/in/alireza787b

🔒 License:
-----------
This code is for educational and demo use only. Use at your own discretion.
"""
import time
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors

# ================================
# 🔧 USER CONFIGURATION SECTION
# ================================

USE_GPU = False  # Set True if running with CUDA
MODEL_PATH_GPU = "yolo11s.pt"
MODEL_PATH_CPU = "yolo11s.pt"

SHOW_FPS = True
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.3
MAX_DETECTION = 20
TRACKER_TYPE = "bytetrack.yaml"
TRACKER_ARGS = {"persist": True, "verbose": False}

colors = Colors()

# ================================
# 🚀 MODEL INITIALIZATION
# ================================

print("🚀 Initializing model...")
if USE_GPU:
    print("Using GPU...")
    model = YOLO(MODEL_PATH_GPU)
    model.to("cuda")
else:
    print("Using CPU...")
    model = YOLO(MODEL_PATH_CPU, task="detect")

# ================================
# 🎥 VIDEO SOURCE
# ================================
cap = cv2.VideoCapture(0)

selected_object_id = None
selected_bbox = None
selected_center = None
object_colors = {}

# ================================
# 🧩 HELPER FUNCTIONS
# ================================

def get_center(x1, y1, x2, y2):
    return (x1 + x2) // 2, (y1 + y2) // 2

def extend_line_from_edge(mid_x, mid_y, direction, img_shape):
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
    x1, y1, x2, y2 = bbox
    mid_top = ((x1 + x2) // 2, y1)
    mid_bottom = ((x1 + x2) // 2, y2)
    mid_left = (x1, (y1 + y2) // 2)
    mid_right = (x2, (y1 + y2) // 2)
    cv2.line(frame, mid_top, extend_line_from_edge(*mid_top, "up", frame.shape), color, 2)
    cv2.line(frame, mid_bottom, extend_line_from_edge(*mid_bottom, "down", frame.shape), color, 2)
    cv2.line(frame, mid_left, extend_line_from_edge(*mid_left, "left", frame.shape), color, 2)
    cv2.line(frame, mid_right, extend_line_from_edge(*mid_right, "right", frame.shape), color, 2)

def draw_label_with_bg(img, text, position, color, font_scale=0.6, thickness=1):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = position
    cv2.rectangle(img, (x - 2, y - h - 4), (x + w + 2, y + 2), (0, 0, 0), -1)  # Background box
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def click_event(event, x, y, flags, param):
    global selected_object_id
    if event == cv2.EVENT_LBUTTONDOWN:
        detections = results[0].boxes.data if results[0].boxes is not None else []
        if detections is not None:
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

# ================================
# 🔄 MAIN LOOP
# ================================
fps_counter, fps_timer, fps_display = 0, time.time(), 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        max_det=MAX_DETECTION,
        tracker=TRACKER_TYPE,
        **TRACKER_ARGS
    )

    frame_overlay = frame.copy()
    detections = results[0].boxes.data if results[0].boxes is not None else []
    detected_objects = []
    tracked_info = ""

    for track in detections:
        track = track.tolist()
        if len(track) < 6:
            continue
        x1, y1, x2, y2 = map(int, track[:4])
        conf = float(track[5])
        class_id = int(track[6]) if len(track) >= 7 else int(track[5])
        track_id = int(track[4]) if len(track) == 7 else -1

        label_name = model.names[class_id]
        color = object_colors.setdefault(track_id, tuple(reversed(colors(track_id, False))))
        detected_objects.append(f"{label_name} (ID {track_id}, {conf:.2f})")

        if track_id == selected_object_id:
            selected_bbox = (x1, y1, x2, y2)
            selected_center = get_center(x1, y1, x2, y2)
            tracked_info = f"🔴 TRACKING: {label_name} (ID {track_id}) | BBox: {selected_bbox} | Center: {selected_center} | Confidence: {conf:.2f}"

    # Optional: Spotlight effect (focus on tracked object)
    if selected_bbox:
        mask = frame.copy()
        overlay = frame.copy()
        overlay[:] = (0, 0, 0)
        x1, y1, x2, y2 = selected_bbox
        mask[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
        frame = cv2.addWeighted(overlay, 0.3, mask, 0.7, 0)

    for track in detections:
        track = track.tolist()
        if len(track) < 6:
            continue
        x1, y1, x2, y2 = map(int, track[:4])
        conf = float(track[5])
        class_id = int(track[6]) if len(track) >= 7 else int(track[5])
        track_id = int(track[4]) if len(track) == 7 else -1
        color = object_colors.setdefault(track_id, tuple(reversed(colors(track_id, False))))
        label = f"{model.names[class_id]} ID {track_id} ({conf:.2f})"

        if track_id == selected_object_id:
            # Highlight selected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
            draw_tracking_scope(frame, (x1, y1, x2, y2), color)
            center = get_center(x1, y1, x2, y2)
            cv2.circle(frame, center, 6, color, -1)

            # Pulsing circle for attention
            pulse_radius = 8 + int(4 * abs(time.time() % 1 - 0.5))
            cv2.circle(frame, center, pulse_radius, color, 2)

            label = f"*ACTIVE* {label}"
        else:
            # Draw dashed box for other objects
            for i in range(x1, x2, 10):
                cv2.line(frame_overlay, (i, y1), (i + 5, y1), color, 3)
                cv2.line(frame_overlay, (i, y2), (i + 5, y2), color, 3)
            for i in range(y1, y2, 10):
                cv2.line(frame_overlay, (x1, i), (x1, i + 5), color, 3)
                cv2.line(frame_overlay, (x2, i), (x2, i + 5), color, 3)

        draw_label_with_bg(frame, label, (x1 + 5, y1 + 20), color)

    if SHOW_FPS:
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()
        cv2.putText(frame, f"FPS: {fps_display}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Blend overlays and show result
    blended_frame = cv2.addWeighted(frame_overlay, 0.5, frame, 0.5, 0)
    cv2.imshow("YOLO Tracking", blended_frame)

    # Terminal logging
    print(f"🟡 DETECTED {len(detections)} OBJECT(S): {' | '.join(detected_objects)}")
    if tracked_info:
        print(tracked_info)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        print("🟢 TRACKING RESET")
        selected_object_id = None

cap.release()
cv2.destroyAllWindows()
