# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import time

import cv2

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import colors

USE_GPU = False  # Set True if running with CUDA
model_file = "yolo11s.pt"  # Path to model file
show_fps = True  # If True, shows current FPS in top-left corner

CONFIDENCE_THRESHOLD = (
    0.3  # Min confidence for object detection (lower = more detections, possibly more false positives)
)
IOU_THRESHOLD = 0.3  # IoU threshold for NMS (higher = less overlap allowed)
MAX_DETECTION = 20  # Maximum objects per im (increase for crowded scenes)

TRACKER_TYPE = "bytetrack.yaml"  # Tracker config: 'bytetrack.yaml', 'botsort.yaml', etc.
TRACKER_ARGS = {
    "persist": True,  # Keep frames history as a stream for continuous tracking
    "verbose": False,  # Print debug info from tracker
}

LOGGER.info("🚀 Initializing model...")
if USE_GPU:
    LOGGER.info("Using GPU...")
    model = YOLO(model_file)
    model.to("cuda")
else:
    LOGGER.info("Using CPU...")
    model = YOLO(model_file, task="detect")

# VIDEO SOURCE
cap = cv2.VideoCapture(0)  # Replace with video path if needed

selected_object_id = None
selected_bbox = None
selected_center = None
object_colors = {}


def get_center(x1, y1, x2, y2):
    return (x1 + x2) // 2, (y1 + y2) // 2


def extend_line_from_edge(mid_x, mid_y, direction, img_shape):
    h, w = img_shape[:2]
    if direction == "left":
        return 0, mid_y
    if direction == "right":
        return w - 1, mid_y
    if direction == "up":
        return mid_x, 0
    if direction == "down":
        return mid_x, h - 1
    return mid_x, mid_y


def draw_tracking_scope(im, bbox, color):
    x1, y1, x2, y2 = bbox
    mid_top = ((x1 + x2) // 2, y1)
    mid_bottom = ((x1 + x2) // 2, y2)
    mid_left = (x1, (y1 + y2) // 2)
    mid_right = (x2, (y1 + y2) // 2)
    cv2.line(im, mid_top, extend_line_from_edge(*mid_top, "up", im.shape), color, 2)
    cv2.line(im, mid_bottom, extend_line_from_edge(*mid_bottom, "down", im.shape), color, 2)
    cv2.line(im, mid_left, extend_line_from_edge(*mid_left, "left", im.shape), color, 2)
    cv2.line(im, mid_right, extend_line_from_edge(*mid_right, "right", im.shape), color, 2)


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

fps_counter, fps_timer, fps_display = 0, time.time(), 0

while cap.isOpened():
    success, im = cap.read()
    if not success:
        break

    results = model.track(
        im, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, max_det=MAX_DETECTION, tracker=TRACKER_TYPE, **TRACKER_ARGS
    )

    frame_overlay = im.copy()
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
        mask = im.copy()
        overlay = im.copy()
        overlay[:] = (0, 0, 0)
        x1, y1, x2, y2 = selected_bbox
        mask[y1:y2, x1:x2] = im[y1:y2, x1:x2]
        im = cv2.addWeighted(overlay, 0.3, mask, 0.7, 0)

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
            cv2.rectangle(im, (x1, y1), (x2, y2), color, 5)
            draw_tracking_scope(im, (x1, y1, x2, y2), color)
            center = get_center(x1, y1, x2, y2)
            cv2.circle(im, center, 6, color, -1)

            # Pulsing circle for attention
            pulse_radius = 8 + int(4 * abs(time.time() % 1 - 0.5))
            cv2.circle(im, center, pulse_radius, color, 2)

            label = f"*ACTIVE* {label}"
        else:
            # Draw dashed box for other objects
            for i in range(x1, x2, 10):
                cv2.line(frame_overlay, (i, y1), (i + 5, y1), color, 3)
                cv2.line(frame_overlay, (i, y2), (i + 5, y2), color, 3)
            for i in range(y1, y2, 10):
                cv2.line(frame_overlay, (x1, i), (x1, i + 5), color, 3)
                cv2.line(frame_overlay, (x2, i), (x2, i + 5), color, 3)

        draw_label_with_bg(im, label, (x1 + 5, y1 + 20), color)

    if show_fps:
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()
        cv2.putText(im, f"FPS: {fps_display}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Blend overlays and show result
    blended_frame = cv2.addWeighted(frame_overlay, 0.5, im, 0.5, 0)
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
