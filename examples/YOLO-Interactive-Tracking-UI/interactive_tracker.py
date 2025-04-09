# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import time

import cv2

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import colors, Annotator

USE_GPU = False  # Set True if running with CUDA
model_file = "yolo11s.pt"  # Path to model file
show_fps = True  # If True, shows current FPS in top-left corner

CONF = 0.3  # Min confidence for object detection (lower = more detections, possibly more false positives)
IOU = 0.3  # IoU threshold for NMS (higher = less overlap allowed)
MAX_DET = 20  # Maximum objects per im (increase for crowded scenes)

TRACKER = "bytetrack.yaml"  # Tracker config: 'bytetrack.yaml', 'botsort.yaml', etc.
TRACK_ARGS = {
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

    results = model.track(im, conf=CONF, iou=IOU, max_det=MAX_DET, tracker=TRACKER, **TRACK_ARGS)
    annotator = Annotator(im, line_width=3, example=model.names)


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
            annotator.box_label((x1, y1, x2, y2), label, color)
            center = get_center(x1, y1, x2, y2)
            label = f"*ACTIVE* {label}"
        else:
            # Slightly dim color for non-active boxes
            if track_id != selected_object_id:
                color = tuple(int(c * 0.5) for c in color)
            annotator.box_label((x1, y1, x2, y2), label, color)


    if show_fps:
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()
        cv2.putText(im, f"FPS: {fps_display}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Blend overlays and show result
    blended_frame = cv2.addWeighted(frame_overlay, 0.5, annotator.result(), 0.5, 0)

    cv2.imshow("YOLO Tracking", annotator.result())

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
