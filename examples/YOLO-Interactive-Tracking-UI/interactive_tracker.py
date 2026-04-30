# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import time

import cv2

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors

enable_gpu = False  # Set True if running with CUDA
model_file = "yolo11s.pt"  # Path to model file
show_fps = True  # If True, shows current FPS in top-left corner
show_conf = False  # Display or hide the confidence score
save_video = False  # Set True to save output video
video_output_path = "interactive_tracker_output.avi"  # Output video file name


conf = 0.3  # Min confidence for object detection (lower = more detections, possibly more false positives)
iou = 0.3  # IoU threshold for NMS (higher = less overlap allowed)
max_det = 20  # Maximum objects per image (increase for crowded scenes)

tracker = "bytetrack.yaml"  # Tracker config: 'bytetrack.yaml', 'botsort.yaml', etc.
track_args = {
    "persist": True,  # Keep frames history as a stream for continuous tracking
    "verbose": False,  # Print debug info from tracker
}

window_name = "Ultralytics YOLO Interactive Tracking"  # Output window name

LOGGER.info("🚀 Initializing model...")
if enable_gpu:
    LOGGER.info("Using GPU...")
    model = YOLO(model_file)
    model.to("cuda")
else:
    LOGGER.info("Using CPU...")
    model = YOLO(model_file, task="detect")

classes = model.names  # Store model class names

cap = cv2.VideoCapture(0)  # Replace with video path if needed
if not cap.isOpened():
    raise SystemError("Failed to open video source.")

vw = None  # Initialized lazily after the first frame is read

selected_object_id = None
selected_bbox = None
selected_center = None
latest_detections: list[list[float]] = []


def get_center(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    """Calculate the center point of a bounding box.

    Args:
        x1 (int): Top-left X coordinate.
        y1 (int): Top-left Y coordinate.
        x2 (int): Bottom-right X coordinate.
        y2 (int): Bottom-right Y coordinate.

    Returns:
        (tuple[int, int]): X and Y coordinates of the center point.
    """
    return (x1 + x2) // 2, (y1 + y2) // 2


def extend_line_from_edge(mid_x: int, mid_y: int, direction: str, img_shape: tuple[int, int, int]) -> tuple[int, int]:
    """Calculate the endpoint to extend a line from the center toward an image edge.

    Args:
        mid_x (int): X-coordinate of the midpoint.
        mid_y (int): Y-coordinate of the midpoint.
        direction (str): Direction to extend ('left', 'right', 'up', 'down').
        img_shape (tuple[int, int, int]): Image shape in (height, width, channels).

    Returns:
        (tuple[int, int]): X and Y coordinates of the endpoint.
    """
    h, w = img_shape[:2]
    if direction == "down":
        return mid_x, h - 1
    elif direction == "left":
        return 0, mid_y
    elif direction == "right":
        return w - 1, mid_y
    elif direction == "up":
        return mid_x, 0
    else:
        return mid_x, mid_y


def draw_tracking_scope(im, bbox: tuple, color: tuple) -> None:
    """Draw tracking scope lines extending from the bounding box to image edges.

    Args:
        im (np.ndarray): Image array to draw on.
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
        color (tuple): Color in BGR format for drawing.
    """
    x1, y1, x2, y2 = bbox
    mid_top = ((x1 + x2) // 2, y1)
    mid_bottom = ((x1 + x2) // 2, y2)
    mid_left = (x1, (y1 + y2) // 2)
    mid_right = (x2, (y1 + y2) // 2)
    cv2.line(im, mid_top, extend_line_from_edge(*mid_top, "up", im.shape), color, 2)
    cv2.line(im, mid_bottom, extend_line_from_edge(*mid_bottom, "down", im.shape), color, 2)
    cv2.line(im, mid_left, extend_line_from_edge(*mid_left, "left", im.shape), color, 2)
    cv2.line(im, mid_right, extend_line_from_edge(*mid_right, "right", im.shape), color, 2)


def click_event(event: int, x: int, y: int, flags: int, param) -> None:
    """Handle mouse click events to select an object for focused tracking.

    Args:
        event (int): OpenCV mouse event type.
        x (int): X-coordinate of the mouse event.
        y (int): Y-coordinate of the mouse event.
        flags (int): Any relevant flags passed by OpenCV.
        param (Any): Additional parameters (not used).
    """
    global selected_object_id, latest_detections
    if event == cv2.EVENT_LBUTTONDOWN:
        if not latest_detections:
            return
        min_area = float("inf")
        best_match = None
        for track in latest_detections:
            if len(track) < 6:
                continue
            x1, y1, x2, y2 = map(int, track[:4])
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area < min_area:
                    track_id = int(track[4]) if len(track) >= 7 else -1
                    class_id = int(track[6]) if len(track) >= 7 else int(track[5])
                    min_area = area
                    best_match = (track_id, classes.get(class_id, str(class_id)))
        if best_match:
            selected_object_id, label = best_match
            LOGGER.info(f"Tracking started: {label} (ID {selected_object_id})")


cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, click_event)

fps_counter, fps_timer, fps_display = 0, time.time(), 0

while cap.isOpened():
    success, im = cap.read()
    if not success:
        break

    results = model.track(im, conf=conf, iou=iou, max_det=max_det, tracker=tracker, **track_args)
    annotator = Annotator(im)
    detections = results[0].boxes.data if results[0].boxes is not None else []
    latest_detections = detections.cpu().tolist() if hasattr(detections, "cpu") else list(detections)  # type: ignore[arg-type]
    detected_objects: list[str] = []
    for track in detections:
        track = track.tolist()
        if len(track) < 6:
            continue
        x1, y1, x2, y2 = map(int, track[:4])
        class_id = int(track[6]) if len(track) >= 7 else int(track[5])
        track_id = int(track[4]) if len(track) == 7 else -1
        color = colors(track_id, True)
        txt_color = annotator.get_txt_color(color)
        conf_score = float(track[5]) if len(track) >= 7 else 0.0
        class_name = classes.get(class_id, str(class_id))
        label = f"{class_name} ID {track_id}" + (f" ({conf_score:.2f})" if show_conf else "")
        center = get_center(x1, y1, x2, y2)
        detected_objects.append(f"{class_name}#{track_id}@{center[0]},{center[1]}")
        if track_id == selected_object_id:
            draw_tracking_scope(im, (x1, y1, x2, y2), color)
            cv2.circle(im, center, 6, color, -1)

            # Pulsing circle for attention
            pulse_radius = 8 + int(4 * abs(time.time() % 1 - 0.5))
            cv2.circle(im, center, pulse_radius, color, 2)

            annotator.box_label([x1, y1, x2, y2], label=f"ACTIVE: TRACK {track_id}", color=color)
        else:
            # Draw dashed box for other objects
            for i in range(x1, x2, 10):
                cv2.line(im, (i, y1), (i + 5, y1), color, 3)
                cv2.line(im, (i, y2), (i + 5, y2), color, 3)
            for i in range(y1, y2, 10):
                cv2.line(im, (x1, i), (x1, i + 5), color, 3)
                cv2.line(im, (x2, i), (x2, i + 5), color, 3)
            # Draw label text with background
            (tw, th), bl = cv2.getTextSize(label, 0, 0.7, 2)
            cv2.rectangle(im, (x1 + 5 - 5, y1 + 20 - th - 5), (x1 + 5 + tw + 5, y1 + 20 + bl), color, -1)
            cv2.putText(im, label, (x1 + 5, y1 + 20), 0, 0.7, txt_color, 1, cv2.LINE_AA)

    if show_fps:
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        # Draw FPS text with background
        fps_text = f"FPS: {fps_display}"
        (tw, th), bl = cv2.getTextSize(fps_text, 0, 0.7, 2)
        cv2.rectangle(im, (10 - 5, 25 - th - 5), (10 + tw + 5, 25 + bl), (255, 255, 255), -1)
        cv2.putText(im, fps_text, (10, 25), 0, 0.7, (104, 31, 17), 1, cv2.LINE_AA)

    if save_video and vw is None:
        h, w = im.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        fps = float(fps) if fps and fps > 0 else 30.0
        ext = video_output_path.lower()
        fourcc = cv2.VideoWriter_fourcc(*("MJPG" if ext.endswith(".avi") else "mp4v"))
        vw = cv2.VideoWriter(video_output_path, fourcc, fps, (w, h))

    cv2.imshow(window_name, im)
    if save_video and vw is not None:
        vw.write(im)
    # Terminal logging
    LOGGER.info(
        f"Detected {len(detections)} object(s): {' | '.join(detected_objects)}"
        if detected_objects
        else f"Detected {len(detections)} object(s)."
    )

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        LOGGER.info("Tracking reset.")
        selected_object_id = None

cap.release()
if save_video and vw is not None:
    vw.release()
cv2.destroyAllWindows()
