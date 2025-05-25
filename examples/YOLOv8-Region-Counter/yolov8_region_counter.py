# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

track_history = defaultdict(list)

current_region = None
counting_regions = [
    {
        "name": "Ultralytics YOLO Polygon Region",
        "polygon": Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (255, 42, 4),  # BGR Value
        "text_color": (255, 255, 255),  # Region Text Color
    },
    {
        "name": "Ultralytics YOLO Rectangle Region",
        "polygon": Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
]


def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
    """
    Handle mouse events for region manipulation in the video frame.

    This function enables interactive region selection and dragging functionality for counting regions. It responds to
    mouse button down, move, and up events to allow users to select and reposition counting regions in real-time.

    Args:
        event (int): The mouse event type (e.g., cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE).
        x (int): The x-coordinate of the mouse pointer.
        y (int): The y-coordinate of the mouse pointer.
        flags (int): Additional flags passed by OpenCV.
        param (Any): Additional parameters passed to the callback.

    Examples:
        Set up mouse callback for interactive region manipulation
        >>> cv2.setMouseCallback("window_name", mouse_callback)
    """
    global current_region

    # Mouse left button down event
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y

    # Mouse move event
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y

    # Mouse left button up event
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False


def run(
    weights: str = "yolo11n.pt",
    source: str = None,
    device: str = "cpu",
    view_img: bool = False,
    save_img: bool = False,
    exist_ok: bool = False,
    classes: List[int] = None,
    line_thickness: int = 2,
    track_thickness: int = 2,
    region_thickness: int = 2,
) -> None:
    """
    Run object detection and counting within specified regions using YOLO and ByteTrack.

    This function performs real-time object detection, tracking, and counting within user-defined polygonal or
    rectangular regions. It supports interactive region manipulation, multiple counting areas, and both live viewing
    and video saving capabilities.

    Args:
        weights (str): Path to the YOLO model weights file.
        source (str): Path to the input video file.
        device (str): Processing device specification ('cpu', '0', '1', etc.).
        view_img (bool): Display results in a live window.
        save_img (bool): Save processed video to file.
        exist_ok (bool): Overwrite existing output files without incrementing.
        classes (List[int], optional): Specific class IDs to detect and track.
        line_thickness (int): Thickness of bounding box lines.
        track_thickness (int): Thickness of object tracking lines.
        region_thickness (int): Thickness of counting region boundaries.

    Examples:
        Run region counting with default settings
        >>> run(source="video.mp4", view_img=True)

        Run with custom model and specific classes
        >>> run(weights="yolo11s.pt", source="traffic.mp4", classes=[0, 2, 3], device="0")
    """
    vid_frame_count = 0

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Setup Model
    model = YOLO(f"{weights}")
    model.to("cuda") if device == "0" else model.to("cpu")

    # Extract classes names
    names = model.names

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width = int(videocapture.get(3))
    frame_height = int(videocapture.get(4))
    fps = int(videocapture.get(5))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Output setup
    save_dir = increment_path(Path("ultralytics_rc_output") / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.avi"), fourcc, fps, (frame_width, frame_height))

    # Iterate over video frames
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        # Extract the results
        results = model.track(frame, persist=True, classes=classes)

        if results[0].boxes.is_track:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

                track = track_history[track_id]  # Tracking Lines plot
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                # Check if detection inside region
                for region in counting_regions:
                    if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        region["counts"] += 1

        # Draw regions (Polygons/Rectangles)
        for region in counting_regions:
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coordinates = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                -1,
            )
            cv2.putText(
                frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness
            )
            cv2.polylines(frame, [polygon_coordinates], isClosed=True, color=region_color, thickness=region_thickness)

        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow("Ultralytics YOLO Region Counter Movable")
                cv2.setMouseCallback("Ultralytics YOLO Region Counter Movable", mouse_callback)
            cv2.imshow("Ultralytics YOLO Region Counter Movable", frame)

        if save_img:
            video_writer.write(frame)

        for region in counting_regions:  # Reinitialize count for each region
            region["counts"] = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    del vid_frame_count
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()


def parse_opt() -> argparse.Namespace:
    """Parse command line arguments for the region counting application."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="initial weights path")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--source", type=str, required=True, help="video file path")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--line-thickness", type=int, default=2, help="bounding box thickness")
    parser.add_argument("--track-thickness", type=int, default=2, help="Tracking line thickness")
    parser.add_argument("--region-thickness", type=int, default=4, help="Region thickness")

    return parser.parse_args()


def main(options: argparse.Namespace) -> None:
    """Execute the main region counting functionality with the provided options."""
    run(**vars(options))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
