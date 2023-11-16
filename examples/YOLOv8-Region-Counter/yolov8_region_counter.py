import argparse
from collections import defaultdict
from pathlib import Path

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
        'name': 'YOLOv8 Polygon Region',
        'polygon': Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),  # Polygon points
        'counts': 0,
        'dragging': False,
        'region_color': (255, 42, 4),  # BGR Value
        'text_color': (255, 255, 255)  # Region Text Color
    },
    {
        'name': 'YOLOv8 Rectangle Region',
        'polygon': Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),  # Polygon points
        'counts': 0,
        'dragging': False,
        'region_color': (37, 255, 225),  # BGR Value
        'text_color': (0, 0, 0),  # Region Text Color
    }, ]


def mouse_callback(event, x, y, flags, param):
    """Mouse call back event."""
    global current_region

    # Mouse left button down event
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region['polygon'].contains(Point((x, y))):
                current_region = region
                current_region['dragging'] = True
                current_region['offset_x'] = x
                current_region['offset_y'] = y

    # Mouse move event
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region['dragging']:
            dx = x - current_region['offset_x']
            dy = y - current_region['offset_y']
            current_region['polygon'] = Polygon([
                (p[0] + dx, p[1] + dy) for p in current_region['polygon'].exterior.coords])
            current_region['offset_x'] = x
            current_region['offset_y'] = y

    # Mouse left button up event
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region['dragging']:
            current_region['dragging'] = False


def run(
    weights='yolov8n.pt',
    source=None,
    device='cpu',
    view_img=False,
    save_img=False,
    exist_ok=False,
    line_thickness=2,
    track_thickness=2,
    region_thickness=2,
):
    """
    Run Region counting on a video using YOLOv8 and ByteTrack.

    Supports movable region for real time counting inside specific area.
    Supports multiple regions counting.
    Regions can be Polygons or rectangle in shape

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        device (str): processing device cpu, 0, 1
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
        line_thickness (int): Bounding box thickness.
        track_thickness (int): Tracking line thickness
        region_thickness (int): Region thickness.
    """
    vid_frame_count = 0

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Setup Model
    model = YOLO(f'{weights}')
    model.to('cuda') if device == '0' else model.to('cpu')

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*'mp4v')

    # Output setup
    save_dir = increment_path(Path('ultralytics_rc_output') / 'exp', exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f'{Path(source).stem}.mp4'), fourcc, fps, (frame_width, frame_height))

    # Iterate over video frames
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        # Extract the results
        results = model.track(frame, persist=True)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            names = results[0].names

            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                x, y, w, h = box
                label = str(names[cls])
                xyxy = (x - w / 2), (y - h / 2), (x + w / 2), (y + h / 2)

                # Bounding box plot
                bbox_color = colors(cls, True)
                annotator.box_label(xyxy, label, color=bbox_color)

                # Tracking Lines plot
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=bbox_color, thickness=track_thickness)

                # Check if detection inside region
                for region in counting_regions:
                    if region['polygon'].contains(Point((x, y))):
                        region['counts'] += 1

        # Draw regions (Polygons/Rectangles)
        for region in counting_regions:
            region_label = str(region['counts'])
            region_color = region['region_color']
            region_text_color = region['text_color']

            polygon_coords = np.array(region['polygon'].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region['polygon'].centroid.x), int(region['polygon'].centroid.y)

            text_size, _ = cv2.getTextSize(region_label,
                                           cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=0.7,
                                           thickness=line_thickness)
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5),
                          region_color, -1)
            cv2.putText(frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color,
                        line_thickness)
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)

        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow('Ultralytics YOLOv8 Region Counter Movable')
                cv2.setMouseCallback('Ultralytics YOLOv8 Region Counter Movable', mouse_callback)
            cv2.imshow('Ultralytics YOLOv8 Region Counter Movable', frame)

        if save_img:
            video_writer.write(frame)

        for region in counting_regions:  # Reinitialize count for each region
            region['counts'] = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    del vid_frame_count
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='initial weights path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--source', type=str, required=True, help='video file path')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-img', action='store_true', help='save results')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', type=int, default=2, help='bounding box thickness')
    parser.add_argument('--track-thickness', type=int, default=2, help='Tracking line thickness')
    parser.add_argument('--region-thickness', type=int, default=4, help='Region thickness')

    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
