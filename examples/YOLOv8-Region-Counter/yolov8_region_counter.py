import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

track_history = defaultdict(lambda: [])

from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

# Region utils
current_region = None
counting_regions = [{
    'name': 'YOLOv8 Region A',
    'roi': (50, 100, 240, 300),
    'counts': 0,
    'dragging': False,
    'region_color': (0, 255, 0)}, {
        'name': 'YOLOv8 Region B',
        'roi': (200, 250, 240, 300),
        'counts': 0,
        'dragging': False,
        'region_color': (255, 144, 31)}]


def is_inside_roi(box, roi):
    """Compare bbox with region box."""
    x, y, _, _ = box
    roi_x, roi_y, roi_w, roi_h = roi
    return roi_x < x < roi_x + roi_w and roi_y < y < roi_y + roi_h


def mouse_callback(event, x, y, flags, param):
    """Mouse call back event."""
    global current_region

    # Mouse left button down event
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            roi_x, roi_y, roi_w, roi_h = region['roi']
            if roi_x < x < roi_x + roi_w and roi_y < y < roi_y + roi_h:
                current_region = region
                current_region['dragging'] = True
                current_region['offset_x'] = x - roi_x
                current_region['offset_y'] = y - roi_y

    # Mouse move event
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region['dragging']:
            current_region['roi'] = (x - current_region['offset_x'], y - current_region['offset_y'],
                                     current_region['roi'][2], current_region['roi'][3])

    # Mouse left button up event
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region['dragging']:
            current_region['dragging'] = False


def run(weights='yolov8n.pt',
        source='test.mp4',
        view_img=False,
        save_img=False,
        exist_ok=False,
        line_thickness=2,
        region_thickness=2):
    """
    Run Region counting on a video using YOLOv8 and ByteTrack.

    Supports movable region for real time counting inside specific area.
    Supports multiple regions counting.

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
        line_thickness (int): Bounding box thickness.
        region_thickness (int): Region thickness.
    """
    vid_frame_count = 0

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Setup Model
    model = YOLO(f'{weights}')

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
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names

        annotator = Annotator(frame, line_width=line_thickness, example=str(names))

        for box, track_id, cls in zip(boxes, track_ids, clss):
            x, y, w, h = box
            label = str(names[cls])
            xyxy = (x - w / 2), (y - h / 2), (x + w / 2), (y + h / 2)

            # Bounding box
            bbox_color = colors(cls, True)
            annotator.box_label(xyxy, label, color=bbox_color)

            # Tracking Lines
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=bbox_color, thickness=line_thickness)

            # Check If detection inside region
            for region in counting_regions:
                if is_inside_roi(box, region['roi']):
                    region['counts'] += 1

        # Draw region boxes
        for region in counting_regions:
            region_label = str(region['counts'])
            roi_x, roi_y, roi_w, roi_h = region['roi']
            region_color = region['region_color']
            center_x = roi_x + roi_w // 2
            center_y = roi_y + roi_h // 2
            text_margin = 15

            # Region plotting
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), region_color, region_thickness)
            t_size, _ = cv2.getTextSize(region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=line_thickness)
            text_x = center_x - t_size[0] // 2 - text_margin
            text_y = center_y + t_size[1] // 2 + text_margin
            cv2.rectangle(frame, (text_x - text_margin, text_y - t_size[1] - text_margin),
                          (text_x + t_size[0] + text_margin, text_y + text_margin), region_color, -1)
            cv2.putText(frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), line_thickness)

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
    parser.add_argument('--source', type=str, required=True, help='video file path')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-img', action='store_true', help='save results')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', type=int, default=2, help='bounding box thickness')
    parser.add_argument('--region-thickness', type=int, default=4, help='Region thickness')
    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
