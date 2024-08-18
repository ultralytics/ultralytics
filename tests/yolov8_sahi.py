# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse
from pathlib import Path

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model

from ultralytics.utils.files import increment_path
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.plotting import Annotator, colors

class DetectionResults:
    def __init__(self, conf, xywh, clss):
        self.conf = conf
        self.xywh = xywh
        self.cls = clss


def run(weights="yolov8n.pt", source="test.mp4", view_img=False, save_img=False, exist_ok=False):
    """
    Run object detection on a video using YOLOv8 and SAHI.

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
    """

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    yolov8_model_path = f"models/{weights}"
    download_yolov8s_model(yolov8_model_path)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=yolov8_model_path, confidence_threshold=0.3, device="cpu"
    )

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # Output setup
    save_dir = increment_path(Path("ultralytics_results_with_sahi") / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))

    tracker = BYTETracker(args=IterableSimpleNamespace(**yaml_load(check_yaml("bytetrack.yaml"))), frame_rate=fps)

    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        annotator = Annotator(frame, 3)
        results = get_sliced_prediction(
            frame, detection_model, slice_height=512, slice_width=512, overlap_height_ratio=0.2, overlap_width_ratio=0.2
        )
        object_prediction_list = results.object_prediction_list

        confidences = [float(object_prediction.score.value) for object_prediction in object_prediction_list]
        clss = [object_prediction.category.id for object_prediction in object_prediction_list]
        bboxes = [(object_prediction.bbox.minx,
                   object_prediction.bbox.miny,
                   object_prediction.bbox.maxx - object_prediction.bbox.minx,
                   object_prediction.bbox.maxy - object_prediction.bbox.miny)
                  for object_prediction in object_prediction_list]

        tracks = tracker.update(DetectionResults(conf=np.array(confidences), xywh=bboxes, clss=np.array(clss)), frame)
        print(tracks)
        for track in tracks:
            x1, x2, y1, y2, _, score, cls_index, t_id = track

            box = x1, x2, y1, y2

            annotator.box_label(box, label="person", color=colors(int(t_id), True))

        if view_img:
            cv2.imshow(Path(source).stem, frame)
        if save_img:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="initial weights path")
    parser.add_argument("--source", type=str, required=True, help="video file path")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
