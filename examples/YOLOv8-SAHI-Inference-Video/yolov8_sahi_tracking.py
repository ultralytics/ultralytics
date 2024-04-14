# Ultralytics YOLO 🚀, AGPL-3.0 license
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import download
from ultralytics.utils.files import increment_path
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.plotting import Annotator

check_requirements("easydict")
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model

from ultralytics.trackers.deep_sort import DeepSort


def run(
    weights="yolov8n.pt",
    source="test.mp4",
    view_img=False,
    save_img=False,
    exist_ok=False,
    slice_width=512,
    slice_height=512,
    slice_ratiow=0.2,
    slice_ratioh=0.2,
    conf=0.25,
    max_age=70,
    device="cpu",
    line_width=3,
):
    """
    Run object detection on a video using YOLOv8 and SAHI.

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
        slice_width (int): Width of Slice
        slice_height (int): Height of Slice
        slice_ratiow (float): Slicing Width Ratio
        slice_ratioh (float): Slicing Height Ratio
        conf (float): Confidence Threshold
        max_age (int): limits track existence without a match, ensuring accurate tracking.
        device (str): Selection of device i.e. cuda & cpu
        line_width (int): Bounding Boxes Width
    """

    # Check source path
    is_webcam = source.isdigit() and int(source) in range(10)
    if not is_webcam and not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    if is_webcam:
        source = int(source)

    yolov8_model_path = f"models/{weights}"
    download_yolov8s_model(yolov8_model_path)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=yolov8_model_path, confidence_threshold=conf, device=device
    )

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # Output setup
    save_dir = increment_path(Path("ultralytics_results_with_sahi_tracking") / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    if is_webcam:
        video_writer = cv2.VideoWriter(
            str(save_dir / f"{str(source)}_output.mp4"), fourcc, fps, (frame_width, frame_height)
        )
    else:
        video_writer = cv2.VideoWriter(
            str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height)
        )

    # Download deepsort weights
    if not os.path.exists("ckpt.t7"):
        download("https://ultralytics.com/assets/ckpt.t7")

    # Init tracker
    tracker = DeepSort(model_path="ckpt.t7", max_age=max_age)

    while videocapture.isOpened():
        success, frame = videocapture.read()
        annotator = Annotator(frame, line_width=line_width)

        if not success:
            break

        results = get_sliced_prediction(
            frame,
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=slice_ratioh,
            overlap_width_ratio=slice_ratiow,
        )

        object_prediction_list = results.object_prediction_list

        boxes_list = []
        clss_list = []
        for ind, _ in enumerate(object_prediction_list):
            boxes = (
                object_prediction_list[ind].bbox.minx,
                object_prediction_list[ind].bbox.miny,
                object_prediction_list[ind].bbox.maxx,
                object_prediction_list[ind].bbox.maxy,
                object_prediction_list[ind].score.value,
            )

            clss = object_prediction_list[ind].category.name
            boxes_list.append(boxes)
            clss_list.append(clss)

        # Plot visuals
        if boxes_list:
            conf = np.array(boxes_list)[:, -1].reshape(-1, 1)
            boxes_np = np.array(boxes_list, dtype=np.int16)[:, :-1]
            trackers = tracker.update(boxes_np, conf, frame)

            for track, box, cls in zip(trackers, boxes_list, clss_list):
                track_box = xyxy2xywh(track[0:4])
                label = cls + ":" + str(track[4])
                annotator.box_label(track_box, label=label, color=(255, 0, 255))

        # Display
        if view_img:
            if is_webcam:
                cv2.imshow("YOLOv8", frame)
            else:
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
    parser.add_argument("--slice-width", type=int, default=512, help="Slicing Width.")
    parser.add_argument("--slice-height", type=int, default=512, help="Slicing Height.")
    parser.add_argument("--slice-ratiow", type=int, default=0.2, help="Slicing Width Ratio.")
    parser.add_argument("--slice-ratioh", type=float, default=0.2, help="Slicing Height Ratio.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence Threshold Value.")
    parser.add_argument(
        "--max-age", type=int, default=70, help="limits track existence without a match, ensuring accurate tracking.."
    )
    parser.add_argument("--line-width", type=int, default=3, help="Bounding Boxes Width")

    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
