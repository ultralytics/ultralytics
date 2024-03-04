import cv2
import cvzone
import argparse
import numpy as np
from pathlib import Path
from sahi import AutoDetectionModel
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker
from deep_sort.utils.parser import get_config
from sahi.predict import get_sliced_prediction
from ultralytics.utils.files import increment_path
from sahi.utils.yolov8 import download_yolov8s_model


def run(
    weights="yolov8n.pt",
    source="test.mp4",
    without_tracking=False,
    view_img=False,
    save_img=False,
    exist_ok=False,
    tracking=False,
):
    """
    Run object detection on a video using YOLOv8 and SAHI.

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        without_tracking (bool): Disable DeepSort tracking.
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
        tracking (bool): Enable DeepSort tracking.
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
        model_type="yolov8", model_path=yolov8_model_path, confidence_threshold=0.3, device="cuda"
    )

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # Output setup
    save_dir = increment_path(Path("ultralytics_results_with_sahi") / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    if is_webcam:
        video_writer = cv2.VideoWriter(
            str(save_dir / f"{str(source)}_output.mp4"), fourcc, fps, (frame_width, frame_height)
        )
    else:
        video_writer = cv2.VideoWriter(
            str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height)
        )

    tracker = None
    if tracking:
        config = get_config()
        tracker = DeepSort(model_path="deep_sort/deep/checkpoint/ckpt.t7", max_age=70)

    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break

        results = get_sliced_prediction(
            frame, detection_model, slice_height=512, slice_width=512, overlap_height_ratio=0.2, overlap_width_ratio=0.2
        )
        object_prediction_list = results.object_prediction_list

        # DeepSort Tracking
        if tracking:
            # uncomment the following line to track only person
            # person_predictions = [obj for obj in object_prediction_list if obj.category.name == "person"]

            person_predictions = object_prediction_list

            boxes_list = []
            data = []
            for person_pred in person_predictions:
                boxes = (
                    person_pred.bbox.minx,
                    person_pred.bbox.miny,
                    person_pred.bbox.maxx,
                    person_pred.bbox.maxy,
                    person_pred.score.value,
                )
                boxes_list.append(boxes)

            if boxes_list:
                conf = np.array(boxes_list)[:, -1].reshape(-1, 1)
                boxes_np = np.array(boxes_list, dtype=np.int16)[:, :-1]
                trackers = tracker.update(boxes_np, conf, frame)
                for track in trackers:
                    bbox = track[0:4]  # Get x1, y1, x2, y2 coordinates directly
                    track_id = track[4]

                    data.append({"id": int(track_id), "bbox": np.array(bbox, dtype=np.int16)})

                    cvzone.cornerRect(
                        frame,
                        (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                        l=0,
                        rt=1,
                        t=1,
                        colorR=(0, 0, 255),
                        colorC=(0, 0, 255),
                    )
                    cvzone.putTextRect(
                        frame,
                        f"Track ID: {int(track_id)}",
                        (max(0, int(bbox[0])), max(0, int(bbox[1]))),
                        1,
                        2,
                        colorT=(255, 255, 255),
                        colorR=(0, 0, 255),
                    )

        if without_tracking:
            boxes_list = []
            clss_list = []
            for ind, _ in enumerate(object_prediction_list):
                boxes = (
                    object_prediction_list[ind].bbox.minx,
                    object_prediction_list[ind].bbox.miny,
                    object_prediction_list[ind].bbox.maxx,
                    object_prediction_list[ind].bbox.maxy,
                )
                clss = object_prediction_list[ind].category.name
                boxes_list.append(boxes)
                clss_list.append(clss)

            for box, cls in zip(boxes_list, clss_list):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 2)
                label = str(cls)
                t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
                cv2.rectangle(
                    frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), (56, 56, 255), -1
                )
                cv2.putText(
                    frame, label, (int(x1), int(y1) - 2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA
                )

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
    parser.add_argument("--without-tracking", action="store_true", help="disable DeepSort tracking")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--tracking", action="store_true", help="enable DeepSort tracking")
    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
