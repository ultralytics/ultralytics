# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
from pathlib import Path

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.ultralytics import download_yolo11n_model

from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors


class SAHIInference:
    """Runs Ultralytics YOLO11 and SAHI for object detection on video with options to view, save, and track results."""

    def __init__(self):
        """Initializes the SAHIInference class for performing sliced inference using SAHI with YOLO11 models."""
        self.detection_model = None

    def load_model(self, weights):
        """Loads a YOLO11 model with specified weights for object detection using SAHI."""
        yolo11_model_path = f"models/{weights}"
        download_yolo11n_model(yolo11_model_path)
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=yolo11_model_path, device="cpu"
        )

    def inference(
        self,
        weights="yolo11n.pt",
        source="test.mp4",
        view_img=False,
        save_img=False,
        exist_ok=False,
    ):
        """
        Run object detection on a video using YOLO11 and SAHI.

        Args:
            weights (str): Model weights path.
            source (str): Video file path.
            view_img (bool): Show results.
            save_img (bool): Save results.
            exist_ok (bool): Overwrite existing files.
        """
        # Video setup
        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), "Error reading video file"
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

        # Output setup
        save_dir = increment_path(Path("ultralytics_results_with_sahi") / "exp", exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(
            str(save_dir / f"{Path(source).stem}.avi"),
            cv2.VideoWriter_fourcc(*"MJPG"),
            int(cap.get(5)),
            (frame_width, frame_height),
        )

        # Load model
        self.load_model(weights)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            annotator = Annotator(frame)  # Initialize annotator for plotting detection and tracking results
            results = get_sliced_prediction(
                frame[..., ::-1],
                self.detection_model,
                slice_height=512,
                slice_width=512,
            )
            detection_data = [
                (det.category.name, det.category.id, (det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy))
                for det in results.object_prediction_list
            ]

            for det in detection_data:
                annotator.box_label(det[2], label=str(det[0]), color=colors(int(det[1]), True))

            if view_img:
                cv2.imshow(Path(source).stem, frame)
            if save_img:
                video_writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        video_writer.release()
        cap.release()
        cv2.destroyAllWindows()

    def parse_opt(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--weights", type=str, default="yolo11n.pt", help="initial weights path")
        parser.add_argument("--source", type=str, required=True, help="video file path")
        parser.add_argument("--view-img", action="store_true", help="show results")
        parser.add_argument("--save-img", action="store_true", help="save results")
        parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
        return parser.parse_args()


if __name__ == "__main__":
    inference = SAHIInference()
    inference.inference(**vars(inference.parse_opt()))
