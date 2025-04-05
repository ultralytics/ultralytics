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
    """
    Runs Ultralytics YOLO11 and SAHI for object detection on video with options to view, save, and track results.

    This class integrates SAHI (Slicing Aided Hyper Inference) with YOLO11 models to perform efficient object detection
    on large images by slicing them into smaller pieces, running inference on each slice, and then merging the results.

    Attributes:
        detection_model (AutoDetectionModel): The loaded YOLO11 model wrapped with SAHI functionality.

    Methods:
        load_model: Loads a YOLO11 model with specified weights.
        inference: Runs object detection on a video using the loaded model.
        parse_opt: Parses command line arguments for the inference process.
    """

    def __init__(self):
        """Initializes the SAHIInference class for performing sliced inference using SAHI with YOLO11 models."""
        self.detection_model = None

    def load_model(self, weights: str) -> None:
        """
        Load a YOLO11 model with specified weights for object detection using SAHI.

        Args:
            weights (str): Path to the model weights file.
        """
        yolo11_model_path = f"models/{weights}"
        download_yolo11n_model(yolo11_model_path)  # Download model if not present
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=yolo11_model_path, device="cpu"
        )

    def inference(
        self,
        weights: str = "yolo11n.pt",
        source: str = "test.mp4",
        view_img: bool = False,
        save_img: bool = False,
        exist_ok: bool = False,
    ) -> None:
        """
        Run object detection on a video using YOLO11 and SAHI.

        The function processes each frame of the video, applies sliced inference using SAHI,
        and optionally displays and/or saves the results with bounding boxes and labels.

        Args:
            weights (str): Model weights path.
            source (str): Video file path.
            view_img (bool): Whether to display results in a window.
            save_img (bool): Whether to save results to a video file.
            exist_ok (bool): Whether to overwrite existing output files.
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

            annotator = Annotator(frame)  # Initialize annotator for plotting detection results

            # Perform sliced prediction using SAHI
            results = get_sliced_prediction(
                frame[..., ::-1],  # Convert BGR to RGB
                self.detection_model,
                slice_height=512,
                slice_width=512,
            )

            # Extract detection data from results
            detection_data = [
                (det.category.name, det.category.id, (det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy))
                for det in results.object_prediction_list
            ]

            # Annotate frame with detection results
            for det in detection_data:
                annotator.box_label(det[2], label=str(det[0]), color=colors(int(det[1]), True))

            # Display results if requested
            if view_img:
                cv2.imshow(Path(source).stem, frame)

            # Save results if requested
            if save_img:
                video_writer.write(frame)

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Clean up resources
        video_writer.release()
        cap.release()
        cv2.destroyAllWindows()

    def parse_opt(self) -> argparse.Namespace:
        """
        Parse command line arguments for the inference process.

        Returns:
            (argparse.Namespace): Parsed command line arguments.
        """
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
