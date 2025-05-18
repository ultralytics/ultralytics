# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.ultralytics import download_model_weights

from ultralytics.utils.files import increment_path


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

    def load_model(self, weights: str, device: str) -> None:
        """
        Load a YOLO11 model with specified weights for object detection using SAHI.

        Args:
            weights (str): Path to the model weights file.
            device (str, optional): CUDA device, i.e., '0' or '0,1,2,3' or 'cpu'. Defaults to "".
        """
        from ultralytics.utils.torch_utils import select_device

        yolo11_model_path = f"models/{weights}"
        download_model_weights(yolo11_model_path)  # Download model if not present
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=yolo11_model_path, device=select_device(device)
        )

    def inference(
        self,
        weights: str = "yolo11n.pt",
        source: str = "test.mp4",
        view_img: bool = False,
        save_img: bool = False,
        exist_ok: bool = False,
        device: str = "",
        hide_conf: bool = False,
        slice_width: int = 512,
        slice_height: int = 512,
    ) -> None:
        """
        Run object detection on a video using YOLO11 and SAHI.

        The function processes each frame of the video, applies sliced inference using SAHI,
        and optionally displays and/or saves the results with bounding boxes and labels.

        Args:
            weights (str): Model weights' path.
            source (str): Video file path.
            view_img (bool): Whether to display results in a window.
            save_img (bool): Whether to save results to a video file.
            exist_ok (bool): Whether to overwrite existing output files.
            device (str, optional): CUDA device, i.e., '0' or '0,1,2,3' or 'cpu'. Defaults to "".
            hide_conf (bool, optional): Flag to show or hide confidences in the output. Defaults to False.
            slice_width (int, optional): Slice width for inference.
            slice_height (int, optional): Slice height for inference.
        """
        # Video setup
        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), "Error reading video file"

        # Output setup
        save_dir = increment_path("runs/detect/predict", exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        self.load_model(weights, device)
        idx = 0  # index for image frame writing
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Perform sliced prediction using SAHI
            results = get_sliced_prediction(
                frame[..., ::-1],  # Convert BGR to RGB
                self.detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
            )

            # Display results if requested
            if view_img:
                cv2.imshow("Ultralytics YOLO Inference", frame)

            # Save results if requested
            if save_img:
                idx += 1
                results.export_visuals(export_dir=save_dir, file_name=f"img_{idx}", hide_conf=hide_conf)

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def parse_opt() -> argparse.Namespace:
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
        parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
        parser.add_argument("--hide-conf", default=False, action="store_true", help="display or hide confidences")
        parser.add_argument("--slice-width", default=512, type=int, help="Slice width for inference")
        parser.add_argument("--slice-height", default=512, type=int, help="Slice height for inference")
        return parser.parse_args()


if __name__ == "__main__":
    inference = SAHIInference()
    inference.inference(**vars(inference.parse_opt()))
