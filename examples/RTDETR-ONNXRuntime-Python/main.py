# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort
import requests
import yaml


def download_file(url: str, local_path: str) -> str:
    """
    Download a file from a URL to a local path.

    Args:
        url (str): URL of the file to download.
        local_path (str): Local path where the file will be saved.
    """
    # Check if the local path already exists
    if os.path.exists(local_path):
        print(f"File already exists at {local_path}. Skipping download.")
        return local_path
    # Download the file from the URL
    print(f"Downloading {url} to {local_path}...")
    response = requests.get(url)
    with open(local_path, "wb") as f:
        f.write(response.content)

    return local_path


class RTDETR:
    """
    RT-DETR (Real-Time Detection Transformer) object detection model for ONNX inference and visualization.

    This class implements the RT-DETR model for object detection tasks, supporting ONNX model inference and
    visualization of detection results with bounding boxes and class labels.

    Attributes:
        model_path (str): Path to the ONNX model file.
        img_path (str): Path to the input image.
        conf_thres (float): Confidence threshold for filtering detections.
        iou_thres (float): IoU threshold for non-maximum suppression.
        session (ort.InferenceSession): ONNX runtime inference session.
        model_input (list): Model input metadata.
        input_width (int): Width dimension required by the model.
        input_height (int): Height dimension required by the model.
        classes (list[str]): List of class names from COCO dataset.
        color_palette (np.ndarray): Random color palette for visualization.
        img (np.ndarray): Loaded input image.
        img_height (int): Height of the input image.
        img_width (int): Width of the input image.

    Methods:
        draw_detections: Draw bounding boxes and labels on the input image.
        preprocess: Preprocess the input image for model inference.
        bbox_cxcywh_to_xyxy: Convert bounding boxes from center format to corner format.
        postprocess: Postprocess model output to extract and visualize detections.
        main: Execute the complete object detection pipeline.

    Examples:
        Initialize RT-DETR detector and run inference
        >>> detector = RTDETR("rtdetr-l.onnx", "image.jpg", conf_thres=0.5)
        >>> output_image = detector.main()
        >>> cv2.imshow("Detections", output_image)
    """

    def __init__(
        self,
        model_path: str,
        img_path: str,
        conf_thres: float = 0.5,
        iou_thres: float = 0.5,
        class_names: str | None = None,
    ):
        """
        Initialize the RT-DETR object detection model.

        Args:
            model_path (str): Path to the ONNX model file.
            img_path (str): Path to the input image.
            conf_thres (float, optional): Confidence threshold for filtering detections.
            iou_thres (float, optional): IoU threshold for non-maximum suppression.
            class_names (Optional[str], optional): Path to a YAML file containing class names.
                If None, uses COCO dataset classes.
        """
        self.model_path = model_path
        self.img_path = img_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = class_names

        # Set up the ONNX runtime session with CUDA and CPU execution providers
        self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        self.model_input = self.session.get_inputs()
        self.input_width = self.model_input[0].shape[2]
        self.input_height = self.model_input[0].shape[3]

        if self.classes is None:
            # Load class names from the COCO dataset YAML file
            self.classes = download_file(
                "https://raw.githubusercontent.com/ultralytics/"
                "ultralytics/refs/heads/main/ultralytics/cfg/datasets/coco8.yaml",
                "coco8.yaml",
            )

        # Parse the YAML file to get class names
        with open(self.classes) as f:
            class_data = yaml.safe_load(f)
            self.classes = list(class_data["names"].values())

        # Ensure the classes are a list
        if not isinstance(self.classes, list):
            raise ValueError("Classes should be a list of class names.")

        # Generate a color palette for drawing bounding boxes
        self.color_palette: np.ndarray = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, box: np.ndarray, score: float, class_id: int) -> None:
        """Draw bounding box and label on the input image for a detected object."""
        # Extract the coordinates of the bounding box
        x1, y1, x2, y2 = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(self.img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            self.img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        # Draw the label text on the image
        cv2.putText(
            self.img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    def preprocess(self) -> np.ndarray:
        """
        Preprocess the input image for model inference.

        Loads the image, converts color space from BGR to RGB, resizes to model input dimensions, and normalizes
        pixel values to [0, 1] range.

        Returns:
            (np.ndarray): Preprocessed image data with shape (1, 3, H, W) ready for inference.
        """
        # Read the input image using OpenCV
        self.img = cv2.imread(self.img_path)

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        return image_data

    def bbox_cxcywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """
        Convert bounding boxes from center format to corner format.

        Args:
            boxes (np.ndarray): Array of shape (N, 4) where each row represents a bounding box in
                (center_x, center_y, width, height) format.

        Returns:
            (np.ndarray): Array of shape (N, 4) with bounding boxes in (x_min, y_min, x_max, y_max) format.
        """
        # Calculate half width and half height of the bounding boxes
        half_width = boxes[:, 2] / 2
        half_height = boxes[:, 3] / 2

        # Calculate the coordinates of the bounding boxes
        x_min = boxes[:, 0] - half_width
        y_min = boxes[:, 1] - half_height
        x_max = boxes[:, 0] + half_width
        y_max = boxes[:, 1] + half_height

        # Return the bounding boxes in (x_min, y_min, x_max, y_max) format
        return np.column_stack((x_min, y_min, x_max, y_max))

    def postprocess(self, model_output: list[np.ndarray]) -> np.ndarray:
        """
        Postprocess model output to extract and visualize detections.

        Applies confidence thresholding, converts bounding box format, scales coordinates to original image
        dimensions, and draws detection annotations.

        Args:
            model_output (list[np.ndarray]): Output tensors from the model inference.

        Returns:
            (np.ndarray): Annotated image with detection bounding boxes and labels.
        """
        # Squeeze the model output to remove unnecessary dimensions
        outputs = np.squeeze(model_output[0])

        # Extract bounding boxes and scores from the model output
        boxes = outputs[:, :4]
        scores = outputs[:, 4:]

        # Get the class labels and scores for each detection
        labels = np.argmax(scores, axis=1)
        scores = np.max(scores, axis=1)

        # Apply confidence threshold to filter out low-confidence detections
        mask = scores > self.conf_thres
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

        # Convert bounding boxes to (x_min, y_min, x_max, y_max) format
        boxes = self.bbox_cxcywh_to_xyxy(boxes)

        # Scale bounding boxes to match the original image dimensions
        boxes[:, 0::2] *= self.img_width
        boxes[:, 1::2] *= self.img_height

        # Draw detections on the image
        for box, score, label in zip(boxes, scores, labels):
            self.draw_detections(box, score, label)

        return self.img

    def main(self) -> np.ndarray:
        """
        Execute the complete object detection pipeline on the input image.

        Performs preprocessing, ONNX model inference, and postprocessing to generate annotated detection results.

        Returns:
            (np.ndarray): Output image with detection annotations including bounding boxes and class labels.
        """
        # Preprocess the image for model input
        image_data = self.preprocess()

        # Run the model inference
        model_output = self.session.run(None, {self.model_input[0].name: image_data})

        # Process and return the model output
        return self.postprocess(model_output)


if __name__ == "__main__":
    # Set up argument parser for command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rtdetr-l.onnx", help="Path to the ONNX model file.")
    parser.add_argument("--img", type=str, default="bus.jpg", help="Path to the input image.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold for object detection.")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="IoU threshold for non-maximum suppression.")
    args = parser.parse_args()

    # Create the detector instance with specified parameters
    detection = RTDETR(args.model, args.img, args.conf_thres, args.iou_thres)

    # Perform detection and get the output image
    output_image = detection.main()

    # Display the annotated output image
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
