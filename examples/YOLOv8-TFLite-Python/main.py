# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
from typing import Tuple, Union

import cv2
import numpy as np
import yaml

from ultralytics.utils import ASSETS

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter


class YOLOv8TFLite:
    """
    A class for performing object detection using the YOLOv8 model with TensorFlow Lite.

    This class handles model loading, preprocessing, inference, and visualization of detection results.

    Attributes:
        model (Interpreter): TensorFlow Lite interpreter for the YOLOv8 model.
        conf (float): Confidence threshold for filtering detections.
        iou (float): Intersection over Union threshold for non-maximum suppression.
        classes (Dict[int, str]): Dictionary mapping class IDs to class names.
        color_palette (np.ndarray): Random color palette for visualization with shape (num_classes, 3).
        in_width (int): Input width required by the model.
        in_height (int): Input height required by the model.
        in_index (int): Input tensor index in the model.
        in_scale (float): Input quantization scale factor.
        in_zero_point (int): Input quantization zero point.
        int8 (bool): Whether the model uses int8 quantization.
        out_index (int): Output tensor index in the model.
        out_scale (float): Output quantization scale factor.
        out_zero_point (int): Output quantization zero point.

    Methods:
        letterbox: Resizes and pads image while maintaining aspect ratio.
        draw_detections: Draws bounding boxes and labels on the input image.
        preprocess: Preprocesses the input image before inference.
        postprocess: Processes model outputs to extract and visualize detections.
        detect: Performs object detection on an input image.
    """

    def __init__(self, model: str, conf: float = 0.25, iou: float = 0.45, metadata: Union[str, None] = None):
        """
        Initialize an instance of the YOLOv8TFLite class.

        Args:
            model (str): Path to the TFLite model file.
            conf (float): Confidence threshold for filtering detections.
            iou (float): IoU threshold for non-maximum suppression.
            metadata (str | None): Path to the metadata file containing class names.
        """
        self.conf = conf
        self.iou = iou
        if metadata is None:
            self.classes = {i: i for i in range(1000)}
        else:
            with open(metadata) as f:
                self.classes = yaml.safe_load(f)["names"]
        np.random.seed(42)  # Set seed for reproducible colors
        self.color_palette = np.random.uniform(128, 255, size=(len(self.classes), 3))

        # Initialize the TFLite interpreter
        self.model = Interpreter(model_path=model)
        self.model.allocate_tensors()

        # Get input details
        input_details = self.model.get_input_details()[0]
        self.in_width, self.in_height = input_details["shape"][1:3]
        self.in_index = input_details["index"]
        self.in_scale, self.in_zero_point = input_details["quantization"]
        self.int8 = input_details["dtype"] == np.int8

        # Get output details
        output_details = self.model.get_output_details()[0]
        self.out_index = output_details["index"]
        self.out_scale, self.out_zero_point = output_details["quantization"]

    def letterbox(
        self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Resize and pad image while maintaining aspect ratio.

        Args:
            img (np.ndarray): Input image with shape (H, W, C).
            new_shape (Tuple[int, int]): Target shape (height, width).

        Returns:
            (np.ndarray): Resized and padded image.
            (Tuple[float, float]): Padding ratios (top/height, left/width) for coordinate adjustment.
        """
        shape = img.shape[:2]  # Current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # Resize if needed
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top / img.shape[0], left / img.shape[1])

    def draw_detections(self, img: np.ndarray, box: np.ndarray, score: np.float32, class_id: int) -> None:
        """
        Draw bounding boxes and labels on the input image based on the detected objects.

        Args:
            img (np.ndarray): The input image to draw detections on.
            box (np.ndarray): Detected bounding box in the format [x1, y1, width, height].
            score (np.float32): Confidence score of the detection.
            class_id (int): Class ID for the detected object.
        """
        x1, y1, w, h = box
        color = self.color_palette[class_id]

        # Draw bounding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create label with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Get text size for background rectangle
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Position label above or below box depending on space
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw label background
        cv2.rectangle(
            img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        # Draw text
        cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Preprocess the input image before performing inference.

        Args:
            img (np.ndarray): The input image to be preprocessed with shape (H, W, C).

        Returns:
            (np.ndarray): Preprocessed image ready for model input.
            (Tuple[float, float]): Padding ratios for coordinate adjustment.
        """
        img, pad = self.letterbox(img, (self.in_width, self.in_height))
        img = img[..., ::-1][None]  # BGR to RGB and add batch dimension (N, H, W, C) for TFLite
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        return img / 255, pad  # Normalize to [0, 1]

    def postprocess(self, img: np.ndarray, outputs: np.ndarray, pad: Tuple[float, float]) -> np.ndarray:
        """
        Process model outputs to extract and visualize detections.

        Args:
            img (np.ndarray): The original input image.
            outputs (np.ndarray): Raw model outputs.
            pad (Tuple[float, float]): Padding ratios from preprocessing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """
        # Adjust coordinates based on padding and scale to original image size
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]
        outputs[:, :4] *= max(img.shape)

        # Transform outputs to [x, y, w, h] format
        outputs = outputs.transpose(0, 2, 1)
        outputs[..., 0] -= outputs[..., 2] / 2  # x center to top-left x
        outputs[..., 1] -= outputs[..., 3] / 2  # y center to top-left y

        for out in outputs:
            # Get scores and apply confidence threshold
            scores = out[:, 4:].max(-1)
            keep = scores > self.conf
            boxes = out[keep, :4]
            scores = scores[keep]
            class_ids = out[keep, 4:].argmax(-1)

            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou).flatten()

            # Draw detections that survived NMS
            [self.draw_detections(img, boxes[i], scores[i], class_ids[i]) for i in indices]

        return img

    def detect(self, img_path: str) -> np.ndarray:
        """
        Perform object detection on an input image.

        Args:
            img_path (str): Path to the input image file.

        Returns:
            (np.ndarray): The output image with drawn detections.
        """
        # Load and preprocess image
        img = cv2.imread(img_path)
        x, pad = self.preprocess(img)

        # Apply quantization if model is int8
        if self.int8:
            x = (x / self.in_scale + self.in_zero_point).astype(np.int8)

        # Set input tensor and run inference
        self.model.set_tensor(self.in_index, x)
        self.model.invoke()

        # Get output and dequantize if necessary
        y = self.model.get_tensor(self.out_index)
        if self.int8:
            y = (y.astype(np.float32) - self.out_zero_point) * self.out_scale

        # Process detections and return result
        return self.postprocess(img, y, pad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n_saved_model/yolov8n_full_integer_quant.tflite",
        help="Path to TFLite model.",
    )
    parser.add_argument("--img", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--metadata", type=str, default="yolov8n_saved_model/metadata.yaml", help="Metadata yaml")
    args = parser.parse_args()

    detector = YOLOv8TFLite(args.model, args.conf, args.iou, args.metadata)
    result = detector.detect(str(ASSETS / "bus.jpg"))

    cv2.imshow("Output", result)
    cv2.waitKey(0)
