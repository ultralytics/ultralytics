# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
from typing import Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
import yaml

from ultralytics.utils import ASSETS

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter


class YOLOv8TFLite:
    """
    YOLOv8TFLite.

    A class for performing object detection using the YOLOv8 model with TensorFlow Lite.

    Attributes:
        model (str): Path to the TensorFlow Lite model file.
        conf (float): Confidence threshold for filtering detections.
        iou (float): Intersection over Union threshold for non-maximum suppression.
        metadata (Optional[str]): Path to the metadata file, if any.

    Methods:
        detect(img_path: str) -> np.ndarray:
            Performs inference and returns the output image with drawn detections.
    """

    def __init__(self, model: str, conf: float = 0.25, iou: float = 0.45, metadata: Union[str, None] = None):
        """
        Initializes an instance of the YOLOv8TFLite class.

        Args:
            model (str): Path to the TFLite model.
            conf (float, optional): Confidence threshold for filtering detections. Defaults to 0.25.
            iou (float, optional): IoU (Intersection over Union) threshold for non-maximum suppression. Defaults to 0.45.
            metadata (Union[str, None], optional): Path to the metadata file or None if not used. Defaults to None.
        """
        self.conf = conf
        self.iou = iou
        if metadata is None:
            self.classes = {i: i for i in range(1000)}
        else:
            with open(metadata) as f:
                self.classes = yaml.safe_load(f)["names"]
        np.random.seed(42)
        self.color_palette = np.random.uniform(128, 255, size=(len(self.classes), 3))

        self.model = Interpreter(model_path=model)
        self.model.allocate_tensors()

        input_details = self.model.get_input_details()[0]

        self.in_width, self.in_height = input_details["shape"][1:3]
        self.in_index = input_details["index"]
        self.in_scale, self.in_zero_point = input_details["quantization"]
        self.int8 = input_details["dtype"] == np.int8

        output_details = self.model.get_output_details()[0]
        self.out_index = output_details["index"]
        self.out_scale, self.out_zero_point = output_details["quantization"]

    def letterbox(self, img: np.ndarray, new_shape: Tuple = (640, 640)) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Resizes and reshapes images while maintaining aspect ratio by adding padding, suitable for YOLO models."""
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top / img.shape[0], left / img.shape[1])

    def draw_detections(self, img: np.ndarray, box: np.ndarray, score: np.float32, class_id: int) -> None:
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img (np.ndarray): The input image to draw detections on.
            box (np.ndarray): Detected bounding box in the format [x1, y1, width, height].
            score (np.float32): Corresponding detection score.
            class_id (int): Class ID for the detected object.

        Returns:
            None
        """
        x1, y1, w, h = box
        color = self.color_palette[class_id]

        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        label = f"{self.classes[class_id]}: {score:.2f}"

        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        cv2.rectangle(
            img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Preprocesses the input image before performing inference.

        Args:
            img (np.ndarray): The input image to be preprocessed.

        Returns:
            Tuple[np.ndarray, Tuple[float, float]]: A tuple containing:
                - The preprocessed image (np.ndarray).
                - A tuple of two float values representing the padding applied (top/bottom, left/right).
        """
        img, pad = self.letterbox(img, (self.in_width, self.in_height))
        img = img[..., ::-1][None]  # N,H,W,C for TFLite
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        return img / 255, pad

    def postprocess(self, img: np.ndarray, outputs: np.ndarray, pad: Tuple[float, float]) -> np.ndarray:
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            img (numpy.ndarray): The input image.
            outputs (numpy.ndarray): The output of the model.
            pad (Tuple[float, float]): Padding used by letterbox.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]
        outputs[:, :4] *= max(img.shape)

        outputs = outputs.transpose(0, 2, 1)
        outputs[..., 0] -= outputs[..., 2] / 2
        outputs[..., 1] -= outputs[..., 3] / 2

        for out in outputs:
            scores = out[:, 4:].max(-1)
            keep = scores > self.conf
            boxes = out[keep, :4]
            scores = scores[keep]
            class_ids = out[keep, 4:].argmax(-1)

            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou).flatten()

            [self.draw_detections(img, boxes[i], scores[i], class_ids[i]) for i in indices]

        return img

    def detect(self, img_path: str) -> np.ndarray:
        """
        Performs inference using a TFLite model and returns the output image with drawn detections.

        Args:
            img_path (str): The path to the input image file.

        Returns:
            np.ndarray: The output image with drawn detections.
        """
        img = cv2.imread(img_path)
        x, pad = self.preprocess(img)
        if self.int8:
            x = (x / self.in_scale + self.in_zero_point).astype(np.int8)
        self.model.set_tensor(self.in_index, x)

        self.model.invoke()

        y = self.model.get_tensor(self.out_index)

        if self.int8:
            y = (y.astype(np.float32) - self.out_zero_point) * self.out_scale

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
