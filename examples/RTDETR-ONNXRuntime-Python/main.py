# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse

import cv2
import numpy as np
import onnxruntime as ort
import torch

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml


class RTDETR:
    """RTDETR object detection model class for handling inference and visualization."""

    def __init__(self, model_path, img_path, conf_thres=0.5, iou_thres=0.5):
        """
        Initializes the RTDETR object with the specified parameters.

        Args:
            model_path: Path to the ONNX model file.
            img_path: Path to the input image.
            conf_thres: Confidence threshold for object detection.
            iou_thres: IoU threshold for non-maximum suppression
        """
        self.model_path = model_path
        self.img_path = img_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Set up the ONNX runtime session with CUDA and CPU execution providers
        self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.model_input = self.session.get_inputs()
        self.input_width = self.model_input[0].shape[2]
        self.input_height = self.model_input[0].shape[3]

        # Load class names from the COCO dataset YAML file
        self.classes = yaml_load(check_yaml("coco8.yaml"))["names"]

        # Generate a color palette for drawing bounding boxes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """
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

    def preprocess(self):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
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

        # Return the preprocessed image data
        return image_data

    def bbox_cxcywh_to_xyxy(self, boxes):
        """
        Converts bounding boxes from (center x, center y, width, height) format to (x_min, y_min, x_max, y_max) format.

        Args:
            boxes (numpy.ndarray): An array of shape (N, 4) where each row represents
                                a bounding box in (cx, cy, w, h) format.

        Returns:
            numpy.ndarray: An array of shape (N, 4) where each row represents
                        a bounding box in (x_min, y_min, x_max, y_max) format.
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

    def postprocess(self, model_output):
        """
        Postprocesses the model output to extract detections and draw them on the input image.

        Args:
            model_output: Output of the model inference.

        Returns:
            np.array: Annotated image with detections.
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

        # Return the annotated image
        return self.img

    def main(self):
        """
        Executes the detection on the input image using the ONNX model.

        Returns:
            np.array: Output image with annotations.
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
    parser.add_argument("--img", type=str, default=str(ASSETS / "bus.jpg"), help="Path to the input image.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold for object detection.")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="IoU threshold for non-maximum suppression.")
    args = parser.parse_args()

    # Check for dependencies and set up ONNX runtime
    check_requirements("onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime")

    # Create the detector instance with specified parameters
    detection = RTDETR(args.model, args.img, args.conf_thres, args.iou_thres)

    # Perform detection and get the output image
    output_image = detection.main()

    # Display the annotated output image
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
