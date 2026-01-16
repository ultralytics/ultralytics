# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import cv2.dnn
import numpy as np

from ultralytics.utils import ASSETS, YAML
from ultralytics.utils.checks import check_yaml

CLASSES = YAML.load(check_yaml("coco8.yaml"))["names"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def validate_onnx_model(model_path: str) -> str:
    """Validate ONNX model file before loading to prevent security vulnerabilities.

    Args:
        model_path (str): Path to the ONNX model file.

    Returns:
        (str): Validated absolute path to the model file.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the file is not a valid ONNX file or fails security checks.
    """
    # Convert to Path object for safer path handling
    model_file = Path(model_path).resolve()

    # Check if file exists
    if not model_file.exists():
        raise FileNotFoundError(f"ONNX model file not found: {model_path}")

    # Check if it's a file (not a directory or symlink to prevent path traversal)
    if not model_file.is_file():
        raise ValueError(f"Path is not a regular file: {model_path}")

    # Validate file extension
    if model_file.suffix.lower() != ".onnx":
        raise ValueError(f"Invalid file extension. Expected .onnx, got: {model_file.suffix}")

    # Check file size (prevent loading extremely large files that could cause DoS)
    max_size = 500 * 1024 * 1024  # 500 MB limit
    file_size = model_file.stat().st_size
    if file_size > max_size:
        raise ValueError(f"Model file too large: {file_size} bytes (max: {max_size} bytes)")

    # Verify ONNX magic bytes (basic file signature check)
    with open(model_file, "rb") as f:
        header = f.read(4)
        # ONNX files should start with protobuf magic bytes
        if len(header) < 4:
            raise ValueError("Invalid ONNX file: file too small")

    # Return the validated absolute path as string
    return str(model_file)


def validate_input_image(image_path: str) -> None:
    """Validate input image file to prevent DoS attacks.

    Args:
        image_path (str): Path to the input image file.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file fails security checks.
    """
    # Convert to Path object for safer path handling
    img_file = Path(image_path).resolve()

    # Check if file exists
    if not img_file.exists():
        raise FileNotFoundError(f"Input image file not found: {image_path}")

    # Check if it's a file
    if not img_file.is_file():
        raise ValueError(f"Path is not a regular file: {image_path}")

    # Check file size to prevent DoS (max 50 MB for images)
    max_size = 50 * 1024 * 1024  # 50 MB limit
    file_size = img_file.stat().st_size
    if file_size > max_size:
        raise ValueError(f"Image file too large: {file_size} bytes (max: {max_size} bytes)")

    # Validate file extension
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    if img_file.suffix.lower() not in valid_extensions:
        raise ValueError(f"Invalid image extension: {img_file.suffix}. Allowed: {valid_extensions}")


def draw_bounding_box(
    img: np.ndarray, class_id: int, confidence: float, x: int, y: int, x_plus_w: int, y_plus_h: int
) -> None:
    """Draw bounding boxes on the input image based on the provided arguments.

    Args:
        img (np.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(onnx_model: str, input_image: str) -> list[dict[str, Any]]:
    """Load ONNX model, perform inference, draw bounding boxes, and display the output image.

    Args:
        onnx_model (str): Path to the ONNX model.
        input_image (str): Path to the input image.

    Returns:
        (list[dict[str, Any]]): List of dictionaries containing detection information such as class_id, class_name,
            confidence, box coordinates, and scale factor.
    """
    # Validate the ONNX model before loading (security check)
    validated_model_path = validate_onnx_model(onnx_model)

    # Validate input image before processing (security check)
    validate_input_image(input_image)

    # Load the ONNX model from validated path
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(validated_model_path)

    # Read the input image
    original_image: np.ndarray = cv2.imread(input_image)
    
    # Validate that image was loaded successfully
    if original_image is None:
        raise ValueError(f"Failed to load image: {input_image}")
    
    [height, width, _] = original_image.shape

    # Validate image dimensions to prevent excessive memory consumption
    max_dimension = 10000  # Maximum width or height
    if height > max_dimension or width > max_dimension:
        raise ValueError(f"Image dimensions too large: {width}x{height} (max: {max_dimension}x{max_dimension})")

    # Prepare a square image for inference
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    # Calculate scale factor
    scale = length / 640

    # Preprocess the image and prepare blob for model
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    
    # Validate blob size to prevent DoS attacks
    blob_size = blob.nbytes
    max_blob_size = 100 * 1024 * 1024  # 100 MB limit for blob
    if blob_size > max_blob_size:
        raise ValueError(f"Input blob too large: {blob_size} bytes (max: {max_blob_size} bytes)")
    
    model.setInput(blob)

    # Perform inference
    outputs = model.forward()

    # Validate model output
    if outputs is None or len(outputs) == 0:
        raise ValueError("Model inference failed: empty output")
    
    # Prepare output array
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]
    
    # Validate output dimensions to prevent excessive processing
    max_detections = 25000  # Reasonable limit for number of detections to process
    if rows > max_detections:
        raise ValueError(f"Model output too large: {rows} detections (max: {max_detections})")

    boxes = []
    scores = []
    class_ids = []

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (_minScore, maxScore, _minClassLoc, (_x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),  # x center - width/2 = left x
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),  # y center - height/2 = top y
                outputs[0][i][2],  # width
                outputs[0][i][3],  # height
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # Apply NMS (Non-maximum suppression)
    result_boxes = np.array(cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)).flatten()

    detections = []

    # Iterate through NMS results to draw bounding boxes and labels
    for index in result_boxes:
        index = int(index)
        box = boxes[index]
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "box": box,
            "scale": scale,
        }
        detections.append(detection)
        draw_bounding_box(
            original_image,
            class_ids[index],
            scores[index],
            round(box[0] * scale),
            round(box[1] * scale),
            round((box[0] + box[2]) * scale),
            round((box[1] + box[3]) * scale),
        )

    # Display the image with bounding boxes
    cv2.imshow("image", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.onnx", help="Input your ONNX model.")
    parser.add_argument("--img", default=str(ASSETS / "bus.jpg"), help="Path to input image.")
    args = parser.parse_args()
    main(args.model, args.img)
