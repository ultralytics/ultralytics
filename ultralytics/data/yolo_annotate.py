# AGPL-3.0 License
# Copyright (C) 2023 Ultralytics (https://ultralytics.com)
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License.

import os
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Supported image extensions
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".mpo",
    ".bmp",
    ".png",
    ".webp",
    ".tiff",
    ".tif",
    ".jfif",
    ".avif",
    ".heic",
    ".heif",
)


def process_image(image_file, image_dir, label_dir, model, conf, dry_run):
    """
    Process a single image and generate its annotation.

    Args:
        image_file (str): Name of the image file.
        image_dir (str): Directory containing the image (e.g., 'folder1/images').
        label_dir (str): Directory to save the labels (e.g., 'folder1/labels').
        model: Loaded YOLO model instance.
        conf (float): Confidence threshold for predictions.
        dry_run (bool): If True, simulate annotation without writing files.
    """
    if image_file.lower().endswith(IMAGE_EXTENSIONS):
        image_path = os.path.join(image_dir, image_file)
        annotation_file = os.path.join(label_dir, os.path.splitext(image_file)[0] + ".txt")

        try:
            # Run model prediction
            results = model.predict(image_path, verbose=False)
            result = results[0]

            # Get image dimensions
            width, height = result.orig_shape

            # Prepare annotation data
            annotations = []
            for box in result.boxes:
                if box.conf >= conf:  # Filter by confidence threshold
                    class_id = int(box.cls)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert tensor to list
                    # Convert to YOLO format (normalized coordinates)
                    x_center = (x1 + x2) / 2 / width
                    y_center = (y1 + y2) / 2 / height
                    box_width = (x2 - x1) / width
                    box_height = (y2 - y1) / height
                    annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

            if dry_run:
                class_names = [model.names[int(a.split()[0])] for a in annotations]
                LOGGER.info(
                    f"Would annotate {image_file} to {annotation_file} with {len(annotations)} boxes: {', '.join(class_names)}"
                )
            elif annotations:  # Only write if there are annotations
                with open(annotation_file, "w") as f:
                    f.write("\n".join(annotations) + "\n")
            return 1  # Success
        except Exception as e:
            LOGGER.error(f"Error processing {image_file} in {image_dir}: {str(e)}")
            return 0  # Failure
    return 0  # Not an image


def yolo_annotate(model_path: str, data_dir: str, conf: float = 0.25, dry_run: bool = False) -> None:
    """
    Annotate images in 'images' subfolders of a dataset using a YOLO model and save results in sibling 'labels' folders.

    Args:
        model_path (str): Path to the YOLO model file (e.g., 'yolov12.pt').
        data_dir (str): Base directory containing subfolders with 'images' directories.
        conf (float): Confidence threshold for predictions (default: 0.25).
        dry_run (bool): If True, simulate annotation without writing files (default: False).

    Example:
        >>> from ultralytics.data.yolo_annotator import yolo_annotate
        >>> yolo_annotate(model="yolov12.pt", data="your/dataset/directory", conf=0.5, dry_run=True)

    Notes:
        - Searches for subfolders named 'images' containing files with extensions: .jpg, .jpeg, .mpo, .bmp, .png,
          .webp, .tiff, .tif, .jfif, .avif, .heic, .heif.
        - Creates a 'labels' subfolder at the same level as each 'images' folder with YOLO-format annotations (.txt files).
        - Uses multi-threading for faster processing.
    """
    # Validate inputs
    if not os.path.exists(model_path):
        LOGGER.error(f"Model file {model_path} does not exist.")
        return
    if not os.path.exists(data_dir):
        LOGGER.error(f"Data directory {data_dir} does not exist.")
        return

    # Load pre-trained model
    try:
        model = YOLO(model_path)
        LOGGER.info(f"Loaded model: {model_path}")
        LOGGER.info(f"Model classes: {model.names}")
    except Exception as e:
        LOGGER.error(f"Failed to load model {model_path}: {str(e)}")
        return

    # Find all 'images' subdirectories
    image_dirs = []
    for root, dirs, files in os.walk(data_dir):
        if os.path.basename(root) == "images" and any(f.lower().endswith(IMAGE_EXTENSIONS) for f in files):
            image_dirs.append(root)

    if not image_dirs:
        LOGGER.warning(f"No 'images' subdirectories with images found in {data_dir}")
        return

    # Process each 'images' directory
    for image_dir in image_dirs:
        # 'images' folder is inside a parent directory; place 'labels' at the same level
        parent_dir = os.path.dirname(image_dir)
        label_dir = os.path.join(parent_dir, "labels")

        # Create labels directory if it doesn't exist (skip in dry-run)
        if not dry_run:
            os.makedirs(label_dir, exist_ok=True)

        LOGGER.info(f"Processing directory: {image_dir}")

        # Get list of image files
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]

        if not image_files:
            LOGGER.warning(f"No images found in {image_dir}")
            continue

        # Process images with multi-threading
        with ThreadPoolExecutor() as executor:
            processed_counts = list(
                tqdm(
                    executor.map(lambda f: process_image(f, image_dir, label_dir, model, conf, dry_run), image_files),
                    total=len(image_files),
                    desc=f"Annotating {os.path.basename(parent_dir)}/images",
                )
            )

        processed_count = sum(processed_counts)
        LOGGER.info(f"Finished processing {image_dir} - {processed_count} images processed")

    LOGGER.info("Annotation generation completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO Dataset Annotation Tool")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model file (e.g., yolov12.pt)")
    parser.add_argument(
        "--data", type=str, required=True, help="Base directory containing subfolders with 'images' directories"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions (default: 0.25)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate annotation without writing files")

    args = parser.parse_args()
    yolo_annotate(model_path=args.model, data_dir=args.data, conf=args.conf, dry_run=args.dry_run)
