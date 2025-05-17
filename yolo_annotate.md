Reference for ultralytics/data/yolo_annotator.py
Note
This file is intended to be placed at ultralytics/data/yolo_annotator.py. If you spot a problem, please help fix it by contributing a Pull Request to the Ultralytics repository at https://github.com/ultralytics/ultralytics. Thank you !
ultralytics.data.yolo_annotator.yolo_annotate
python

yolo_annotate(model_path, data_dir, conf=0.25, dry_run=False)

Description
Automatically annotate images in 'images' subfolders using a pre-trained YOLO object detection model. This function processes images in a specified base directory, detects objects using the provided YOLO model, and generates bounding box annotations in YOLO format. The annotations are saved as .txt files in a sibling 'labels' folder for each 'images' directory.
This tool streamlines dataset preparation for training YOLO models by creating bounding box annotations from an existing model's predictions. It leverages multi-threading for efficiency and includes a progress bar for user feedback.
Args  
model_path (str): Path to the pre-trained YOLO model file (e.g., 'yolov12.pt'). Required.

data_dir (str): Base directory containing subfolders with 'images' directories to be annotated. Required.

conf (float, optional): Confidence threshold for object detections. Only boxes with confidence ≥ conf are saved. Defaults to 0.25.

dry_run (bool, optional): If True, simulates the annotation process without creating files, logging actions instead. Defaults to False.

Examples
Basic Annotation  
python

from ultralytics.data.yolo_annotator import yolo_annotate
yolo_annotate(model_path="yolov12.pt", data_dir="your/dataset/directory")

Annotates all images in 'images' subfolders, creating sibling 'labels' folders with .txt files.
With Confidence Threshold  
python

yolo_annotate(model_path="yolov12.pt", data_dir="your/dataset/directory", conf=0.5)

Only annotates detections with confidence ≥ 0.5.
Dry Run Mode  
python

yolo_annotate(model_path="yolov12.pt", data_dir="your/dataset/directory", dry_run=True)

Simulates annotation, logging what would be done without writing files.
Command Line Usage  
bash

python ultralytics/data/yolo_annotator.py --model yolov12.pt --data your/dataset/directory --conf 0.5 --dry-run

Runs the script with a confidence threshold of 0.5 in dry-run mode.
Notes  
Supported Image Formats: Processes images with extensions (case-insensitive): .jpg, .jpeg, .mpo, .bmp, .png, .webp, .tiff, .tif, .jfif, .avif, .heic, .heif.

Directory Structure: Targets subfolders named 'images' containing supported image files. Creates a sibling 'labels' folder at the same level (e.g., folder1/images/ and folder1/labels/).

Annotation Format: Saves annotations in YOLO format (class_id x_center y_center width height), with coordinates normalized to image dimensions.

Multi-threading: Utilizes concurrent.futures.ThreadPoolExecutor for parallel image processing, enhancing performance for large datasets.

Progress Feedback: Displays a progress bar via tqdm for each 'images' directory.

Empty Annotations: If no objects meet the confidence threshold, no .txt file is created (logged in dry-run mode).

Example Directory Structure
Before:

your/dataset/directory/
├── folder1/
│ ├── images/
│ │ ├── image1.jpg
│ │ ├── image2.png
├── folder2/
│ ├── images/
│ │ ├── photo.tiff

After:

your/dataset/directory/
├── folder1/
│ ├── images/
│ │ ├── image1.jpg
│ │ ├── image2.png
│ ├── labels/
│ │ ├── image1.txt
│ │ ├── image2.txt
├── folder2/
│ ├── images/
│ │ ├── photo.tiff
│ ├── labels/
│ │ ├── photo.txt

Source Code
The full implementation is available in ultralytics/data/yolo_annotator.py. Below is a snippet with Google-style docstrings:
python

def yolo_annotate(model_path: str, data_dir: str, conf: float = 0.25, dry_run: bool = False) -> None:
"""Automatically annotate images in 'images' subfolders using a YOLO model.

    Args:
        model_path: Path to the pre-trained YOLO model file (e.g., 'yolov12.pt').
        data_dir: Base directory containing subfolders with 'images' directories.
        conf: Confidence threshold for object detections (default: 0.25).
        dry_run: If True, simulates annotation without writing files (default: False).

    Returns:
        None

    Raises:
        FileNotFoundError: If model_path or data_dir does not exist.
        Exception: If model loading or image processing fails.
    """
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
        if os.path.basename(root) == 'images' and any(f.lower().endswith(IMAGE_EXTENSIONS) for f in files):
            image_dirs.append(root)

    # Process each 'images' directory
    for image_dir in image_dirs:
        parent_dir = os.path.dirname(image_dir)
        label_dir = os.path.join(parent_dir, 'labels')
        if not dry_run:
            os.makedirs(label_dir, exist_ok=True)

        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
        with ThreadPoolExecutor() as executor:
            processed_counts = list(tqdm(
                executor.map(lambda f: process_image(f, image_dir, label_dir, model, conf, dry_run), image_files),
                total=len(image_files),
                desc=f"Annotating {os.path.basename(parent_dir)}/images"
            ))

Dependencies  
ultralytics: For YOLO model and logging utilities (pip install ultralytics).

tqdm: For progress bar (pip install tqdm).

PR Summary
Made with by Ultralytics Actions
Summary
A new yolo_annotate.py script has been added to automate the annotation of datasets using YOLO models, making it easier to generate YOLO-format labels from images.
Key Changes  
New Script: Introduced yolo_annotate.py for dataset annotation.

Multi-threading: Utilizes multi-threading for faster processing of images.

Confidence Threshold: Allows users to set a confidence threshold for predictions.

Dry-Run Mode: Added a simulation mode to preview annotations without saving files.

File Handling: Automatically creates labels directories alongside images directories for storing YOLO-format annotations.

Command-Line Interface: Includes CLI support with arguments like model path, data directory, confidence threshold, and dry-run mode.

Purpose & Impact  
Streamlined Annotation: Simplifies generating YOLO-format labels, saving time and effort for users.

Improved Efficiency: Multi-threading ensures faster annotation of large datasets.

Customizable Workflow: Confidence threshold and dry-run options provide flexibility and control.

User-Friendly: CLI makes it accessible for both developers and non-expert users with minimal setup.
This addition enhances the usability of YOLO models for dataset preparation, making it a valuable tool for machine learning practitioners.

I have read the CLA Document and I sign the CLA
