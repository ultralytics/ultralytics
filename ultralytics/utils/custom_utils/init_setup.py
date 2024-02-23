import re
from datetime import datetime

import cv2
import fiftyone as fo
from tqdm import tqdm

from ultralytics.config import ORIGINAL_CLASSES, DATA_PATH, DATASET_DESCRIPTION, DATASET_NAME


def delete_all_fiftyone_datasets():
    for dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(dataset_name)
        dataset.delete()

def export_dataset(dataset, export_dir, classes):
    dataset_type = fo.types.YOLOv5Dataset

    label_field = "segmentations"

    dataset.export(
        export_dir = export_dir,
        dataset_type = dataset_type,
        label_field = label_field,
        classes=classes
    )


def get_date_and_transect(filename):
    # Define a regex pattern to match the date and Transect in the filename
    pattern = re.compile(r"(\d{14})[^a-zA-Z]*(Transect\s*\d+)?", re.IGNORECASE)

    # Search for the pattern in the filename
    match = pattern.search(filename)

    time = None
    if match:
        date_str, transect_match = match.groups()
        
        if date_str:
            date = datetime.strptime(date_str, "%Y%m%d%H%M%S")
        else:
            date = None

        if transect_match:
            # If Transect information is present, extract it
            transect = transect_match.strip()
        else:
            transect = None
    else:
        date = None
        transect = filename.split("_")[0].strip("_-")
        if transect[0] != "t":
            time = filename.split("_")[1].strip("_-")

    return date, transect, time

def calculate_mean_cv2(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Convert the image to RGB (OpenCV uses BGR by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Flatten the image to get a list of pixels
    pixel_data = img.reshape((-1, 3))

    # Separate the RGB values
    red_values = pixel_data[:, 0]
    green_values = pixel_data[:, 1]
    blue_values = pixel_data[:, 2]

    # Calculate mean values
    mean_red = red_values.mean() / 255
    mean_green = green_values.mean() / 255
    mean_blue = blue_values.mean() / 255

    return mean_red, mean_green, mean_blue

def setup(rank):
    name = f"{DATASET_NAME}{rank}"

    dataset_type = fo.types.YOLOv5Dataset
    dataset_dir = DATA_PATH

    # The splits to load
    splits = ["train", "val", "test"]

    # Load the dataset, using tags to mark the samples in each split
    print("Loading dataset")
    dataset = fo.Dataset(name, overwrite=True)
    for split in splits:
        dataset.add_dir(
            dataset_dir=dataset_dir,
            dataset_type=dataset_type,
            yaml_path=DATASET_DESCRIPTION,
            split=split,
            tags=split,
            label_field="detections"
    )

    dataset.persistent = True
    
    print("Adding transects")
    for sample in tqdm(dataset.iter_samples(autosave=True)):
        filename = sample.filepath.split("/")[-1]
        date, transect, time = get_date_and_transect(filename)
        sample["transect"] = transect

    print("Adding dataset attributes")
    for sample in tqdm(dataset):
        if sample.detections is not None:
            detections = sample.detections.detections
            new_detections = []
            for detection in detections:
                if detection["label"] in ORIGINAL_CLASSES:
                    detection["label"] = detection["label"]
                    bounding_box = detection["bounding_box"]
                    detection["bbox_area_percentage"] = bounding_box[2] * bounding_box[3] * 100
                    detection["bbox_aspect_ratio"] = bounding_box[2] / bounding_box[3]
                    if detection["bbox_area_percentage"] > 5: 
                        detection["bbox_area_percentage"] = 5
                    if detection["bbox_aspect_ratio"] > 2:
                        detection["bbox_aspect_ratio"] = 2
                    new_detections.append(detection)
            sample.detections.detections = new_detections
            sample.save()
    
    # print("Adding mean color attribute")
    # for sample in tqdm(dataset):
    #     # print(sample)
    #     # img = Image.open(sample.filepath)
    #     R, G, B = calculate_mean_cv2(sample.filepath)
    #     sample["mean_red"] = R
    #     sample["mean_green"] = G
    #     sample["mean_blue"] = B
    #     sample["is_yellow"] = R > 0.6
    #     sample.save()
    dataset.save()

if __name__ == "__main__":
    setup(0)
    setup(1)
