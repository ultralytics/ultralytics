import json
import yaml
import os
import fiftyone as fo

from ultralytics.config import ROOT_DIR


def modify_coco_json(annotations_path):
    # annotations_path = f"../data/DUO/annotations/instances_{split}"
    # Load the JSON file
    with open(f"{annotations_path}.json", "r") as f:
        annotations = json.load(f)

    # Iterate through annotations
    for annotation in annotations["annotations"]:
        # Check if the 'segmentation' key is present and it's not in the correct format
        if (
            "segmentation" in annotation
            and isinstance(annotation["segmentation"], list)
            and not isinstance(annotation["segmentation"][0], list)
        ):
            # Convert the segmentation to the correct format
            annotation["segmentation"] = [annotation["segmentation"]]

    # Save the modified JSON back to file
    with open(f"{annotations_path}_corrected.json", "w") as f:
        json.dump(annotations, f)


def load_dataset_fiftyone(data_path, labels_path):
    dataset_name = "export_dataset"

    dataset_type = fo.types.COCODetectionDataset

    # The splits to load
    splits = ["train", "test"]

    # Load the dataset, using tags to mark the samples in each split
    print("Loading dataset")
    dataset = fo.Dataset(dataset_name, overwrite=True)
    for split in splits:
        dataset.add_dir(
            data_path=data_path(split),
            labels_path=labels_path(split),
            dataset_type=dataset_type,
            tags=split,
        )
    return dataset


def export_dataset(
    split, dataset_path, dataset: fo.Dataset, dataset_type=fo.types.YOLOv5Dataset, label_field="detections"
):
    classes = dataset.default_classes[1:]

    # Export dataset

    dataset.match_tags(split).export(
        export_dir=f"{dataset_path}/{split}",
        dataset_type=dataset_type,
        label_field=label_field,
        classes=classes,
    )

    # Create yaml file
    d = {"train": "train", "val": "test", "nc": len(classes), "names": classes}

    with open(f"{dataset_path}/data.yaml", "w") as f:
        yaml.dump(d, f, sort_keys=False, indent=2)


def convert_dataset_yolo(dataset_folder="DUO"):
    """Converts a COCO dataset to YOLO format using FiftyOne."""

    dataset_path = f"{ROOT_DIR}/data/{dataset_folder}"
    for split in ["train", "test"]:
        annotations_path = f"{dataset_path}/annotations/instances_{split}"
        modify_coco_json(annotations_path)

    data_path = lambda split: f"{dataset_path}/images/{split}"
    labels_path = lambda split: f"{dataset_path}/annotations/instances_{split}_corrected.json"

    dataset = load_dataset_fiftyone(data_path, labels_path)

    for split in ["train", "test"]:
        dataset_path = f"{ROOT_DIR}/data/{dataset_folder}"
        export_dataset(split, dataset_path, dataset)

    return dataset


def create_missing_label_files(root_dir):
    """
    Generate label txt files where there are missing files.

    Args:
        root_dir (str): root directory of the repo
    """
    # Define the root directory
    data_dir = os.path.join(root_dir, "data", "data")

    # Iterate through train, valid, and test folders
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(data_dir, split)

        # Iterate through images folder in each split
        images_dir = os.path.join(split_dir, "images")
        for image_file in os.listdir(images_dir):
            image_name, image_ext = os.path.splitext(image_file)
            label_file = image_name + ".txt"
            label_path = os.path.join(split_dir, "labels", label_file)

            # Check if the label file exists, if not create an empty one
            if not os.path.exists(label_path):
                with open(label_path, "w") as f:
                    pass  # Create an empty file


def convert_mask_to_yolo_bbox(yolo_annotation: str):
    """
    Converts a file line from mask format to yolo bbox format.

    Args:
        yolo_annotation (str): string on the format `<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>`

    Returns:
        float: converted values with `class_id, x_center, y_center, width, height`
    """
    # Parse the YOLO annotation string
    parts = yolo_annotation.split()
    class_id = int(parts[0])
    coordinates = list(map(float, parts[1:]))

    # Extract x and y coordinates
    x_coords = coordinates[::2]
    y_coords = coordinates[1::2]

    # Calculate bounding box coordinates
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # Calculate box width and height
    width = x_max - x_min
    height = y_max - y_min

    # Calculate box center
    x_center = x_min + width / 2
    y_center = y_min + height / 2

    return class_id, x_center, y_center, width, height


def convert_annotations(folder):
    """
    Converts label txt files from mask format to yolo bbox format.

    Args:
        folder (str): path to dataset folder
    """
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(subdir, file)
                with open(file_path, "r") as f:
                    lines = f.readlines()
                with open(file_path, "w") as f:
                    for line in lines:
                        if len(line.split()) > 5:  # Must be mask if more than 5 numbers in line
                            class_id, x_center, y_center, width, height = convert_mask_to_yolo_bbox(line)
                            bbox_line = f"{class_id} {x_center} {y_center} {width} {height}\n"
                            f.write(bbox_line)


def convert_mask_data_to_yolo_bbox():
    """Convert dataset labels from mask to yolo bbox format and creates txt files that do not exist."""
    # Example usage
    create_missing_label_files(ROOT_DIR)

    # Change the directory accordingly
    convert_annotations(os.path.join(ROOT_DIR, "data", "data"))
