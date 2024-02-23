import json
import yaml
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
