# This Python code converts a dataset in YOLO format into the COCO format.
# The YOLO dataset contains images of bottles and the bounding box annotations in the
# YOLO format. The COCO format is a widely used format for object detection datasets.

# The input and output directories are specified in the code. The categories for
# the COCO dataset are also defined, with only one category for "bottle". A dictionary for the COCO dataset is initialized with empty values for "info", "licenses", "images", and "annotations".

# The code then loops through each image in the input directory. The dimensions
# of the image are extracted and added to the COCO dataset as an "image" dictionary,
# including the file name and an ID. The bounding box annotations for each image are
# read from a text file with the same name as the image file, and the coordinates are
# converted to the COCO format. The annotations are added to the COCO dataset as an
# "annotation" dictionary, including an ID, image ID, category ID, bounding box coordinates,
# area, and an "iscrowd" flag.

# The COCO dataset is saved as a JSON file in the output directory.

import json
import os
from PIL import Image

# Set the paths for the input and output directories
# input_dir = '/data-fast/127-data2/ierregue/datasets/custom_dataset_v1'
input_dir = '../inference_tools/datasets/Client_Validation_Set'
output_dir = input_dir

# Define the categories for the COCO dataset
categories = [{"id": 0, "name": "person"},
              {"id": 1, "name": "car"},
              {"id": 2, "name": "truck"},
              {"id": 3, "name": "uav"},
              {"id": 4, "name": "airplane"},
              {"id": 5, "name": "boat"}]


# Define the COCO dataset dictionary
coco_dataset = {
    "info": {},
    "licenses": [],
    "categories": categories,
    "images": [],
    "annotations": []
}

# Loop through the images in the input directory
val_images_dir = os.path.join(input_dir, 'images', 'val')
for image_file in os.listdir(val_images_dir):

    # Load the image and get its dimensions
    image_path = os.path.join(val_images_dir, image_file)
    image = Image.open(image_path)
    width, height = image.size

    # Add the image to the COCO dataset
    image_dict = {
        "id": int(image_file.split('.')[0]),
        "width": width,
        "height": height,
        "file_name": image_file
    }
    coco_dataset["images"].append(image_dict)

    # Load the bounding box annotations for the image
    with open(os.path.join(input_dir, 'labels', 'val', f'{image_file.split(".")[0]}.txt')) as f:
        annotations = f.readlines()

    # Loop through the annotations and add them to the COCO dataset
    for ann in annotations:
        x, y, w, h = map(float, ann.strip().split()[1:])
        x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
        x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
        ann_dict = {
            "id": len(coco_dataset["annotations"]),
            "image_id": int(image_file.split('.')[0]),
            "category_id": int(ann.strip().split()[0]),
            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
            "area": (x_max - x_min) * (y_max - y_min),
            "iscrowd": 0
        }
        coco_dataset["annotations"].append(ann_dict)

# Save the COCO dataset to a JSON file
val_annotations_path = os.path.join(output_dir, 'annotations')
if not os.path.exists(val_annotations_path):
    os.makedirs(val_annotations_path)

# JSON file is saved as 'instances_val2017.json' in the output directory to avoid more code modification
with open(os.path.join(val_annotations_path, 'instances_val2017.json'), 'w') as f:
    json.dump(coco_dataset, f)