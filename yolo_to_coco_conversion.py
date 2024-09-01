import glob
import json
import os
import re

# Configuration
yolo_labels_dir = "path/to/yolo/labels"  # Path containing your YOLO .txt files
output_json_path = "path/to/output.json"  # Output COCO format JSON file

# Initialize COCO dataset structure
coco_dataset = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "person",
            "supercategory": "person",
            "keypoints": [
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            ],
            "skeleton": [
                (16, 14),
                (14, 12),
                (17, 15),
                (15, 13),
                (12, 13),
                (6, 12),
                (7, 13),
                (6, 7),
                (6, 8),
                (7, 9),
                (8, 10),
                (9, 11),
                (2, 3),
                (1, 2),
                (1, 3),
                (2, 4),
                (3, 5),
                (4, 6),
                (5, 7),
            ],
        }
    ],
}


# Function to convert YOLO annotations to COCO format
def yolo_to_coco(yolo_annotation, image_id, img_width, img_height, ann_id_start):
    annotations = []
    for ann_line in yolo_annotation.splitlines():
        parts = ann_line.strip().split()
        bbox = [float(x) for x in parts[1:5]]
        keypoints = [float(x) for x in parts[5:]]

        # Convert bbox to COCO format
        x_center, y_center, bbox_width, bbox_height = bbox
        x_top_left = (x_center - bbox_width / 2) * img_width
        y_top_left = (y_center - bbox_height / 2) * img_height
        width = bbox_width * img_width
        height = bbox_height * img_height

        # Process keypoints (assuming they are provided in x, y, visibility order)
        coco_keypoints = []
        for i in range(0, len(keypoints), 3):
            x, y, v = keypoints[i : i + 3]
            x *= img_width
            y *= img_height
            v = 2 if (x != 0 or y != 0) else 0
            coco_keypoints.extend([x, y, int(v)])

        annotations.append(
            {
                "id": ann_id_start,
                "image_id": image_id,
                "category_id": 1,  # Assuming 'person' category
                "bbox": [x_top_left, y_top_left, width, height],
                "area": width * height,
                "iscrowd": 0,
                "keypoints": coco_keypoints,
                "num_keypoints": int(len(coco_keypoints) / 3),
            }
        )
        ann_id_start += 1

    return annotations, ann_id_start


# Iterate over YOLO label files and convert annotations
annotation_id = 1  # Starting ID for COCO annotations
for label_file in glob.glob(os.path.join(yolo_labels_dir, "*.txt")):
    match = re.search(r"image_(\d+)", os.path.basename(label_file))
    if match:
        image_id = int(match.group(1))
    else:
        raise ValueError(f"No numeric part found in filename: {label_file}")

    img_width, img_height = 640, 640  # Replace with actual image sizes

    with open(label_file) as file:
        yolo_annotation = file.read()

    coco_anns, annotation_id = yolo_to_coco(yolo_annotation, image_id, img_width, img_height, annotation_id)
    coco_dataset["annotations"].extend(coco_anns)

    coco_dataset["images"].append(
        {
            "id": image_id,
            "width": img_width,
            "height": img_height,
            "file_name": os.path.basename(label_file).replace(".txt", ".png"),  # Adjust image file extension
        }
    )

# Save the COCO dataset to a JSON file
with open(output_json_path, "w") as f:
    json.dump(coco_dataset, f, indent=4)

print(f"Converted annotations have been saved to {output_json_path}")
