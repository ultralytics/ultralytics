"""yaml_names = ['UAV', 'aeroplane', 'boat', 'car', 'person', 'truck']

# Your COCO categories
coco_categories = [{"id": 0, "name": "person"},
                   {"id": 1, "name": "car"},
                   {"id": 2, "name": "truck"},
                   {"id": 3, "name": "uav"},
                   {"id": 4, "name": "aeroplane"},
                   {"id": 5, "name": "boat"}]

# Mapping YAML names to COCO category IDs
yaml_to_coco_id = {}

for name in yaml_names:
    for category in coco_categories:
        if category['name'] == name.lower():
            yaml_to_coco_id[name] = category['id']
            break

print(yaml_to_coco_id)
"""
import os


def modify_class_ids(file_path, yaml_to_coco_id):
    modified_lines = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            elements = line.strip().split()
            yolo_class_id = int(elements[0])

            # Map YOLO class ID to the new ID
            new_class_id = yaml_to_coco_id.get(yolo_class_id, yolo_class_id)  # Default to original if no mapping

            # Modify the line with the new class ID
            elements[0] = str(new_class_id)
            modified_line = ' '.join(elements)
            modified_lines.append(modified_line)

    return modified_lines


# YAML to COCO category ID mapping
yaml_to_coco_id = {
    0: 3,  # UAV to uav
    1: 4,  # aeroplane to airplane
    2: 5,  # boat to boat
    3: 1,  # car to car
    4: 0,  # person to person
    5: 2  # truck to truck
}

# Directory containing .txt files
txt_directory = '../inference_tools/Evaluation/datasets/Client_Validation_Set/labels/val'

# Iterate and modify each .txt file
for txt_file in os.listdir(txt_directory):
    if txt_file.endswith('.txt'):
        print("Modifying file: " + txt_file)
        txt_file_path = os.path.join(txt_directory, txt_file)

        # Get the modified lines with updated class IDs
        modified_annotations = modify_class_ids(txt_file_path, yaml_to_coco_id)

        # Write the modified annotations back to the file
        with open(txt_file_path, 'w') as file:
            for line in modified_annotations:
                file.write(line + "\n")

print("Class IDs have been updated in all .txt files.")