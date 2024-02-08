import argparse
import os
import json
from PIL import Image
from tqdm import tqdm
import yaml

def load_categories(yaml_path):
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    categories = [{'id': int(k), 'name': v} for k, v in yaml_data['names'].items()]
    return categories

def yolo_to_coco(image_dir, label_dir, categories):
    # Initialize data dict
    data = {'info': {}, 'licenses': [], 'images': [], 'annotations': [], 'categories': categories}

    # Get image and label files
    image_files = sorted(os.listdir(image_dir))

    # Loop over images
    cumulative_id = 0
    with tqdm(total=len(image_files), desc='Processing images') as pbar:
        for filename in image_files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue  # Skip non-image files

            image_path = os.path.join(image_dir, filename)
            try:
                im = Image.open(image_path)
            except (IOError, OSError):
                print(f"Error opening image: {filename}")
                continue

            # Use filename with extension as image_id
            im_id = filename

            data['images'].append({
                'id': im_id,
                'file_name': filename,
                'width': im.size[0],
                'height': im.size[1]
            })

            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(label_dir, label_filename)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    yolo_data = f.readlines()

                for line in yolo_data:
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    class_id = int(class_id)
                    bbox_x = (x_center - width / 2) * im.size[0]
                    bbox_y = (y_center - height / 2) * im.size[1]
                    bbox_width = width * im.size[0]
                    bbox_height = height * im.size[1]

                    data['annotations'].append({
                        'id': cumulative_id,
                        'image_id': im_id,
                        'category_id': class_id,
                        'bbox': [bbox_x, bbox_y, bbox_width, bbox_height],
                        'area': bbox_width * bbox_height,
                        'iscrowd': 0
                    })

                    cumulative_id += 1
            else:
                print(f"No label file found for image: {filename}")

            pbar.update(1)

    # Save data to JSON file
    # output_file = os.path.join(image_dir, 'annotations.json')
    with open('annotations.json', 'w') as f:
        json.dump(data, f, indent=4)

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert YOLO format labels to COCO JSON format.")
    parser.add_argument("--image-dir", type=str, help="Directory containing the images.")
    parser.add_argument("--label-dir", type=str, default=None, help="Directory containing the YOLO label files.")
    parser.add_argument("--yaml-path", type=str, help="Path to the YAML file containing category names.")

    args = parser.parse_args()

    categories = load_categories(args.yaml_path)
    label_dir = args.label_dir if args.label_dir else args.image_dir.replace('images', 'labels')
    yolo_to_coco(args.image_dir, label_dir, categories)
    print("Conversion completed successfully.")
