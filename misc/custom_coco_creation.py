import os
import shutil

#root_path = '/Users/inaki-eab/Desktop/datasets'
root_path = '/data-fast/127-data2/ierregue/datasets'

dataset_name = 'custom_coco'

new_dataset_root = os.path.join(root_path, dataset_name)

# Create folder structure
if not os.path.isdir(new_dataset_root):
    os.makedirs(new_dataset_root)
    os.makedirs(os.path.join(new_dataset_root, 'labels','train'))
    os.makedirs(os.path.join(new_dataset_root, 'labels','test'))
    os.makedirs(os.path.join(new_dataset_root, 'labels','val'))
    os.makedirs(os.path.join(new_dataset_root, 'images','train'))
    os.makedirs(os.path.join(new_dataset_root, 'images','test'))
    os.makedirs(os.path.join(new_dataset_root, 'images','val'))

wanted_indices = [
    0, #person
    2, #car
    4, #airplane
    5, #bus
    7, #truck
    8, #boat
]


def clean_map_file(file):
    # Mapping should be performed only once

    class_index_map = {
        0: 0,  # person/personnel
        2: 1,  # car
        7: 2,  # truck
        5: 2,  # bus to truck
        4: 4,  # airplane/aircraft
        8: 5  # boat/ship
    }

    with open(file, 'r+') as fp:
        # read an store all lines into list
        lines = fp.readlines()
        # move file pointer to the beginning of a file
        fp.seek(0)
        # truncate the file
        fp.truncate()

        # start writing lines
        # iterate line and line number
        for number, line in enumerate(lines):
            # Only write rows of interested instances
            old_class_id = int(line.split()[0])
            if old_class_id in wanted_indices:
                # map old class indices to new ones
                new_class_id = class_index_map[old_class_id]
                new_line_splitted = line.split()
                new_line_splitted[0] = str(new_class_id)
                new_line = ' '.join(new_line_splitted)
                fp.write(new_line + '\n')


def move_desired_files(
        original_dataset_root,  # ../datasets/old_dataset
        target_dataset_root,  # ../datasets/new_dataset
        original_dataset_slice,  # liketrain,test,val
        target_dataset_slice,  # train,test,val
        wanted_indices  # list of desired indices
):
    # Empty list to store the selected files containing at list one of the desired objects
    selected_images = []

    original_labels_dir = os.path.join(original_dataset_root, 'labels', original_dataset_slice)
    original_images_dir = os.path.join(original_dataset_root, 'images', original_dataset_slice)

    # Iterate over all files in the original dataset labels folder
    for filename in os.listdir(original_labels_dir):
        if filename.endswith('.txt'):
            # Read file
            with open(os.path.join(original_labels_dir, filename), "r") as f:
                # Empty list to store objects/instances present in image
                indices_in_file = []
                # Iterate over instances in image and get present class ids
                for line in f:
                    indices_in_file.append(int(line.split()[0]))
                # If any present class ids is a class id of interest, get its filename
                if any((True for x in indices_in_file if x in wanted_indices)):
                    # Get only name, no '.txt' extension
                    selected_images.append(os.path.splitext(filename)[0])
                    # Copy *.txt folder
                    shutil.copy(os.path.join(original_labels_dir, filename),
                                os.path.join(target_dataset_root, 'labels', target_dataset_slice))
                    # Copy *jpg image
                    img_path = os.path.join(original_images_dir, os.path.splitext(filename)[0] + '.jpg')
                    shutil.copy(img_path, os.path.join(target_dataset_root, 'images', target_dataset_slice))

                    # Map old index to new one and delete unwanted instances
                    clean_map_file(os.path.join(target_dataset_root, 'labels', target_dataset_slice, filename))

    return selected_images

original_dataset_path = 'coco'
original_dataset_root = os.path.join(root_path, original_dataset_path)

val_indices = move_desired_files(original_dataset_root,
                                   new_dataset_root,
                                   'val2017',
                                   'val',
                                   wanted_indices)

train_indices = move_desired_files(original_dataset_root,
                                   new_dataset_root,
                                   'train2017',
                                   'train',
                                   wanted_indices)