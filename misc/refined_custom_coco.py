# TO GO FROM COCO TO YOLO:

#from ultralytics.data.converter import convert_coco
#convert_coco('/Users/inaki-eab/Desktop/testing_archs/refined_annotations/sama-coco-val')
#convert_coco('/Users/inaki-eab/Desktop/testing_archs/refined_annotations/sama-coco-train')
# To merge folders:
"""
#!/bin/bash

# Set the source directory containing subfolders
SOURCE_DIR="/Users/inaki-eab/Desktop/small-fast-detector/misc/yolo_labels/labels"

# Set the destination directory for merging
DESTINATION_DIR="/Users/inaki-eab/Desktop/testing_archs/refined_annotations/labels/train"

# Ensure the destination directory exists
mkdir -p "$DESTINATION_DIR"

# Move files from subfolders to the destination directory
find "$SOURCE_DIR" -mindepth 2 -type f -exec mv -i {} "$DESTINATION_DIR" \;

echo "Merging completed."
"""
import os
import shutil
import re

#CREATING A NEW EMPTY DATASET
root_path = '/data-fast/127-data2/ierregue/datasets'

dataset_name = 'refined_custom_coco'

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

# List of classes of interest in the original dataset
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
    slice_postfix = 'val2017' if 'val' in original_dataset_slice else 'train2017'
    original_images_dir = os.path.join(original_dataset_root, 'images', slice_postfix)

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
                                   'val2017_refined',
                                   'val',
                                   wanted_indices)

train_indices = move_desired_files(original_dataset_root,
                                   new_dataset_root,
                                   'train2017_refined',
                                   'train',
                                   wanted_indices)

# Rename files to match the new dataset
old_len_img_train = len(os.listdir(os.path.join(new_dataset_root,'images','train')))
old_len_img_val = len(os.listdir(os.path.join(new_dataset_root,'images','val')))
old_len_txt_val = len(os.listdir(os.path.join(new_dataset_root,'labels','val')))
old_len_txt_train = len(os.listdir(os.path.join(new_dataset_root,'labels','train')))


def rename_dir(dir, shift=1):
    files = [f for f in os.listdir(dir)]
    files.sort(key=lambda x: int(re.match(r'\d+', x).group()))

    for idx, old_filename in enumerate(files):
        extension = os.path.splitext(old_filename)[1]
        new_filename = f'{idx + shift}{extension}'
        old_path = os.path.join(dir, old_filename)
        new_path = os.path.join(dir, new_filename)
        os.rename(old_path, new_path)

rename_dir(os.path.join(new_dataset_root,'images','train'), shift=1+50495)
rename_dir(os.path.join(new_dataset_root,'images','val'), shift=1+7595)
rename_dir(os.path.join(new_dataset_root,'labels','train'), shift=1+50495)
rename_dir(os.path.join(new_dataset_root,'labels','val'), shift=1+7595)

# Inspect the new dataset
import pandas as pd
import imagesize
import matplotlib.pyplot as plt

class_names = {
    0:'person',
    1:'car',
    2:'truck',
    3:'uav',
    4:'airplane',
    5:'ship'
}


def inspect_dataset(
        target_dataset_root,  # ../datasets/new_dataset
        target_dataset_slice,  # train,test,val
):
    info = []  # list of lists, each list corresponds to an instance [cls_id, x, y, w, h, img]

    target_labels_dir = os.path.join(target_dataset_root, 'labels', target_dataset_slice)

    # Iterate over all files in the original dataset labels folder
    for filename in os.listdir(target_labels_dir):
        if filename.endswith('.txt'):
            # Read file
            with open(os.path.join(target_labels_dir, filename), "r") as f:
                # Iterate over instances in image and get present class ids
                for line in f:
                    line_data = []
                    # label data
                    line_data = line.split()
                    # Image name
                    line_data.append(os.path.splitext(filename)[0])
                    # Image size: could be done at image level and not row level
                    img_path = os.path.join(target_dataset_root, 'images', target_dataset_slice,
                                            os.path.splitext(filename)[0] + '.jpg')
                    img_w, img_h = imagesize.get(img_path)
                    line_data.extend([img_w, img_h])
                    # Append line data to info
                    info.append(line_data)

    df = pd.DataFrame(info, columns=['new_class_id', 'xcn', 'ycn', 'wn', 'hn', 'img', 'img_w', 'img_h'])
    df = df.astype('float32').astype({'new_class_id': 'int32', 'img': 'int32'})
    df['class_name'] = df['new_class_id'].map(class_names)
    return df

df = inspect_dataset(new_dataset_root, 'train')

print(f"The number of objects is {len(df)}")
print(f"The number of images is {len(df['img'].unique())}")

# Create dir to store plots
save_dir = './data/dataset_creation'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

fig = plt.figure(figsize=(8, 6))
ax = df['class_name'].value_counts().plot(kind='bar')
# Set x-axis label
ax.set_xlabel("Classes", weight='bold', size=12)
# Set y-axis label
ax.set_ylabel("Counts", weight='bold', size=12)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
None
fig.savefig(save_dir+'/refined_coco_class_counts.png', bbox_inches = 'tight')

fig = plt.figure(figsize=(10, 6))
ax = df.groupby(by=['img'])['img'].count().value_counts().sort_index().plot(kind='bar')
# Set x-axis label
ax.set_xlabel("Instances per image", weight='bold', size=12)
# Set y-axis label
ax.set_ylabel("Counts", weight='bold', size=12)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
None
fig.savefig(save_dir+'/refined_instances_count.png', bbox_inches = 'tight')

df['bbox_area'] = (df['wn']*df['img_w'])*(df['hn']*df['img_h'])
bin_edges = [0, 16**2, 32**2, 96**2, float('inf')]
bin_labels = ['tiny', 'small', 'medium', 'large']
df['bbox_size_category'] = pd.cut(df['bbox_area'], bins=bin_edges, labels=bin_labels, right=False)

fig = plt.figure(figsize=(8, 6))
ax = df['bbox_size_category'].value_counts().sort_index().plot(kind='bar')
# Set x-axis label
ax.set_xlabel("Object category size", weight='bold', size=12, labelpad=10)
# Set y-axis label
ax.set_ylabel("Counts", weight='bold', size=12)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
None
fig.savefig(save_dir+'/refined_coco_objsz_counts.png', bbox_inches = 'tight')