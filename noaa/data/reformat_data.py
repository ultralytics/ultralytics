import numpy as np
import shutil
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import time

MAX_RETRIES = 5
RETRY_DELAY = 2  # 2 seconds

def custom_copy(src, dest):
    with open(src, 'rb') as fsrc, open(dest, 'wb') as fdst:
        shutil.copyfileobj(fsrc, fdst)

def copy_image(data, dest_folder):
    image_names = []
    for img_info in tqdm(data):
        src = img_info['file_name']

        # Ensure the destination folder exists; if not, create it
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        # Extract the filename from the source path
        folder_name = os.path.basename(os.path.dirname(os.path.dirname(src)))
        filename = os.path.basename(src)
        filename = 'folder_'+folder_name+'-file_'+filename

        # Create the full destination path
        dest = os.path.join(dest_folder, filename)

        # Assert the filename is not in the current save folder
        assert not os.path.exists(dest), f"File '{filename}' already exists in destination folder"

        # Copy the file
        shutil.copy(src, dest)
        image_names.append(filename)


def transform_save_box(data, dest_folder):
    for img_info in tqdm(data):
        src = img_info['file_name']
        # Extract the filename from the source path
        filename = os.path.basename(src)
        filename = filename.replace('jpg', 'txt')

        folder_name = os.path.basename(os.path.dirname(os.path.dirname(src)))
        filename = 'folder_' + folder_name + '-file_' + filename

        annotations = img_info['annotations']

        # Ensure the destination folder exists; if not, create it
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        # Create the full destination path
        dest = os.path.join(dest_folder, filename)

        # Assert the filename is not in the current save folder
        assert not os.path.exists(dest), f"File '{filename}' already exists in destination folder"

        with open(dest, "w") as file:
            for ann in annotations:
                category_id = ann['category_id']
                bbox = ann['bbox']
                xmin = bbox[0]
                ymin = bbox[1]
                xmax = bbox[2]
                ymax = bbox[3]
                img_height = img_info['height']
                img_width = img_info['width']

                x_center = (xmin+xmax)/(2* img_width)
                y_center = (ymin + ymax)/(2*img_height)
                box_w = (xmax - xmin)/img_width
                box_h = (ymax - ymin)/img_height
                file.write(f"{category_id} {x_center} {y_center} {box_w} {box_h}\n")

def multiprocess_task(data, dest_folder, task):
    with Pool(cpu_count()) as pool:
        # Use partial to create a new function with dest_folder "baked in"
        func = partial(task, dest_folder=dest_folder)
        list(tqdm(pool.imap(func, data), total=len(data)))

def copy_worker(img_info, dest_folder):
    src = img_info['file_name']
    if not os.path.exists(dest_folder):
        # The directory will also not be overridden if it already exists.
        os.makedirs(dest_folder, exist_ok=True)
    filename = os.path.basename(src)
    dest = os.path.join(dest_folder, filename)
    assert not os.path.exists(dest), f"File '{filename}' already exists in destination folder"

    for _ in range(MAX_RETRIES):
        try:
            custom_copy(src, dest)
            break
        except OSError as e:
            if "Input/output error" in str(e):
                time.sleep(RETRY_DELAY)
            else:
                raise e


def transform_worker(img_info, dest_folder):
    src = img_info['file_name']
    filename = os.path.basename(src).replace('jpg', 'txt')
    annotations = img_info['annotations']
    if not os.path.exists(dest_folder):
        # The directory will also not be overridden if it already exists.
        os.makedirs(dest_folder, exist_ok=True)
    dest = os.path.join(dest_folder, filename)
    assert not os.path.exists(dest), f"File '{filename}' already exists in destination folder"
    with open(dest, "w") as file:
        for ann in annotations:
            category_id = ann['category_id']
            bbox = ann['bbox']
            xmin, ymin, xmax, ymax = bbox
            img_height = ann['height']
            img_width = ann['width']
            x_center = (xmin+xmax)/(2* img_width)
            y_center = (ymin + ymax)/(2*img_height)
            box_w = (xmax - xmin)/img_width
            box_h = (ymax - ymin)/img_height
            file.write(f"{category_id} {x_center} {y_center} {box_w} {box_h}\n")

npz_file = '/home/jiemei/Documents/rail_detection/dataset_preprocess/rail_data_yolov8/dataset_dicts.npz'
save_folder = '/run/user/1000/gvfs/afp-volume:host=IPL-NOAA.local,user=Jie%20Mei,volume=homes/Jie Mei/yolov8_rail_data/yolov8_rail_data_full'

# all data info
dataset_dicts = list(np.load(npz_file, allow_pickle=True)['arr_0'])

# partial data for quick training
dataset_dicts = dataset_dicts

# split into train and val, 9:1
total_data_num = len(dataset_dicts)
idx = int(total_data_num*0.9)
train_data = dataset_dicts[:idx]
val_data = dataset_dicts[idx:]

# copy images to a new folder on FTP
copy_image(train_data, save_folder+'/images/train')
copy_image(val_data, save_folder+'/images/val')

# create and save train val txt labels in a folder on FTP
# class x_center y_center width height format. Box coordinates must be in normalized xywh format (from 0 to 1)
transform_save_box(train_data, save_folder+'/labels/train')
transform_save_box(val_data, save_folder+'/labels/val')

# Replace your original function calls with multiprocessed versions:
# multiprocess_task(train_data, save_folder+'/images/train', copy_worker)
# multiprocess_task(val_data, save_folder+'/images/val', copy_worker)
# multiprocess_task(train_data, save_folder+'/labels/train', transform_worker)
# multiprocess_task(val_data, save_folder+'/labels/val', transform_worker)