import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import YOLOManitou_MultiCam
from ultralytics.data.manitou_api import ManitouAPI, get_manitou_calibrations

# Dataset
# data_root = '/home/shu/Documents/PROTECH/ultralytics/datasets/manitou'
data_root = "/datasets/dataset/manitou/"
radar_dir = "radars"
cam_dir = "key_frames"
calib_params = get_manitou_calibrations("/datasets/dataset/manitou/calibration/")

annotations_path = os.path.join(data_root, "annotations_multi_view", "manitou_coco_val_remap.json")
manitou = ManitouAPI(annotations_path)
manitou.info()


def get_bag_ids_from_name(name):
    valid_bag_ids = manitou.get_bag_ids()
    for bag_id in valid_bag_ids:
        if manitou.bags[bag_id]["name"] == name:
            return bag_id
    return -1


def draw_results_on_image(img, boxes, confs, global_indices):
    for i, (box, conf, idx) in enumerate(zip(boxes, confs, global_indices)):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"ID:{idx} Conf:{conf:.2f}"
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return img


# get bag ids
selected_bag_id1 = get_bag_ids_from_name("rosbag2_2025_01_22-10_53_25")
selected_bag_id2 = get_bag_ids_from_name("rosbag2_2025_01_22-11_28_05")
selected_bag_id3 = get_bag_ids_from_name("rosbag2_2025_01_22-11_40_06")
selected_bag_id4 = get_bag_ids_from_name("rosbag2_2025_01_22-11_48_43")
selected_bag_id5 = get_bag_ids_from_name("rosbag2_2025_02_17-14_25_51")
selected_bag_id6 = get_bag_ids_from_name("rosbag2_2025_02_17-14_29_31")

video_ids = manitou.get_vid_ids(bagIds=selected_bag_id5)

cam1_ids = []
cam2_ids = []
cam3_ids = []
cam4_ids = []

for video_id in video_ids:
    imgs = manitou.get_img_ids_from_vid(video_id)
    if "camera1" in manitou.vids[video_id]["name"]:
        cam1_ids = imgs
    elif "camera2" in manitou.vids[video_id]["name"]:
        cam2_ids = imgs
    elif "camera3" in manitou.vids[video_id]["name"]:
        cam3_ids = imgs
    elif "camera4" in manitou.vids[video_id]["name"]:
        cam4_ids = imgs

cam1_paths = []
cam2_paths = []
cam3_paths = []
cam4_paths = []
radar1_paths = []
radar2_paths = []
radar3_paths = []
radar4_paths = []

start = 5
for c1, c2, c3, c4 in zip(
    cam1_ids[start : start + 1], cam2_ids[start : start + 1], cam3_ids[start : start + 1], cam4_ids[start : start + 1]
):
    assert (
        manitou.imgs[c1]["frame_name"]
        == manitou.imgs[c2]["frame_name"]
        == manitou.imgs[c3]["frame_name"]
        == manitou.imgs[c4]["frame_name"]
    ), "Frame names do not match across cameras"
    cam1_name = manitou.imgs[c1]["file_name"]
    cam2_name = manitou.imgs[c2]["file_name"]
    cam3_name = manitou.imgs[c3]["file_name"]
    cam4_name = manitou.imgs[c4]["file_name"]
    cam1_paths.append(os.path.join(data_root, cam_dir, cam1_name))
    cam2_paths.append(os.path.join(data_root, cam_dir, cam2_name))
    cam3_paths.append(os.path.join(data_root, cam_dir, cam3_name))
    cam4_paths.append(os.path.join(data_root, cam_dir, cam4_name))
    radar1_name = manitou.radars[c1]["file_name"]
    radar2_name = manitou.radars[c2]["file_name"]
    radar3_name = manitou.radars[c3]["file_name"]
    radar4_name = manitou.radars[c4]["file_name"]
    radar1_paths.append(os.path.join(data_root, radar_dir, radar1_name))
    radar2_paths.append(os.path.join(data_root, radar_dir, radar2_name))
    radar3_paths.append(os.path.join(data_root, radar_dir, radar3_name))
    radar4_paths.append(os.path.join(data_root, radar_dir, radar4_name))

data_cfg = {
    "camera1": cam1_paths,
    "camera2": cam2_paths,
    "camera3": cam3_paths,
    "camera4": cam4_paths,
    "radar1": radar1_paths,
    "radar2": radar2_paths,
    "radar3": radar3_paths,
    "radar4": radar4_paths,
    "calib_params": calib_params,
    "filter_cfg": {},
    "radar_accumulation": 3,  # number of frames to accumulate for radar
}

# project = 'runs/manitou_remap'
# data_cfg = "/root/workspace/ultralytics/ultralytics/cfg/datasets/manitou.yaml"
weights = "/root/workspace/ultralytics/tools/runs/manitou_remap/train/weights/best.pt"
device = [0]
imgsz = (1552, 1936)  # (height, width)
model = YOLOManitou_MultiCam(model="yolo11s.yaml").load(weights)
results = model.predict(data_cfg=data_cfg, imgsz=imgsz, conf=0.50, max_det=100, save=False)

processed_images = []
global_index = 1
features_list = []

for cam_idx, res in enumerate(results[0]):
    img = res.orig_img.copy()
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
    confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
    features = res.feats.cpu().numpy() if res.feats is not None else []
    n = len(boxes)
    indices = list(range(global_index, global_index + n))
    global_index += n

    img = draw_results_on_image(img, boxes, confs, indices)
    processed_images.append(img)
    features_list.append(features)

features_tensor = torch.tensor(np.concatenate(features_list, axis=0), dtype=torch.float32)

dist_matrix = torch.mm(features_tensor, features_tensor.t())

features_norm = F.normalize(features_tensor, p=2, dim=1)
similarity_matrix = torch.mm(features_norm, features_norm.t())

print("Cosine similarity matrix :")
for i in range(similarity_matrix.shape[0]):
    line = [f"{similarity_matrix[i, j].item():.3f}" for j in range(similarity_matrix.shape[1])]
    print(" ".join(line))

target_size = (1920, 1536)
resized_images = [cv2.resize(img, target_size) for img in processed_images]

top_row = np.hstack([resized_images[0], resized_images[2]])
bottom_row = np.hstack([resized_images[1], resized_images[3]])
combined_image = np.vstack([top_row, bottom_row])

cv2.imwrite("grid_output.jpg", combined_image)
print("Image enregistrée : grid_output.jpg")
