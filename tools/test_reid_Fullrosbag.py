import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import YOLOManitou_MultiCam
from ultralytics.data.manitou_api import get_manitou_calibrations


def draw_results_on_image(img, boxes, confs, global_indices):
    for i, (box, conf, idx) in enumerate(zip(boxes, confs, global_indices)):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"ID:{idx} Conf:{conf:.2f}"
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return img


# CONFIGURATION
weights = "/root/workspace/ultralytics/tools/runs/manitou_remap/train/weights/best.pt"
device = [0]
imgsz = (1552, 1936)
rosbag_name = "rosbag2_2025_02_17-14_29_31"
calib_path = "/datasets/dataset/manitou/calibration/"
root_dir = f"/datasets/dataset/manitou/key_frames/{rosbag_name}"
output_dir = f"outputs/{rosbag_name}"
os.makedirs(output_dir, exist_ok=True)

# LOAD MODEL
model = YOLOManitou_MultiCam(model="yolo11s.yaml").load(weights)
calib_params = get_manitou_calibrations(calib_path)

# LIST FRAMES
frame_files = sorted(os.listdir(os.path.join(root_dir, "camera1")))
frame_files = [f for f in frame_files if f.endswith(".jpg")]

print(f"Nombre d'images : {len(frame_files)}")

# LOOP OVER FRAMES
for frame_file in frame_files:
    frame_id = os.path.splitext(frame_file)[0]

    cam1_path = os.path.join(root_dir, "camera1", frame_file)
    cam2_path = os.path.join(root_dir, "camera2", frame_file)
    cam3_path = os.path.join(root_dir, "camera3", frame_file)
    cam4_path = os.path.join(root_dir, "camera4", frame_file)

    data_cfg = {
        "camera1": [cam1_path],
        "camera2": [cam2_path],
        "camera3": [cam3_path],
        "camera4": [cam4_path],
        "radar1": None,
        "radar2": None,
        "radar3": None,
        "radar4": None,
        "calib_params": calib_params,
        "filter_cfg": {},
        "radar_accumulation": 1,
    }

    results = model.predict(data_cfg=data_cfg, imgsz=imgsz, conf=0.50, max_det=100, save=False)

    processed_images = []
    global_index = 1
    features_list = []
    camera_idx = []
    all_boxes = []
    all_confs = []
    all_img_indices = []

    for cam_idx, res in enumerate(results[0]):
        img = res.orig_img.copy()
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
        confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
        features = res.feats.cpu().numpy() if res.feats is not None else []
        n = len(boxes)
        indices = list(range(global_index, global_index + n))
        global_index += n

        camera_idx.extend([cam_idx] * n)
        all_boxes.extend(boxes)
        all_confs.extend(confs)
        all_img_indices.extend([cam_idx] * n)

        processed_images.append(img)
        features_list.append(features)

    if not features_list or all(len(f) == 0 for f in features_list):
        print(f"[{frame_id}] No detections.")
        continue

    reid_threshold = 0.7
    features_tensor = torch.tensor(np.concatenate(features_list, axis=0), dtype=torch.float32)
    features_norm = F.normalize(features_tensor, p=2, dim=1)
    similarity_matrix = torch.mm(features_norm, features_norm.t())

    camera_idx_tensor = torch.tensor(camera_idx)
    same_cam_mask = camera_idx_tensor.unsqueeze(0) == camera_idx_tensor.unsqueeze(1)
    similarity_matrix[same_cam_mask] = 0

    top_matches = torch.argmax(similarity_matrix, dim=1).tolist()
    top_scores = torch.max(similarity_matrix, dim=1).values.tolist()

    annotated_images = [img.copy() for img in processed_images]
    for i, (box, conf, img_idx, match_idx, sim_score) in enumerate(
        zip(all_boxes, all_confs, all_img_indices, top_matches, top_scores)
    ):
        x1, y1, x2, y2 = map(int, box)
        label = f"ID: {i + 1}"
        if sim_score > reid_threshold:
            label += f" -> ReID: {match_idx + 1} | {sim_score:.2f}"
        cv2.rectangle(annotated_images[img_idx], (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_images[img_idx], label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    target_size = (1920, 1536)
    resized_images = [cv2.resize(img, target_size) for img in annotated_images]

    if len(resized_images) == 4:
        top_row = np.hstack([resized_images[0], resized_images[2]])
        bottom_row = np.hstack([resized_images[1], resized_images[3]])
        combined_image = np.vstack([top_row, bottom_row])
    else:
        combined_image = np.hstack(resized_images)

    out_path = os.path.join(output_dir, f"{frame_id}.jpg")
    cv2.imwrite(out_path, combined_image)
    print(f"[{frame_id}] Image enregistrée : {out_path}")
