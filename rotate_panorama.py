import os.path
import time

import requests
import torch

from ultralytics import YOLO
import cv2
import numpy as np
import math
from equilib import equi2pers

def get_theta_phi(_x, _y, _z):
    dv = math.sqrt(_x * _x + _y * _y + _z * _z)
    x = _x / dv
    y = _y / dv
    z = _z / dv
    theta = math.atan2(y, x)
    phi = math.asin(z)
    return theta, phi


# x,y position in cubemap
# cw  cube width
# W,H size of equirectangular image
def map_cube(x, y, side, cw, W, H):
    u = 2 * (float(x) / cw - 0.5)
    v = 2 * (float(y) / cw - 0.5)

    if side == "front":
        theta, phi = get_theta_phi(1, u, v)
    elif side == "right":
        theta, phi = get_theta_phi(-u, 1, v)
    elif side == "left":
        theta, phi = get_theta_phi(u, -1, v)
    elif side == "back":
        theta, phi = get_theta_phi(-1, -u, v)
    elif side == "bottom":
        theta, phi = get_theta_phi(-v, u, 1)
    elif side == "top":
        theta, phi = get_theta_phi(v, u, -1)

    _u = 0.5 + 0.5 * (theta / math.pi)
    _v = 0.5 + (phi / math.pi)
    return int(_u * W), int(_v * H)


# Load a model
model = YOLO('runs/detect/train33/weights/best.pt')  # pretrained YOLOv8n model
# Run batched inference on a list of images
images = [
    'https://ow-prod-cdn.survey.work/platform_id_1/app_id_null/roled_user_id_null/type_1/52f3a0273dfb466e85e6bf59ade05c2e.jpg',

]

img_names = [x.split('/')[-1].split('.')[0] for x in images]

name_dict = {
    0: '路面破损',
    # 1: '沿街晾晒',
    # 2: '垃圾满冒',
    3: '乱扔垃圾',
    # 4: '垃圾正常盛放',
}

confs = [0.1]

unit = 180 / math.pi

for idx_conf in confs:
    if not os.path.exists(f'conf_{idx_conf}'):
        os.mkdir(f'conf_{idx_conf}')
    for img_idx, image_url in enumerate(images):
        now = time.time()
        res = requests.get(image_url)
        img = cv2.imdecode(np.fromstring(res.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        # img = cv2.imread(image_url)
        if img is None:
            continue
        # img = cv2.imread(image_url)
        print(f'下载图片耗时{time.time() - now:.2f}s')
        # img = e2c(img, face_w=int(img.shape[0] / 3))
        now = time.time()
        (height, width, depth) = img.shape
        pic = np.zeros((height, width, depth))
        cut_width = int(width / 4)
        cut_height = int(height / 3)
        equi_img = np.transpose(img, (2, 0, 1))
        yaw = math.pi

        equi_img = torch.tensor(equi_img)
        pers_should_height = height / 4
        pers_should_width = width / 4
        for i in range(36):
            print(math.degrees(yaw))
            # rotations
            # pitch = np.pi / 8
            pitch = 0
            rots = {
                'roll': 0.,
                'pitch': pitch,  # rotate vertical
                'yaw': yaw,  # rotate horizontal
            }
            # Run equi2pers
            fov_deg = 90.0
            pers_height = 480
            pers_width = 640
            pers_img = equi2pers(
                equi=equi_img,
                rots=rots,
                height=pers_height,
                width=pers_width,
                fov_x=fov_deg,
                mode="bilinear",
            )
            cube_result = np.ascontiguousarray(np.transpose(pers_img, (1, 2, 0)))
            cv2.circle(cube_result, (10, 10), 10, (0, 0, 255), -1)
            cv2.imshow('img', cube_result)
            cv2.waitKey(0)
            # cv2.imwrite(f'rotate_{i}.jpg', cube_result)
            _u = 0.5 + 0.5 * (-yaw / math.pi)
            _v = 0.5 + (pitch / math.pi)
            center = np.array([int(pers_width / 2), int(pers_height / 2)])
            vec = np.array([10 - center[0], 10 - center[1]])
            vec = [int(vec[0] * pers_should_width / pers_width), int(vec[1] * pers_should_height / pers_height)]
            center_equi_pos = np.array([int(_u * width), int(_v * height)])
            print(f'center_equi_pos: {center_equi_pos}')
            print(f'vec: {vec}')
            point = center_equi_pos + vec
            point[0] = point[0] + width if point[0] < 0 else point[0]
            point = [a for a in point]
            print(f'offset point: {point}')
            cv2.circle(img, center_equi_pos, 10, (0, 0, 255), -1)
            cv2.circle(img, point, 10, (0, 0, 255), -1)
            cv2.imshow('img', cv2.resize(img, (int(width / 4), int(height / 4))))
            cv2.waitKey(0)
            yaw -= np.pi / 18
