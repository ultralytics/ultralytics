import os

import cv2
import numpy as np
import requests

confs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

img_width = 1920
img_height = 960
for idx in range(0, 248):
    imgs = []
    for conf in confs:
        path = f"conf_{conf}/cube_result_{idx}.jpg"
        print(path)
        img = cv2.imread(path)
        img = cv2.resize(img, (img_width, img_height))
        cv2.rectangle(img, (0, 0), (img_width, img_height), (0, 0, 255), thickness=5)
        imgs.append(img)
    canva = np.zeros((img_height * 3, img_width * 3, 3), dtype=np.uint8)
    for img_idx, img in enumerate(imgs):
        horizontal = img_idx % 3
        vertical = int(img_idx / 3)
        canva[
            vertical * img_height : (vertical + 1) * img_height, horizontal * img_width : (horizontal + 1) * img_width
        ] = img[:]
    img_file = f"conf_compare1/compare_result_{idx}.jpg"
    cv2.imwrite(img_file, canva)
    print(img_file)
