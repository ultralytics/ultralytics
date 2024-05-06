import os

import cv2
import requests
import numpy as np

lixiang = '/home/lixiang/下载/TRI_road_infer/TRI_road_infer/pano_val/lixiang'
chengye = '/home/lixiang/下载/TRI_road_infer/TRI_road_infer/pano_val/chengye'

for img in os.listdir(lixiang):
    lixiang_img = cv2.imread(os.path.join(lixiang, img))
    (height, width, depth) = lixiang_img.shape
    cv2.line(lixiang_img, (width, 0), (width, height), (0, 0, 255), thickness=20)
    chengye_img = cv2.imread(os.path.join(chengye, img))
    # size = 600
    # lixiang_img = cv2.resize(lixiang_img, (size * 2, size))
    # chengye_img = cv2.resize(chengye_img, (size * 2, size))
    hconcat = cv2.hconcat([lixiang_img, chengye_img])
    # cv2.imshow('img', hconcat)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join('/home/lixiang/下载/TRI_road_infer/TRI_road_infer/pano_val', 'compare', img), hconcat)
