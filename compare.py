import os

import cv2
import requests
import numpy as np

lixiang = '/home/lixiang/PycharmProjects/ultralytics/val_res'
chengye = '/home/lixiang/下载/val'

for img in os.listdir(lixiang):
    lixiang_img = cv2.imread(os.path.join(lixiang, img))
    chengye_img = cv2.imread(os.path.join(chengye, img))
    size = 600
    lixiang_img = cv2.resize(lixiang_img, (size, size))
    chengye_img = cv2.resize(chengye_img, (size, size))
    hconcat = cv2.hconcat([lixiang_img, chengye_img])
    cv2.imshow('img', hconcat)
    cv2.waitKey(0)
