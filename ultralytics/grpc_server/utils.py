# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-08-18 16:29:57
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-08-18 18:20:11
@FilePath: /ultralytics/ultralytics/grpc/utils.py
@Description:
'''
import os
import base64

import numpy as np
import cv2

from .proto import dldetection_pb2
import torch
from torchvision.ops import box_area

def cv2imread(img_path,flag=cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION):
    img=cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),flag)
    return img

def get_img(img_info):
    if os.path.isfile(img_info):
        if not os.path.exists(img_info):
            return None
        else:
            return cv2imread(img_info,cv2._COLOR|cv2.IMREAD_IGNORE_ORIENTATION)  #ignore
    else:
        img_str = base64.b64decode(img_info)
        img_np = np.fromstring(img_str, np.uint8)
        return cv2.imdecode(img_np, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)


def np2tensor_proto(np_ndarray: np.ndarray):
    shape = list(np_ndarray.shape)
    data = np_ndarray.flatten().tolist()
    tensor_pb = dldetection_pb2.Tensor()
    tensor_pb.shape.extend(shape)
    tensor_pb.data.extend(data)
    return tensor_pb


def tensor_proto2np(tensor_pb):
    np_matrix = np.array(tensor_pb.data,
                         dtype=np.float).reshape(tensor_pb.shape)
    return np_matrix

def version_gt(version_current:str,version_benchmark:str="1.37.0") -> bool:
    for i,j in zip(version_current.split("."),version_benchmark.split(".")):
        if int(i)>=int(j):
            return True

    return False

def restore_bbox_after_letterbox(bboxes, ratio, dwdh, letter_size=(640, 640), area_thresh=0.5, ret_ind=False):
    area_src = box_area(bboxes)  # 原来的面积

    bboxes = bboxes.clone()#兼容老版的inplace错误
    bboxes[:, 0::2] -= dwdh[0]
    bboxes[:, 1::2] -= dwdh[1]

    letter_size = (letter_size[0] - 2 * dwdh[0], letter_size[1] - 2 * dwdh[1])
    bboxes_after_handle = bboxes.clone()
    bboxes_after_handle[:, 0::2] = torch.clamp(bboxes_after_handle[:, 0::2], 0, letter_size[0] - 1)
    bboxes_after_handle[:, 1::2] = torch.clamp(bboxes_after_handle[:, 1::2], 0, letter_size[1] - 1)
    area_dst = box_area(bboxes_after_handle.clone())
    keep_ind = torch.where((area_dst / area_src) > area_thresh)

    if ret_ind:
        return bboxes_after_handle[keep_ind] / ratio, keep_ind
    else:
        return bboxes_after_handle[keep_ind] / ratio