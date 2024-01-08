# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-08-18 16:35:03
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2024-01-08 16:55:20
@FilePath: /ultralytics/ultralytics/grpc_server/model.py
@Description:
'''
import base64
from collections import defaultdict
from typing import Union, Tuple
import torch
from torch import nn
import datetime

from ..models import YOLO
from ..engine.results import Results
from ultralytics.nn.modules import SPPF, SPP
from ultralytics.utils import LOGGER
import numpy as np
import cv2

from .proto import dldetection_pb2
from .proto import dldetection_pb2_grpc as dld_pb2_grpc

from .utils import restore_bbox_after_letterbox, np2tensor_proto


class Detector(dld_pb2_grpc.AiServiceServicer):

    def __init__(self,
                 ckpt_path,
                 thr: Union[float, dict],
                 change_label: dict = {},
                 device: str = 'cuda:0',
                 nms: float = 0.5):
        self.model = YOLO(ckpt_path)
        self.label_dict: dict = self.model.model.names
        for k, v in self.label_dict.items():
            new_label_name = change_label.get(v, None)
            if new_label_name:
                self.label_dict[k] = new_label_name
        self.device = device
        self.nms = nms
        if isinstance(thr, float):
            self.thr = defaultdict(lambda: thr)
        else:
            if 'default' not in thr.keys() and len(thr.keys()) != len(self.label_dict.values()):
                raise ValueError("thr args must be dict or float or have default values")
            else:
                if 'default' not in thr.keys():
                    self.thr = thr
                else:
                    default_value = thr.pop('default')
                    self.thr = defaultdict(lambda: default_value)
                    self.thr.update(thr)
        print("model init done!")

        self.embedding_model = None

    def _standardized_result(self, result: Results) -> list:
        bboxes: torch.Tensor = result.boxes.data[:, :4].detach()
        scores: torch.Tensor = result.boxes.data[:, 4].detach()
        labels: torch.Tensor = result.boxes.cls.detach().type(torch.int64)
        if hasattr(result, "letter_box_info"):
            bboxes, keep_ind = restore_bbox_after_letterbox(bboxes,
                                                            ratio=result.letter_box_info["ratio"],
                                                            dwdh=result.letter_box_info["dwdh"],
                                                            ret_ind=True)
            scores = scores[keep_ind]
            labels = labels[keep_ind]
        thr_tensor = torch.tensor([self.thr[l] for l in labels], dtype=scores.dtype, device=labels.device)
        remain_idx = scores >= thr_tensor
        remain_bboxes = bboxes[remain_idx].cpu().numpy()
        remain_scores = scores[remain_idx].cpu().numpy()
        remain_labels = labels[remain_idx].cpu().numpy()

        final_result = []
        for box, s, l in zip(remain_bboxes, remain_scores, remain_labels):
            final_result.append((self.label_dict[l], s, *box.tolist()))
        return final_result

    def infer(self, img):
        result: Results = self.model.predict(img, conf=0.05, device=self.device, iou=self.nms)[0]
        new_result = self._standardized_result(result)
        return new_result

    def DlDetection(self, request, context):
        img_base64 = base64.b64decode(request.imdata)

        img_array = np.fromstring(img_base64, np.uint8)
        img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB | cv2.IMREAD_IGNORE_ORIENTATION)
        if len(img.shape) <3:
            img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        result = self.infer(img)
        current_time = datetime.datetime.now()
        LOGGER.info("{} --- result:{}".format(current_time.strftime("%Y-%m-%d %H:%M:%S"), result))
        result_pro = dldetection_pb2.DlResponse()
        for obj in result:
            obj_pro = result_pro.results.add()
            obj_pro.classid = obj[0]
            obj_pro.score = float(obj[1])
            obj_pro.rect.x = int(obj[2])
            obj_pro.rect.y = int(obj[3])
            obj_pro.rect.w = int(obj[4] - obj[2])
            obj_pro.rect.h = int(obj[5] - obj[3])
        torch.cuda.empty_cache()
        return result_pro

    def DlEmbeddingGet(self, request, context):
        img_base64 = base64.b64decode(request.imdata)

        img_array = np.fromstring(img_base64, np.uint8)
        img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB | cv2.IMREAD_IGNORE_ORIENTATION)

        im_size = tuple(request.imsize)

        result: np.ndarray = self.extract_feat(self.model, img, img_size=im_size)
        current_time = datetime.datetime.now()
        LOGGER.info("{} --- embedding size:{}".format(current_time.strftime("%Y-%m-%d %H:%M:%S"), result.shape))
        torch.cuda.empty_cache()
        return np2tensor_proto(result)

    def extract_feat(self, model, img, img_size: Tuple[int, int]):
        im: np.ndarray = cv2.resize(img, img_size)
        im = np.expand_dims(im, axis=0)
        im = im.transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

        img = im.to(self.device)
        # TODO: 这里可能有fp16问题
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0

        if self.embedding_model is None:
            self.embedding_model = nn.Sequential()
            all_nn = next(model.model.children())
            for i in all_nn:
                self.embedding_model.append(i)
                if isinstance(i, SPPF) or isinstance(i, SPP):
                    break
            self.embedding_model.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.embedding_model.to(self.device)
            self.embedding_model.eval()

        with torch.no_grad():
            result = self.embedding_model(img)
            result = result.cpu().numpy()
        return result
