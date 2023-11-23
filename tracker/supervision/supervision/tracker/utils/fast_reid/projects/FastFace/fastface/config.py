# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

from supervision.tracker.utils.fast_reid.fastreid.config import CfgNode as CN


def add_face_cfg(cfg):
    _C = cfg

    _C.DATASETS.REC_PATH = ""

    _C.MODEL.BACKBONE.DROPOUT = 0.

    _C.MODEL.HEADS.PFC = CN({"ENABLED": False})
    _C.MODEL.HEADS.PFC.SAMPLE_RATE = 0.1
