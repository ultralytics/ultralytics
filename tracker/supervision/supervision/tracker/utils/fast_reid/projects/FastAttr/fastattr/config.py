# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

from supervision.tracker.utils.fast_reid.fastreid.config import CfgNode as CN


def add_attr_config(cfg):
    _C = cfg

    _C.MODEL.LOSSES.BCE = CN({"WEIGHT_ENABLED": True})
    _C.MODEL.LOSSES.BCE.SCALE = 1.

    _C.TEST.THRES = 0.5
