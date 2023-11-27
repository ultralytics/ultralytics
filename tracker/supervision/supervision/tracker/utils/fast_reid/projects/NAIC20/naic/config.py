# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""


def add_naic_config(cfg):
    _C = cfg

    _C.DATASETS.RM_LT = True
    _C.TEST.SAVE_DISTMAT = False
