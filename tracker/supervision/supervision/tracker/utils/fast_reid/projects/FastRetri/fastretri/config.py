# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""


def add_retri_config(cfg):
    _C = cfg

    _C.TEST.RECALLS = [1, 2, 4, 8, 16, 32]
