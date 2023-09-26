# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-08-17 13:56:28
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-08-18 12:41:04
@FilePath: /ultralytics/ultralytics/data/chiebot_augment/origin_ag_ext.py
@Description:
'''
import numpy as np
from typing import Tuple,Optional
from functools import wraps


def skip_class_support(cls):
    """transform support skip some class now
    """
    original_init=cls.__init__
    @wraps(original_init)
    def __init__(self,*args,skip_class_idx:Optional[Tuple[int]]=tuple(),**kwargs):
        self.skip_class=skip_class_idx
        original_init(self,*args,**kwargs)

    cls.__init__=__init__

    original_call = cls.__call__

    def __call__(self, data):
        """
        labels = {
            "im_file":str img_path
            "cls": Nx1 np.ndarray class labels
            "img": HxWx3 np.ndarray image
            "ori_shape": Tuple[int,int] origin hw
            "resized_shape": Tuple[int,int] resized HW
            "ratio_pad": Tuple[float,float] ratio of H/h W/w
            "instances": ultralytics/utils/instance.py:Instances
        }
        """
        label_idx = data['cls']
        skip_idx = np.array([x in self.skip_class for x in label_idx], dtype=bool)

        if skip_idx.any():
            return data
        else:
            return original_call(self,data)

    cls.__call__ = __call__
    return cls
