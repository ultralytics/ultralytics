# -*- coding: utf-8 -*-
"""
@Author: captainfffsama
@Date: 2023-08-17 13:56:28
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-11-03 12:37:59
@FilePath: /ultralytics/ultralytics/data/chiebot_augment/origin_ag_ext.py
@Description:
"""
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union, List
from functools import wraps
import random

from albumentations.core.transforms_interface import DualTransform
import albumentations.augmentations.crops.functional as ACF


def skip_class_support(cls):
    """transform support skip some class now"""
    original_init = cls.__init__

    @wraps(original_init)
    def __init__(self, *args, skip_class_idx: Optional[Tuple[int]] = tuple(), **kwargs):
        self.skip_class = skip_class_idx
        original_init(self, *args, **kwargs)

    cls.__init__ = __init__

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
        label_idx = data["cls"]
        skip_idx = np.array([x in self.skip_class for x in label_idx], dtype=bool)

        if skip_idx.any():
            return data
        else:
            return original_call(self, data)

    cls.__call__ = __call__
    return cls


class CropBox(DualTransform):
    def __init__(
        self,
        part_shift_range: Tuple[float, float] = (0.3, 0.3),
        crop_label_idx: List[int] = (),
        min_wh: float = 0.05,
        always_apply=False,
        p=1.0,
    ):
        """以某个目标为准裁剪图片

        Args:
            part_shift_range (Tuple[float, float], optional): 扩大box裁剪时的范围,按照图片对应边的百分比来确定. Defaults to (0.3, 0.3).
            crop_label_idx (List[int], optional): 仅涉及这些label时才裁剪. Defaults to ().
            min_wh (float, optional): 跳过过小的box,即长或宽小于这个百分比的box跳过. Defaults to 0.05.
            always_apply (bool, optional): _description_. Defaults to False.
            p (float, optional): _description_. Defaults to 1.0.

        Raises:
            ValueError: _description_
        """
        super(CropBox, self).__init__(always_apply, p)
        self.part_shift_range = part_shift_range
        self.crop_label_idx = crop_label_idx
        self.min_wh = min_wh

        if min(self.part_shift_range) < 0 or max(self.part_shift_range) > 1:
            raise ValueError(
                "Invalid part_shift_range. Got: {}".format(part_shift_range)
            )

    def apply(
        self,
        img: np.ndarray,
        x_min: int = 0,
        x_max: int = 0,
        y_min: int = 0,
        y_max: int = 0,
        **params,
    ) -> np.ndarray:
        return ACF.clamping_crop(img, x_min, y_min, x_max, y_max)

    def apply_to_bbox(self, bbox, **params):
        return ACF.bbox_crop(
            bbox,
            **params,
        )

    def apply_to_keypoint(
        self,
        keypoint: Tuple[float, float, float, float],
        x_min: int = 0,
        x_max: int = 0,
        y_min: int = 0,
        y_max: int = 0,
        **params,
    ) -> Tuple[float, float, float, float]:
        return ACF.crop_keypoint_by_coords(
            keypoint, crop_coords=(x_min, y_min, x_max, y_max)
        )

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("part_shift_range", "crop_label_idx", "min_wh")

    def apply_with_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        image = kwargs.get("image", None)
        # have been convert to albumentations format,like voc but norm,(xmin,ymin,xmax,ymax,label)
        bboxes: List[Tuple[float, float, float, float, int]] = kwargs.get("bboxes", [])

        if image is None or not bboxes:
            print("image or bboxes")
            return kwargs

        bbox = self._select_box(bboxes)
        if bbox is None:
            print("bbox is None")
            return kwargs
        bbox = (
            int(bbox[0] * image.shape[1]),
            int(bbox[1] * image.shape[0]),
            int(bbox[2] * image.shape[1]),
            int(bbox[3] * image.shape[0]),
            bbox[4],
        )
        params = self._get_crop_box(bbox, image.shape[1], image.shape[0])
        r = super(CropBox, self).apply_with_params(params, **kwargs)
        return r

    def _get_crop_box(self, bbox, img_w, img_h) -> Dict[str, int]:
        h_shift_range = (
            round(img_h * self.part_shift_range[0]),
            round(img_h * self.part_shift_range[1]),
        )
        w_shift_range = (
            round(img_w * self.part_shift_range[0]),
            round(img_w * self.part_shift_range[1]),
        )

        x_min = bbox[0] - random.randint(*w_shift_range)
        x_max = bbox[2] + random.randint(*w_shift_range)

        y_min = bbox[1] - random.randint(*h_shift_range)
        y_max = bbox[3] + random.randint(*h_shift_range)

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w, x_max)
        y_max = min(img_h, y_max)

        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def _select_box(
        self, bboxes: Tuple[float, float, float, float, int]
    ) -> Union[Tuple[float, float, float, float, int], None]:
        available_bboxes = []
        for box in bboxes:
            if (
                box[-1] in self.crop_label_idx
                and box[2] - box[0] >= self.min_wh
                and box[3] - box[1] >= self.min_wh
            ):
                available_bboxes.append(box)
        if available_bboxes:
            return random.choice(available_bboxes)
        else:
            return None
