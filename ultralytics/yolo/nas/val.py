# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.yolo.data import YOLODataset
from ultralytics.yolo.data.augment import Compose, Format, LetterBox
from ultralytics.yolo.utils import colorstr, ops
from ultralytics.yolo.utils.ops import xyxy2xywh
from ultralytics.yolo.v8.detect import DetectionValidator

__all__ = ['NASValidator']


# TODO: Temporarily, RT-DETR does not need padding.
class NASDataset(YOLODataset):

    def __init__(self, *args, data=None, **kwargs):
        super().__init__(*args, data=data, use_segments=False, use_keypoints=False, **kwargs)

    def build_transforms(self, hyp=None):
        """Temporarily, only for evaluation."""
        transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scaleFill=True)])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms


class NASValidator(DetectionValidator):

    def build_dataset(self, img_path, mode='val', batch=None):
        """Build YOLO Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return NASDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f'{mode}: '),
            data=self.data)

    def postprocess(self, preds_in):
        """Apply Non-maximum suppression to prediction outputs."""
        boxes = xyxy2xywh(preds_in[0][0])
        preds = torch.cat((boxes, preds_in[0][1]), -1).permute(0, 2, 1)
        return ops.non_max_suppression(preds,
                                       self.args.conf,
                                       self.args.iou,
                                       labels=self.lb,
                                       multi_label=False,
                                       agnostic=self.args.single_cls,
                                       max_det=self.args.max_det,
                                       max_time_img=0.5)
