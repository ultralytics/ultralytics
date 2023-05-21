# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.ops import xyxy2xywh
from ultralytics.yolo.v8.detect import DetectionValidator

__all__ = ['NASValidator']


class NASValidator(DetectionValidator):

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
