# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        if isinstance(preds[0], tuple):
            for i, pred in enumerate(preds):
                preds[i] = ops.non_max_suppression(
                    pred,
                    self.args.conf,
                    self.args.iou,
                    agnostic=self.args.agnostic_nms,
                    max_det=self.args.max_det,
                    classes=self.args.classes,
                )
        else:
            preds = ops.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det,
                classes=self.args.classes,
            )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        import copy

        tp = copy.copy(preds)
        # preds = preds[0]
        delelethead = []  # 删除头操作
        preds = []  # 图片有多少张就有多少张
        if isinstance(tp[0], list):
            for i in delelethead:
                del tp[i]
            for i, pred in enumerate(tp[0]):  # 每张图片遍历一次,取第一个head的为初始
                res = pred  # preds coco预测
                for j, det in enumerate(tp):  # 遍历每一个头的这张图片下的输出
                    if j == 0:
                        continue
                    det[i][:, -1] += 80 * j
                    res = torch.cat((res, det[i]), dim=0)
                preds.append(res)
        else:
            preds = tp

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
