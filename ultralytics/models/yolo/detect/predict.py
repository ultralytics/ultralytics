# Ultralytics YOLO 🚀, AGPL-3.0 license
import torch
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.nn.modules.block import DFL
from ultralytics.yolo.utils.tal import dist2bbox, make_anchors


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
    def decode_bbox(self, preds, img):
        import math
        num_classes = next((o.shape[2] for o in preds if o.shape[2] != 64), -1)
        strides = [int(math.sqrt(img.shape[1] * img.shape[2] / o.shape[1])) for o in preds if o.shape[2] != 64]
        assert num_classes != -1, 'cannot infer postprocessor inputs via output shape if there are 64 classes'
        pos =  [i for i,_ in sorted(enumerate(preds), key = lambda x: (x[1].shape[2] if num_classes > 64 else -x[1].shape[2], -x[1].shape[1]))]
        x = torch.permute(torch.cat([torch.cat([preds[i] for i in pos[:len(pos)//2]], 1), torch.cat([preds[i] for i in pos[len(pos)//2:]], 1)], 2), (0, 2, 1))
        reg_max = (x.shape[1] - num_classes) // 4
        dfl = DFL(reg_max) if reg_max > 1 else torch.nn.Identity()
        img_h, img_w = img.shape[1], img.shape[2]
        dims = [(img_h // s, img_w // s) for s in strides]
        fake_feats = [torch.zeros((1, 1, h, w), device=self.device) for h, w in dims] 
        anchors, strides = (x.transpose(0, 1) for x in make_anchors(fake_feats, strides, 0.5))  # generate anchors and strides
        dbox = dist2bbox(dfl(x[:,:-num_classes,:].cpu()).to(self.device), anchors.unsqueeze(0), xywh=True, dim=1) * strides
        return torch.cat((dbox, x[:,-num_classes:, :].sigmoid()), 1)

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        if len(preds) == 6:  # DeGirum export
            preds = self.decode_bbox(preds, img)
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        is_list = isinstance(orig_imgs, list)  # input images are a list, not a torch.Tensor
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if is_list else orig_imgs
            if is_list:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
