# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.ops import xyxy2xywh
from ultralytics.yolo.utils.torch_utils import select_device


class NASPredictor(BasePredictor):

    def postprocess(self, preds_in, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""

        # Cat boxes and class scores
        boxes = xyxy2xywh(preds_in[0][0])
        preds = torch.cat((boxes, preds_in[0][1]), -1).permute(0, 2, 1)

        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        device = select_device(self.args.device, verbose=verbose)
        model = model or self.args.model
        model.stride = torch.tensor([32])
        model.names = {i: f'class{i}' for i in range(1000)}
        self.args.half &= device.type != 'cpu'  # half precision only supported on CUDA
        self.model = AutoBackend(model,
                                 device=device,
                                 dnn=self.args.dnn,
                                 data=self.args.data,
                                 fp16=self.args.half,
                                 fuse=False,
                                 verbose=verbose)
        self.device = device
        self.model.eval()
