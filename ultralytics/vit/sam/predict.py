import cv2
import torch
import numpy as np

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.engine.results import Results
from .modules.mask_generator import SamAutomaticMaskGenerator

class Predictor(BasePredictor):
    def setup_model(self, model):
        device = select_device(self.args.device)
        model.eval()
        self.model = SamAutomaticMaskGenerator(model.to(device), pred_iou_thresh=self.args.conf, box_nms_thresh=self.args.iou)
        self.device = device

    def postprocess(self, preds, path, orig_imgs):
        names = dict(enumerate(list(range(len(preds)))))
        masks = torch.from_numpy(np.stack([p["segmentation"] for p in preds], axis=0))
        result = Results(orig_img=orig_imgs, path=path, names=names, masks=masks)
        return result

    def __call__(self, source=None, model=None, stream=False):
        frame = cv2.imread(source)
        preds = self.model.generate(frame)
        return self.postprocess(preds, source, frame)
