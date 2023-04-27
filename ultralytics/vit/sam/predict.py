import numpy as np
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils.torch_utils import select_device

from .modules.mask_generator import SamAutomaticMaskGenerator


class Predictor(BasePredictor):

    def preprocess(self, im):
        """Prepares input image for inference."""
        # TODO: Only support bs=1 for now
        # im = ResizeLongestSide(1024).apply_image(im[0])
        # im = torch.as_tensor(im, device=self.device)
        # im = im.permute(2, 0, 1).contiguous()[None, :, :, :]
        return im[0]

    def setup_model(self, model):
        """Set up YOLO model with specified thresholds and device."""
        device = select_device(self.args.device)
        model.eval()
        self.model = SamAutomaticMaskGenerator(model.to(device),
                                               pred_iou_thresh=self.args.conf,
                                               box_nms_thresh=self.args.iou)
        self.device = device
        # TODO: Temporary settings for compatibility
        self.model.pt = False
        self.model.triton = False
        self.model.stride = 32
        self.model.fp16 = False
        self.done_warmup = True

    def postprocess(self, preds, path, orig_imgs):
        """Postprocesses inference output predictions to create detection masks for objects."""
        names = dict(enumerate(list(range(len(preds)))))
        results = []
        # TODO
        for i, pred in enumerate([preds]):
            masks = torch.from_numpy(np.stack([p['segmentation'] for p in pred], axis=0))
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=names, masks=masks))
        return results

    # def __call__(self, source=None, model=None, stream=False):
    #     frame = cv2.imread(source)
    #     preds = self.model.generate(frame)
    #     return self.postprocess(preds, source, frame)
