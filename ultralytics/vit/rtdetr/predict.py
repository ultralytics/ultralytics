# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops


class RTDETRPredictor(BasePredictor):

    def postprocess(self, preds, img, orig_imgs):
        """Postprocess predictions and returns a list of Results objects."""
        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        results = []
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1, keepdim=True)  # (300, 1)
            idx = score.squeeze(-1) > self.args.conf  # (300, )
            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
            pred = torch.cat([bbox, score, cls], dim=-1)[idx]  # filter
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            oh, ow = orig_img.shape[:2]
            if not isinstance(orig_imgs, torch.Tensor):
                pred[..., [0, 2]] *= ow
                pred[..., [1, 3]] *= oh
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def pre_transform(self, im):
        """Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Return: A list of transformed imgs.
        """
        # The size must be square(640) and scaleFilled.
        return [LetterBox(self.imgsz, auto=False, scaleFill=True)(image=x) for x in im]
