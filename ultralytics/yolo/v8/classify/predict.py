# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT


class ClassificationPredictor(BasePredictor):

    def preprocess(self, img):
        """Converts input image to model-compatible data type."""
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32

    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses predictions to return Results objects."""
        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, probs=pred))

        return results


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Run YOLO model predictions on input images/videos."""
    model = cfg.model or 'yolov8n-cls.pt'  # or "resnet18"
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = ClassificationPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()
