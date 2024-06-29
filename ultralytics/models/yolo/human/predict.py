# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.results import Results
from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils import DEFAULT_CFG, ops
from PIL import Image
import torch


class HumanPredictor(BasePredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a human model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.human import HumanPredictor

        args = dict(model='yolov8n-human.pt', source=ASSETS)
        predictor = HumanPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the HumanPredictor with the provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "human"

    def preprocess(self, img):
        """Converts input image to model-compatible data type."""
        if not isinstance(img, torch.Tensor):
            img = torch.stack([self.transforms(Image.fromarray(im[..., ::-1])) for im in img], dim=0)
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions to return Results objects."""
        preds = preds[0] if isinstance(preds, (tuple, list)) else preds
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            results.append(Results(orig_img, path=self.batch[0][i], names=self.model.names, human=pred))
        return results
