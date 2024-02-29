# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2
import torch
from PIL import Image

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.nn.autobackend import AutoBackend


class RegressionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a classification model.

    Notes:
        - Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.regress import RegressionPredictor

        args = dict(model='yolov8n-regress.pt', source=ASSETS)
        predictor = RegressionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes RegressionPredictor setting the task to 'regress'."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "regress"
        self._legacy_transform_name = "ultralytics.yolo.data.augment.ToTensor"
        self.max = None
        self.min = None

    def preprocess(self, img):
        """Converts input image to model-compatible data type."""
        if not isinstance(img, torch.Tensor):
            is_legacy_transform = any(
                self._legacy_transform_name in str(transform) for transform in self.transforms.transforms
            )
            if is_legacy_transform:  # to handle legacy transforms
                img = torch.stack([self.transforms(im) for im in img], dim=0)
            else:
                img = torch.stack(
                    [self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
                )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions to return Results objects."""
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        if not self.max:
            if not (isinstance(self.model, AutoBackend) and not self.model.pt):
                layers = list(self.model.model.model._modules.keys())
            self.max = self.model.metadata["max_value"] if (isinstance(self.model, AutoBackend) and not self.model.pt) else self.model.model.model._modules[layers[-1]].max
            self.min = self.model.metadata["min_value"] if (isinstance(self.model, AutoBackend) and not self.model.pt) else self.model.model.model._modules[layers[-1]].min
        results = []
        for i in range(len(preds)):
            preds[i] = preds[i] * (self.max - self.min) / 6 + self.min
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, regress=pred))
        return results
