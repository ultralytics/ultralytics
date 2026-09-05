# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics.data.augment import classify_transforms
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops


class ClassificationPredictor(BasePredictor):
    """A class extending the BasePredictor class for prediction based on a classification model.

    This predictor handles the specific requirements of classification models, including preprocessing images and
    postprocessing predictions to generate classification results.

    Attributes:
        args (dict): Configuration arguments for the predictor.

    Methods:
        pre_transform: Resize and crop images on the host before the device-side conversion.
        preprocess: Convert input images to model-compatible format.
        postprocess: Process model predictions into Results objects.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.classify import ClassificationPredictor
        >>> args = dict(model="yolo26n-cls.pt", source=ASSETS)
        >>> predictor = ClassificationPredictor(overrides=args)
        >>> predictor.predict_cli()

    Notes:
        - Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize the ClassificationPredictor with the specified configuration and set task to 'classify'.

        This constructor initializes a ClassificationPredictor instance, which extends BasePredictor for classification
        tasks. It ensures the task is set to 'classify' regardless of input configuration.

        Args:
            cfg (dict): Default configuration dictionary containing prediction settings.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (dict, optional): Dictionary of callback functions to be executed during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "classify"

    def setup_source(self, source):
        """Set up source and inference mode and classify transforms."""
        import torchvision.transforms as T  # scope for faster 'import ultralytics'

        super().setup_source(source)
        transforms = getattr(self.model.model, "transforms", None)  # missing on YAML-built and legacy checkpoints
        size = getattr(transforms.transforms[0], "size", max(self.imgsz)) if transforms is not None else None
        self.transforms = (
            transforms if size == max(self.imgsz) and self.model.format == "pt" else classify_transforms(self.imgsz)
        )
        tfl = getattr(self.transforms, "transforms", ())
        split = (
            type(self.transforms) is T.Compose
            and tuple(map(type, tfl)) == (T.Resize, T.CenterCrop, T.ToTensor, T.Normalize)
            and getattr(self.model, "channels", 3) == 3
        )
        self.host_transforms = T.Compose(tfl[:2]) if split else None
        self.device_transform = tfl[-1] if split else None

    def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
        """Resize and crop images on the host, leaving uint8 BGR for the device-side conversion."""
        return [np.array(self.host_transforms(Image.fromarray(x))) for x in im]

    def preprocess(self, img):
        """Convert input images to model-compatible tensor format with appropriate normalization."""
        if self.device_transform is None and not isinstance(img, torch.Tensor):
            img = torch.stack([self.transforms(Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))) for x in img], 0)
            img = img.to(self.model.device)
            return img.half() if self.model.fp16 else img.float()
        is_tensor = isinstance(img, torch.Tensor)
        img = super().preprocess(img)
        return img if is_tensor else self.device_transform(img)

    def postprocess(self, preds, img, orig_imgs):
        """Process predictions to return Results objects with classification probabilities.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Input images after preprocessing.
            orig_imgs (list[np.ndarray] | torch.Tensor): Original images before preprocessing.

        Returns:
            (list[Results]): List of Results objects containing classification results for each image.
        """
        if not isinstance(orig_imgs, list):  # Input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        return [
            Results(orig_img, path=img_path, names=self.model.names, probs=pred)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]
