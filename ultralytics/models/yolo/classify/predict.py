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

        def flatten(tfl):
            """Inline nested Compose transforms into one ordered list of leaf transforms."""
            return [x for t in tfl for x in (flatten(t.transforms) if type(t) is T.Compose else [t])]

        super().setup_source(source)
        transforms = getattr(self.model.model, "transforms", None)  # missing on YAML-built and legacy checkpoints
        size = getattr(transforms.transforms[0], "size", max(self.imgsz)) if transforms is not None else None
        self.transforms = (
            transforms if size == max(self.imgsz) and self.model.format == "pt" else classify_transforms(self.imgsz)
        )
        # Split at ToTensor so PIL ops stay on the host and tensor ops run once per batch on device, which is
        # only valid for the shape classify_transforms() emits. Before it, Resize/CenterCrop resample each
        # channel independently, so deferring BGR to RGB past them is exact; a channel-mixing stage such as
        # Grayscale or ColorJitter needs real RGB order. After it, Normalize preserves dtype and applies
        # elementwise, so batching it is equivalent; ConvertImageDtype would override the model-dtype cast and
        # a random transform would draw once per batch instead of once per image. Anything else keeps the
        # original single-pass host path. Matched by exact type, not isinstance: a subclass may override
        # __call__ with behavior that depends on being invoked once per image rather than once per batch.
        tfl = flatten([self.transforms])  # a Compose subclass stays an opaque leaf, so the split rejects it
        i = next((i for i, t in enumerate(tfl) if type(t) is T.ToTensor), None)
        split = (
            i is not None
            and all(type(t) in (T.Resize, T.CenterCrop) for t in tfl[:i])
            and all(type(t) is T.Normalize for t in tfl[i + 1 :])
        )
        self.pil_transforms = T.Compose(tfl[:i]) if split else None
        self.device_transforms = T.Compose(tfl[i + 1 :]) if split else None

    def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
        """Resize and crop images on the host, leaving uint8 BGR for the device-side conversion."""
        return [np.array(self.pil_transforms(Image.fromarray(x))) for x in im]

    def preprocess(self, img):
        """Convert input images to model-compatible tensor format with appropriate normalization."""
        if self.device_transforms is None and not isinstance(img, torch.Tensor):  # non-splittable pipeline
            img = torch.stack([self.transforms(Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))) for x in img], 0)
            img = img.to(self.model.device)
            return img.half() if self.model.fp16 else img.float()
        is_tensor = isinstance(img, torch.Tensor)
        img = super().preprocess(img)
        return img if is_tensor else self.device_transforms(img)

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
