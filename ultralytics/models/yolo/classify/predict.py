# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
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

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the ClassificationPredictor with the specified configuration and set task to 'classify'.

        This constructor initializes a ClassificationPredictor instance, which extends BasePredictor for classification
        tasks. It ensures the task is set to 'classify' regardless of input configuration.

        Args:
            cfg (dict): Default configuration dictionary containing prediction settings.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be executed during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "classify"

    def setup_source(self, source):
        """Set up source and inference mode and classify transforms."""
        super().setup_source(source)
        updated = (
            self.model.model.transforms.transforms[0].size != max(self.imgsz)
            if hasattr(self.model.model, "transforms") and hasattr(self.model.model.transforms.transforms[0], "size")
            else False
        )
        self.transforms = (
            classify_transforms(self.imgsz) if updated or not self.model.pt else self.model.model.transforms
        )

    def preprocess(self, img):
        """Convert input images to model-compatible tensor format with appropriate normalization."""
        if not isinstance(img, torch.Tensor):
            img = torch.stack(
                [self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
            )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()  # Convert uint8 to fp16/32

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
