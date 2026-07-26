# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG


class HumanPredictor(DetectionPredictor):
    """A class extending the DetectionPredictor class for prediction based on a human model.

    This class specializes in human attribute estimation, attaching weight, height, age, gender and ethnicity
    predictions to the detections inherited from DetectionPredictor.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO-Human model.

    Methods:
        construct_result: Construct the result object from the prediction, including human attributes.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.human import HumanPredictor
        >>> args = dict(model="yolov8n-human.pt", source=ASSETS)
        >>> predictor = HumanPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize the HumanPredictor with the provided configuration, overrides, and callbacks.

        Args:
            cfg (Any): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (dict, optional): Dictionary of callback functions to be invoked during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "human"

    def construct_result(self, pred, img, orig_img, img_path):
        """Construct the result object from the prediction, including human attributes.

        Args:
            pred (torch.Tensor): Predicted boxes, scores and attributes with shape (N, 17).
            img (torch.Tensor): The processed input image tensor with shape (B, C, H, W).
            orig_img (np.ndarray): The original unprocessed image as a numpy array.
            img_path (str): The path to the original image file.

        Returns:
            (Results): The result object containing boxes and the predicted human attributes.
        """
        result = super().construct_result(pred, img, orig_img, img_path)
        result.update(human=pred[:, 6:])
        return result
