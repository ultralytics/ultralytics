# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.results import MDEResults
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class MDEPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a Monocular Depth Estimation (MDE) model.

    This predictor specializes in depth estimation tasks, handling depth values alongside standard object detection
    capabilities inherited from DetectionPredictor.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO MDE model with depth estimation capabilities.

    Methods:
        postprocess: Process raw model predictions into MDE results with depth information.
        construct_result: Construct the result object from the prediction, including depth values.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.depth import MDEPredictor
        >>> args = dict(model="yolo11n-mde.pt", source=ASSETS)
        >>> predictor = MDEPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize MDEPredictor for monocular depth estimation tasks.

        Sets up an MDEPredictor instance, configuring it for depth estimation tasks.

        Args:
            cfg (Any): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.

        Examples:
            >>> from ultralytics.utils import ASSETS
            >>> from ultralytics.models.yolo.depth import MDEPredictor
            >>> args = dict(model="yolo11n-mde.pt", source=ASSETS)
            >>> predictor = MDEPredictor(overrides=args)
            >>> predictor.predict_cli()
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "mde"

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Construct the result object from the prediction, including depth values.

        Extends the parent class implementation by extracting depth data from predictions and adding them to the
        result object.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and depth with shape (N, 7) where N is
                the number of detections. Format: [x1, y1, x2, y2, conf, cls, depth].
            img (torch.Tensor): The processed input image tensor with shape (B, C, H, W).
            orig_img (np.ndarray): The original unprocessed image as a numpy array.
            img_path (str): The path to the original image file.

        Returns:
            (MDEResults): The result object containing the original image, image path, class names, bounding boxes, and
                depth values.
        """
        # Scale bounding boxes to original image dimensions
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

        # For MDE, pred should have 7 columns: [x1, y1, x2, y2, conf, cls, depth]
        # The depth value is already in the 7th column (index 6)
        return MDEResults(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :7])
