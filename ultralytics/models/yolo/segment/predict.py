# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class SegmentationPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
    prediction results.

    Attributes:
        args (Dict): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO segmentation model.
        batch (List): Current batch of images being processed.

    Methods:
        postprocess: Applies non-max suppression and processes detections.
        construct_results: Constructs a list of result objects from predictions.
        construct_result: Constructs a single result object from a prediction.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.segment import SegmentationPredictor
        >>> args = dict(model="yolo11n-seg.pt", source=ASSETS)
        >>> predictor = SegmentationPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the SegmentationPredictor with configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        """Apply non-max suppression and process detections for each image in the input batch."""
        # Extract protos - tuple if PyTorch model or array if exported
        protos = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
        return super().postprocess(preds[0], img, orig_imgs, protos=protos)

    def construct_results(self, preds, img, orig_imgs, protos):
        """
        Construct a list of result objects from the predictions.

        Args:
            preds (List[torch.Tensor]): List of predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_imgs (List[np.ndarray]): List of original images before preprocessing.
            protos (List[torch.Tensor]): List of prototype masks.

        Returns:
            (List[Results]): List of result objects containing the original images, image paths, class names,
                bounding boxes, and masks.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path, proto)
            for pred, orig_img, img_path, proto in zip(preds, orig_imgs, self.batch[0], protos)
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto):
        """
        Construct a single result object from the prediction.

        Args:
            pred (np.ndarray): The predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.
            proto (torch.Tensor): The prototype masks.

        Returns:
            (Results): Result object containing the original image, image path, class names, bounding boxes, and masks.
        """
        if not len(pred):  # save empty boxes
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
        else:
            masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks
            pred, masks = pred[keep], masks[keep]
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
