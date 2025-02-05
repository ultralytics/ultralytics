# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
from ultralytics.utils.postprocess_utils import decode_bbox, decode_kpts, separate_outputs_decode


class PosePredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a pose model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.pose import PosePredictor

        args = dict(model="yolo11n-pose.pt", source=ASSETS)
        predictor = PosePredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes PosePredictor, sets task to 'pose' and logs a warning for using 'mps' as device."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose"
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def postprocess(self, preds, img, orig_imgs):
        """Return detection results for a given input image or list of images."""
        if self.separate_outputs:  # Quant friendly export with separated outputs
            pred_order, nkpt = separate_outputs_decode(preds, self.args.task)
            pred_decoded = decode_bbox(pred_order, img.shape, self.device)
            nc = pred_decoded.shape[1] - 4
            kpt_shape = (nkpt.shape[-1] // 3, 3)
            kpts_decoded = decode_kpts(pred_order,
                                       img.shape,
                                       torch.permute(nkpt, (0, 2, 1)),
                                       kpt_shape,
                                       self.device,
                                       bs=1)
            pred_order = torch.cat([pred_decoded, kpts_decoded], 1)
            preds = ops.non_max_suppression(
                pred_order,
                self.args.conf,
                self.args.iou,
                agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det,
                classes=self.args.classes,
                nc=nc
            )
        else:
            preds = ops.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det,
                classes=self.args.classes,
                nc=len(self.model.names),
            )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            if self.separate_outputs:
                pred_kpts = pred[:, 6:].view(len(pred), *kpt_shape) if len(pred) else pred[:, 6:]
            else:
                pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=pred_kpts)
            )
        return results

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Constructs the result object from the prediction.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and keypoints.
            img (torch.Tensor): The image after preprocessing.
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.
        Returns:
            (Results): The result object containing the original image, image path, class names, bounding boxes, and keypoints.
        """

        result = super().construct_result(pred, img, orig_img, img_path)
        if self.separate_outputs:
            pred_kpts = pred[:, 6:].view(len(pred), *kpt_shape) if len(pred) else pred[:, 6:]
        else:
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        result.update(keypoints=pred_kpts)
        return result