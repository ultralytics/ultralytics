# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Interface for Baidu's RT-DETR, a Vision Transformer-based real-time object detector.

RT-DETR offers real-time performance and high accuracy, excelling in accelerated backends like CUDA with TensorRT.
It features an efficient hybrid encoder and IoU-aware query selection for enhanced detection accuracy.

References:
    https://arxiv.org/pdf/2304.08069.pdf
"""

from pathlib import Path
from typing import Any

import torch

from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import ops
from ultralytics.utils.torch_utils import TORCH_1_11

from .deim import (
    RTDETRDEIMOBBTrainer,
    RTDETRDEIMOBBValidator,
    RTDETRDEIMPoseTrainer,
    RTDETRDEIMPoseValidator,
    RTDETRDEIMSegmentTrainer,
    RTDETRDEIMSegmentValidator,
    RTDETRDEIMTrainer,
    RTDETRDEIMTrainerV2,
    RTDETRDEIMValidator,
)
from .predict import RTDETRPredictor
from .train import RTDETRTrainer
from .val import RTDETRValidator


class RTDETR(Model):
    """Interface for Baidu's RT-DETR model, a Vision Transformer-based real-time object detector.

    This model provides real-time performance with high accuracy. It supports efficient hybrid encoding, IoU-aware query
    selection, and adaptable inference speed.

    Attributes:
        model (str): Path to the pre-trained model.

    Methods:
        task_map: Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

    Examples:
        Initialize RT-DETR with a pre-trained model
        >>> from ultralytics import RTDETR
        >>> model = RTDETR("rtdetr-l.pt")
        >>> results = model("image.jpg")
    """

    def __init__(self, model: str = "rtdetr-l.pt", task: str | None = None) -> None:
        """Initialize the RT-DETR model with the given pre-trained model file.

        Args:
            model (str): Path to the pre-trained model. Supports .pt, .yaml, and .yml formats.
            task (str, optional): Task of the model. If None, inferred from the YAML head (defaults to 'detect').
        """
        assert TORCH_1_11, "RTDETR requires torch>=1.11"
        if task is None and isinstance(model, (str, Path)) and str(model).endswith((".yaml", ".yml")):
            from ultralytics.nn.tasks import yaml_model_load

            head_name = str(yaml_model_load(model)["head"][-1][-2]).lower()
            for candidate in ("segment", "pose", "obb"):
                if candidate in head_name:
                    task = candidate
                    break
        super().__init__(model=model, task=task or "detect")

    @property
    def task_map(self) -> dict:
        """Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

        Returns:
            (dict): A dictionary mapping task names to Ultralytics task classes for the RT-DETR model.
        """
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRTrainer,
                "model": RTDETRDetectionModel,
            }
        }


class RTDETRDEIMPredictor(RTDETRPredictor):
    """Predictor that applies ImageNet normalization when ``rtdetr_input_normalize`` is set."""

    @staticmethod
    def _normalize_input(img: torch.Tensor) -> torch.Tensor:
        mean = img.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = img.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return (img - mean) / std

    def preprocess(self, im):
        im = super().preprocess(im)
        if getattr(self.args, "rtdetr_input_normalize", False):
            im = self._normalize_input(im)
        return im


class RTDETRDEIMSegmentPredictor(RTDETRDEIMPredictor):
    """DEIM predictor for instance segmentation models built on DeimSegmentDecoder.

    Splits mask coefficients from the decoder top-k rows, assembles masks from the proto, scales boxes to the
    original image, and drops detections with empty masks (mirrors SegmentationPredictor).
    """

    def postprocess(self, preds, img, orig_imgs):
        """Postprocess predictions into Results with boxes and masks.

        Args:
            preds (list | tuple): Model predictions `((y, proto), x)` where `y` has shape (bs, num_queries, 6 + nm)
                with format [cx, cy, w, h, score, class, mc...] (normalized cxcywh).
            img (torch.Tensor): Processed input images with shape (N, 3, H, W).
            orig_imgs (list | torch.Tensor): Original, unprocessed images.

        Returns:
            (list[Results]): A list of Results objects containing boxes and masks.
        """
        protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        y = preds[0][0] if isinstance(preds[0], tuple) else preds[0]
        bboxes, scores, labels, coeffs = y.split((4, 1, 1, protos.shape[1]), dim=-1)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []
        for bbox, score, label, coeff, proto, orig_img, img_path in zip(
            bboxes, scores, labels, coeffs, protos, orig_imgs, self.batch[0]
        ):
            idx = score.squeeze(-1) > self.args.conf
            if self.args.classes is not None:
                idx = (label.squeeze(-1) == torch.tensor(self.args.classes, device=label.device)).any(1) & idx
            bbox, coeff = ops.xywh2xyxy(bbox[idx]), coeff[idx]
            bbox[:, [0, 2]] *= img.shape[3]  # normalized -> model input pixels
            bbox[:, [1, 3]] *= img.shape[2]
            pred = torch.cat([bbox, score[idx], label[idx]], dim=-1)
            if pred.shape[0] == 0:  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                pred[:, [0, 2]] *= orig_img.shape[1] / img.shape[3]  # model input -> original image
                pred[:, [1, 3]] *= orig_img.shape[0] / img.shape[2]
                masks = ops.process_mask_native(proto, coeff, pred[:, :4], orig_img.shape[:2])  # NHW
            else:
                masks = ops.process_mask(proto, coeff, pred[:, :4], img.shape[2:], upsample=True)  # NHW
                pred[:, [0, 2]] *= orig_img.shape[1] / img.shape[3]  # model input -> original image
                pred[:, [1, 3]] *= orig_img.shape[0] / img.shape[2]
            if masks is not None:
                keep = masks.amax((-2, -1)) > 0  # only keep predictions with masks
                if not all(keep):  # most predictions have masks
                    pred, masks = pred[keep], masks[keep]  # indexing is slow
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred, masks=masks))
        return results


class RTDETRDEIMPosePredictor(RTDETRDEIMPredictor):
    """DEIM predictor for pose estimation models built on DeimPoseDecoder.

    Splits keypoints from the decoder top-k rows and scales boxes and keypoint xy per-axis (scale-fill convention)
    to the original image, mirroring RTDETRDEIMSegmentPredictor.
    """

    def postprocess(self, preds, img, orig_imgs):
        """Postprocess predictions into Results with boxes and keypoints.

        Args:
            preds (list | tuple): Model predictions `(y, x)` where `y` has shape (bs, num_queries, 6 + nk) with
                format [cx, cy, w, h, score, class, kpts...] (keypoint xy normalized).
            img (torch.Tensor): Processed input images with shape (N, 3, H, W).
            orig_imgs (list | torch.Tensor): Original, unprocessed images.

        Returns:
            (list[Results]): A list of Results objects containing boxes and keypoints.
        """
        y = preds[0][0] if isinstance(preds[0], tuple) else preds[0]
        bboxes, scores, labels, kpts = y.split((4, 1, 1, y.shape[-1] - 6), dim=-1)
        head = self.model.model.model[-1] if isinstance(getattr(self.model, "model", None), torch.nn.Module) else None
        kpt_shape = getattr(head, "kpt_shape", [17, 3])

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []
        for bbox, score, label, kpt, orig_img, img_path in zip(bboxes, scores, labels, kpts, orig_imgs, self.batch[0]):
            idx = score.squeeze(-1) > self.args.conf
            if self.args.classes is not None:
                idx = (label.squeeze(-1) == torch.tensor(self.args.classes, device=label.device)).any(1) & idx
            bbox = ops.xywh2xyxy(bbox[idx])
            bbox[:, [0, 2]] *= orig_img.shape[1]  # normalized -> original image (scale-fill)
            bbox[:, [1, 3]] *= orig_img.shape[0]
            pred = torch.cat([bbox, score[idx], label[idx]], dim=-1)
            kpt = kpt[idx].view(-1, *kpt_shape).clone()
            kpt[..., 0] *= orig_img.shape[1]  # normalized -> original image (scale-fill)
            kpt[..., 1] *= orig_img.shape[0]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred, keypoints=kpt))
        return results


class RTDETRDEIMOBBPredictor(RTDETRDEIMPredictor):
    """DEIM predictor for oriented bounding box models built on DeimOBBDecoder.

    Splits rotation angles from the decoder top-k rows and scales the xywh part per-axis (scale-fill convention) to
    the original image, leaving angles untouched, mirroring OBBPredictor.
    """

    def postprocess(self, preds, img, orig_imgs):
        """Postprocess predictions into Results with oriented bounding boxes.

        Args:
            preds (list | tuple): Model predictions `(y, x)` where `y` has shape (bs, num_queries, 7) with format
                [cx, cy, w, h, score, class, angle] (xywh normalized, raw angle).
            img (torch.Tensor): Processed input images with shape (N, 3, H, W).
            orig_imgs (list | torch.Tensor): Original, unprocessed images.

        Returns:
            (list[Results]): A list of Results objects containing oriented bounding boxes.
        """
        y = preds[0][0] if isinstance(preds[0], tuple) else preds[0]
        bboxes, scores, labels, angles = y.split((4, 1, 1, 1), dim=-1)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []
        for bbox, score, label, angle, orig_img, img_path in zip(
            bboxes, scores, labels, angles, orig_imgs, self.batch[0]
        ):
            idx = score.squeeze(-1) > self.args.conf
            if self.args.classes is not None:
                idx = (label.squeeze(-1) == torch.tensor(self.args.classes, device=label.device)).any(1) & idx
            bbox = bbox[idx].clone()
            bbox[:, [0, 2]] *= orig_img.shape[1]  # normalized -> original image (scale-fill)
            bbox[:, [1, 3]] *= orig_img.shape[0]
            obb = torch.cat([bbox, angle[idx], score[idx], label[idx]], dim=-1)  # [xywhr, conf, cls]
            results.append(Results(orig_img, path=img_path, names=self.model.names, obb=obb))
        return results


class RTDETRDEIM(RTDETR):
    """RT-DETR interface that routes training/validation through isolated DEIM classes."""

    _EXTRA_CKPT_ARGS = {"rtdetr_input_normalize"}

    @staticmethod
    def _reset_ckpt_args(args: dict[str, Any]) -> dict[str, Any]:
        include = {"imgsz", "data", "task", "single_cls"} | RTDETRDEIM._EXTRA_CKPT_ARGS
        return {k: v for k, v in args.items() if k in include}

    @property
    def task_map(self) -> dict:
        """Return a task map that uses DEIM-specific trainer/validator implementations."""
        return {
            "detect": {
                "predictor": RTDETRDEIMPredictor,
                "validator": RTDETRDEIMValidator,
                "trainer": RTDETRDEIMTrainer,
                "model": RTDETRDetectionModel,
            },
            "segment": {
                "predictor": RTDETRDEIMSegmentPredictor,
                "validator": RTDETRDEIMSegmentValidator,
                "trainer": RTDETRDEIMSegmentTrainer,
                "model": RTDETRDetectionModel,
            },
            "pose": {
                "predictor": RTDETRDEIMPosePredictor,
                "validator": RTDETRDEIMPoseValidator,
                "trainer": RTDETRDEIMPoseTrainer,
                "model": RTDETRDetectionModel,
            },
            "obb": {
                "predictor": RTDETRDEIMOBBPredictor,
                "validator": RTDETRDEIMOBBValidator,
                "trainer": RTDETRDEIMOBBTrainer,
                "model": RTDETRDetectionModel,
            },
        }


class RTDETRDEIMv2(RTDETR):
    """RT-DETR interface that routes training through DEIM v2 trainer with stage-switch EMA refresh."""

    @staticmethod
    def _reset_ckpt_args(args: dict[str, Any]) -> dict[str, Any]:
        include = {"imgsz", "data", "task", "single_cls"} | RTDETRDEIM._EXTRA_CKPT_ARGS
        return {k: v for k, v in args.items() if k in include}

    @property
    def task_map(self) -> dict:
        """Return a task map that uses DEIM v2 trainer implementation."""
        return {
            "detect": {
                "predictor": RTDETRDEIMPredictor,
                "validator": RTDETRDEIMValidator,
                "trainer": RTDETRDEIMTrainerV2,
                "model": RTDETRDetectionModel,
            },
            "segment": {
                "predictor": RTDETRDEIMSegmentPredictor,
                "validator": RTDETRDEIMSegmentValidator,
                "trainer": RTDETRDEIMSegmentTrainer,
                "model": RTDETRDetectionModel,
            },
            "pose": {
                "predictor": RTDETRDEIMPosePredictor,
                "validator": RTDETRDEIMPoseValidator,
                "trainer": RTDETRDEIMPoseTrainer,
                "model": RTDETRDetectionModel,
            },
            "obb": {
                "predictor": RTDETRDEIMOBBPredictor,
                "validator": RTDETRDEIMOBBValidator,
                "trainer": RTDETRDEIMOBBTrainer,
                "model": RTDETRDetectionModel,
            },
        }
