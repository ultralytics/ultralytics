# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Instance-segmentation runtime support for the DEIM query mask decoder."""

from __future__ import annotations

from copy import copy

import torch
import torch.nn.functional as F

from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils import DEFAULT_CFG, RANK, colorstr, ops
from ultralytics.utils.torch_utils import unwrap_model

from .deim import RTDETRDEIMDataset, RTDETRDEIMTrainer
from .predict import RTDETRPredictor
from .val import RTDETRValidator

__all__ = (
    "RTDETRDEIMSegmentationPredictor",
    "RTDETRDEIMSegmentationTrainer",
    "RTDETRDEIMSegmentationValidator",
)


def _unwrap_segment_predictions(preds):
    """Return `(detections, mask_logits)` from PyTorch or exported DEIM segment outputs."""
    primary = (
        preds[0]
        if isinstance(preds, (list, tuple)) and len(preds) == 2 and isinstance(preds[0], tuple)
        else preds
    )
    if not isinstance(primary, (list, tuple)) or len(primary) != 2:
        raise ValueError("DeimSegmentDecoder must return `(detections, mask_logits)` during inference.")
    return primary


class RTDETRDEIMSegmentationPredictor(RTDETRPredictor):
    """Create instance-mask Results from direct DEIM query mask logits."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.task = "segment"

    def setup_model(self, model, verbose=True):
        """Set up inference and apply an optional decoder early-exit index."""
        super().setup_model(model, verbose=verbose)
        RTDETRValidator.apply_eval_idx(self.model, getattr(self.args, "eval_idx", None))

    def preprocess(self, im):
        """Apply standard RT-DETR preprocessing and optional ImageNet normalization."""
        im = super().preprocess(im)
        if getattr(self.args, "rtdetr_input_normalize", False):
            mean = im.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
            std = im.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
            im = (im - mean) / std
        return im

    def postprocess(self, preds, img, orig_imgs):
        """Threshold detections and resize their aligned direct masks to each original image."""
        detections, mask_logits = _unwrap_segment_predictions(preds)
        bboxes, scores, labels = detections.split((4, 1, 1), dim=-1)
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []
        for bbox, score, label, masks, orig_img, img_path in zip(
            bboxes, scores, labels, mask_logits, orig_imgs, self.batch[0]
        ):
            keep = score.squeeze(-1) > self.args.conf
            if self.args.classes is not None:
                classes = torch.tensor(self.args.classes, device=label.device)
                keep &= torch.isin(label.squeeze(-1), classes)
            pred = torch.cat((ops.xywh2xyxy(bbox), score, label), dim=-1)[keep]
            masks = masks[keep]
            oh, ow = orig_img.shape[:2]
            pred[..., [0, 2]] *= ow
            pred[..., [1, 3]] *= oh
            if masks.numel():
                masks = F.interpolate(masks.unsqueeze(1), (oh, ow), mode="bilinear", align_corners=False).squeeze(1)
                masks = masks.sigmoid() > 0.5
            else:
                masks = None
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred, masks=masks))
        return results


class RTDETRDEIMSegmentationValidator(SegmentationValidator):
    """Segmentation validator for normalized RT-DETR boxes and direct query masks."""

    _pop_batch_flag = staticmethod(RTDETRValidator._pop_batch_flag)
    _normalize_input = staticmethod(RTDETRValidator._normalize_input)

    def __call__(self, trainer=None, model=None):
        if trainer is not None:
            self._val_epoch = int(trainer.epoch)
            self._val_training_progress = min(max(trainer.epoch / max(trainer.epochs - 1, 1), 0.0), 1.0)
        return super().__call__(trainer=trainer, model=model)

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize box/mask metrics and apply an optional decoder early-exit index."""
        super().init_metrics(model)
        RTDETRValidator.apply_eval_idx(model, getattr(self.args, "eval_idx", None))

    def preprocess(self, batch):
        """Use RT-DETR image preprocessing and retain floating instance masks."""
        batch = RTDETRValidator.preprocess(self, batch)
        batch["masks"] = batch["masks"].float()
        if self.training:
            batch["epoch"] = int(self._val_epoch)
            batch["training_progress"] = float(self._val_training_progress)
        return batch

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build a mask-returning DEIM dataset."""
        return RTDETRDEIMDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
            task="segment",
        )

    def postprocess(self, preds):
        """Convert normalized query detections and direct masks to metric inputs."""
        detections, mask_logits = _unwrap_segment_predictions(preds)
        bboxes, scores, labels = detections.split((4, 1, 1), dim=-1)
        bboxes = ops.xywh2xyxy(bboxes) * self.args.imgsz
        outputs = []
        for boxes, score, label, masks in zip(bboxes, scores.squeeze(-1), labels.squeeze(-1), mask_logits):
            keep = score > self.args.conf
            masks = masks[keep]
            if self.process is ops.process_mask_native and masks.numel():
                masks = F.interpolate(
                    masks.unsqueeze(1), (self.args.imgsz, self.args.imgsz), mode="bilinear", align_corners=False
                ).squeeze(1)
            outputs.append(
                {
                    "bboxes": boxes[keep],
                    "conf": score[keep],
                    "cls": label[keep],
                    "masks": (masks.sigmoid() > 0.5).byte(),
                }
            )
        return outputs


class RTDETRDEIMSegmentationTrainer(RTDETRDEIMTrainer):
    """DEIM trainer that loads polygon labels and reports box plus mask losses."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = {} if overrides is None else dict(overrides)
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Reuse the RT-DETR model class; its loss path activates when the head emits masks."""
        return super().get_model(cfg, weights, verbose=verbose and RANK == -1)

    def get_validator(self):
        """Return the direct-mask validator and configure the displayed DEIM losses."""
        loss_names = ["giou_loss", "cls_loss", "l1_loss"]
        loss_gain = self.model_yaml.get("loss", {}).get("loss_gain", {})
        for key, name in (("fgl", "fgl_loss"), ("ddf", "ddf_loss"), ("rank", "rank_loss")):
            if loss_gain.get(key, 0) > 0:
                loss_names.append(name)
        if loss_gain.get("mask", 0) > 0:
            loss_names.append("mask_loss")
        if loss_gain.get("dice", 0) > 0:
            loss_names.append("dice_loss")
        model = unwrap_model(self.model)
        if getattr(model.model[-1], "one_to_many_groups", 0) > 0:
            loss_names.extend(("giou_o2m", "cls_o2m", "l1_o2m"))
        self.loss_names = tuple(loss_names)
        return RTDETRDEIMSegmentationValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
