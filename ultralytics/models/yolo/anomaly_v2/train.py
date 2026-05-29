# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 trainer.

Thin extension of ``DetectionTrainer``. The differences:
  * ``get_model`` returns a ``YOLOAnomalyV2Model`` instead of a plain ``DetectionModel``.
  * ``get_validator`` returns an ``AnomalyV2Validator`` that runs val twice
    (mask-on and mask-off) — see ``val.py``.
  * When the model has a SegBranch (v2.2), an alpha curriculum anneals the fusion
    prior from the GT mask to the predicted heatmap, and ``seg_loss`` is reported.

Everything else (dataset, dataloader, augmentation, loss aggregation, plot,
auto_batch, etc.) is inherited from ``DetectionTrainer`` unchanged. Mask
rendering happens inside ``YOLOAnomalyV2Model.loss`` from ``batch["bboxes"]``;
no special handling needed at the trainer level.
"""

from __future__ import annotations

from copy import copy

from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import YOLOAnomalyV2Model
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import unwrap_model


class AnomalyV2Trainer(DetectionTrainer):
    """Trainer for YOLOAnomalyV2 (Phase 0 + v2.2 SegBranch)."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.add_callback("on_train_epoch_start", AnomalyV2Trainer._update_seg_alpha)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Return a YOLOAnomalyV2Model.

        Args:
            cfg (str, optional): Path to model YAML.
            weights (str, optional): Path to pretrained weights (yolo26m.pt etc.).
            verbose (bool): Verbose info.
        """
        model = YOLOAnomalyV2Model(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return an AnomalyV2Validator which runs val twice (mask-on / mask-off)."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        if getattr(unwrap_model(self.model), "seg_branch", None) is not None:
            self.loss_names = "box_loss", "cls_loss", "dfl_loss", "seg_loss"
        return yolo.anomaly_v2.AnomalyV2Validator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    @staticmethod
    def _update_seg_alpha(trainer: "AnomalyV2Trainer") -> None:
        """Anneal the GT->prediction blend: alpha = 1 at epoch 0, linear to 0 when mosaic closes.

        Past the mosaic-close epoch (``epochs - close_mosaic``) the fusion runs purely on the
        SegBranch prediction, so the mask-on validation pass measures real prior-free inference.
        The value is mirrored onto the EMA model, which validation actually runs on.
        """
        model = unwrap_model(trainer.model)
        if getattr(model, "seg_branch", None) is None:
            return
        end_epoch = max(1, trainer.epochs - trainer.args.close_mosaic)
        alpha = max(0.0, 1.0 - trainer.epoch / end_epoch)
        model.seg_alpha = alpha
        if trainer.ema is not None:
            unwrap_model(trainer.ema.ema).seg_alpha = alpha
