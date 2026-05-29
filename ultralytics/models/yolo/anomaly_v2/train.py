# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 trainer.

Thin extension of ``DetectionTrainer``. The only differences:
  * ``get_model`` returns a ``YOLOAnomalyV2Model`` instead of a plain ``DetectionModel``.
  * ``get_validator`` returns an ``AnomalyV2Validator`` that runs val twice
    (mask-on and mask-off) — see ``val.py``.

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
from ultralytics.utils import RANK


class AnomalyV2Trainer(DetectionTrainer):
    """Trainer for YOLOAnomalyV2 (Phase 0)."""

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
        return yolo.anomaly_v2.AnomalyV2Validator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
