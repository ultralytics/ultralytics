# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 trainer.

Thin extension of ``DetectionTrainer``. The differences:
  * ``get_model`` returns a ``YOLOAnomalyV2Model`` instead of a plain ``DetectionModel``.
  * ``get_validator`` returns a ``YOLOAnomalyValidator``.

Everything else (dataset, dataloader, augmentation, loss aggregation, plot,
auto_batch, etc.) is inherited from ``DetectionTrainer`` unchanged.
"""

from __future__ import annotations

from copy import copy

from ultralytics.data import YOLOConcatDataset, build_yolo_dataset
from ultralytics.data.augment import LoadAnomalyPriorMask
from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import YOLOAnomalyV2Model
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import unwrap_model


class AnomalyV2Trainer(DetectionTrainer):
    """Trainer for YOLOAnomalyV2."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)

    def _polygon_prior_active(self) -> bool:
        """True when the fusion prior uses v6 polygon masks (seg_target_polygon)."""
        return bool(getattr(unwrap_model(self.model), "seg_target_polygon", False))

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build the dataset, forcing polygon-mask loading when polygon-prior mode is on.

        For training only, appends ``LoadAnomalyPriorMask`` so the collated batch contains
        ``prior_mask`` for the model to consume directly. Validation uses the heatmap prior
        (or passthrough) instead of an explicit mask.
        """
        if self._polygon_prior_active() and mode == "train":
            # task="segment" flips YOLODataset.use_segments -> Format(return_mask=True) ->
            # batch["masks"] (overlap-union instance map). Detection head/loss unaffected
            # (bboxes still present). copy() so the persistent self.args stays task="anomaly_v2".
            args = copy(self.args)
            args.task = "segment"
            gs = max(int(unwrap_model(self.model).stride.max()), 32)
            dataset = build_yolo_dataset(args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
        else:
            dataset = super().build_dataset(img_path, mode=mode, batch=batch)

        # Attach the prior-mask builder only to the training transform pipeline.
        if mode == "train":
            v2_cfg = getattr(unwrap_model(self.model), "yaml", {}).get("anomaly_v2", {})
            transform = LoadAnomalyPriorMask(v2_cfg, mode=mode)
            if isinstance(dataset, YOLOConcatDataset):
                for d in dataset.datasets:
                    d.transforms.append(transform)
            else:
                dataset.transforms.append(transform)
        return dataset

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Return a ``YOLOAnomalyV2Model``.

        Args:
            cfg (str, optional): Path to model YAML.
            weights (str, optional): Path to pretrained weights (yolo26m.pt etc.).
            verbose (bool): Verbose info.
        """
        model = YOLOAnomalyV2Model(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return the anomaly validator."""
        loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        self.loss_names = tuple(loss_names)
        return yolo.anomaly_v2.YOLOAnomalyValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
