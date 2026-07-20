# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly trainer.

Thin extension of ``DetectionTrainer``. The differences:
  * ``get_model`` returns a ``YOLOAnomalyModel`` instead of a plain ``DetectionModel``.
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
from ultralytics.nn.tasks import YOLOAnomalyModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model


class AnomalyTrainer(DetectionTrainer):
    """Trainer for YOLOAnomaly."""

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
            # (bboxes still present). copy() so the persistent self.args stays task="detect".
            args = copy(self.args)
            args.task = "segment"
            gs = max(int(unwrap_model(self.model).stride.max()), 32)
            dataset = build_yolo_dataset(args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
        else:
            dataset = super().build_dataset(img_path, mode=mode, batch=batch)

        # Attach the prior-mask builder only to the training transform pipeline.
        if mode == "train":
            v2_cfg = getattr(unwrap_model(self.model), "yaml", {}).get("anomaly", {})
            transform = LoadAnomalyPriorMask(v2_cfg, mode=mode)
            if isinstance(dataset, YOLOConcatDataset):
                for d in dataset.datasets:
                    d.transforms.append(transform)
            else:
                dataset.transforms.append(transform)
        return dataset

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Build the standard optimizer, then move fusion params into a ``lr * fusion_lr_scale`` group.

        ``anomaly.fusion_lr_scale`` (default 1.0) multiplies the LR of every ``heatmap_bias_fusion``
        parameter (the fusion module lives on the AnomalyDetect head). Splitting AFTER the standard
        build keeps each moved param's group hyperparameters (weight_decay / momentum / use_muon for
        MuSGD); the scheduler and warmup both scale from the group's own lr, so the ratio holds
        across the whole schedule. No-op when scale == 1.0.
        """
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        m = unwrap_model(model)
        scale = float((getattr(m, "yaml", {}) or {}).get("anomaly", {}).get("fusion_lr_scale", 1.0))
        fusion = getattr(m.model[-1], "heatmap_bias_fusion", None)
        if scale == 1.0 or fusion is None:
            return optimizer
        fusion_ids = {id(p) for p in fusion.parameters()}
        n_moved = 0
        for group in list(optimizer.param_groups):
            moved = [p for p in group["params"] if id(p) in fusion_ids]
            if not moved:
                continue
            group["params"] = [p for p in group["params"] if id(p) not in fusion_ids]
            new_group = {k: v for k, v in group.items() if k != "params"}
            new_group.update(params=moved, lr=group["lr"] * scale)
            optimizer.add_param_group(new_group)
            n_moved += len(moved)
        LOGGER.info(
            f"anomaly: fusion_lr_scale={scale} -> {n_moved} heatmap_bias_fusion params split into "
            f"scaled groups ({len(optimizer.param_groups)} groups total)"
        )
        return optimizer

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Return a ``YOLOAnomalyModel``.

        Args:
            cfg (str, optional): Path to model YAML.
            weights (str, optional): Path to pretrained weights (yolo26m.pt etc.).
            verbose (bool): Verbose info.
        """
        model = YOLOAnomalyModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1, p_drop=self.args.p_drop
        )
        if weights:
            model.load(weights)
        # Warm-start the fusion block from a donor run (anomaly.fusion_load in the model YAML), AFTER
        # the pretrained load so the donor fusion weights win. Skipped on resume: last.pt already
        # carries the trained fusion state, so re-loading the donor would roll it back.
        fusion_load = (model.yaml.get("anomaly", {}) or {}).get("fusion_load")
        if fusion_load and not self.args.resume:
            model.load_fusion_weights(fusion_load)
        return model

    def get_validator(self):
        """Return the anomaly validator."""
        loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        self.loss_names = tuple(loss_names)
        return yolo.anomaly.YOLOAnomalyValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
