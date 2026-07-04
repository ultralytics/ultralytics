# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 trainer.

Thin extension of ``DetectionTrainer``. The differences:
  * ``get_model`` returns a ``YOLOAnomalyV2Model`` instead of a plain ``DetectionModel``.
  * ``get_validator`` returns a ``YOLOAnomalyValidator``.

Everything else (dataset, dataloader, augmentation, loss aggregation, plot,
auto_batch, etc.) is inherited from ``DetectionTrainer`` unchanged.
"""

from __future__ import annotations

import math
from copy import copy, deepcopy
import torch

from ultralytics.data import YOLOConcatDataset, build_yolo_dataset
from ultralytics.data.augment import LoadAnomalyPriorMask
from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.head import AnomalyMCDetect
from ultralytics.nn.tasks import YOLOAnomalyV2Model
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model

from .benchmark import resolve_mvtec_root, run_mvtec_ood_eval


class AnomalyV2Trainer(DetectionTrainer):
    """Trainer for YOLOAnomalyV2."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        # NOTE: _mvtec_ood_eval is NOT a callback — it runs inside validate() so best.pt
        # selection sees the current epoch's OOD score (no one-epoch lag).
        self._ood_best_fitness = None

    def validate(self):
        """Run validation; when OOD eval is enabled, use OOD heatmap mAP10 as fitness.

        The OOD eval runs here (not as an ``on_fit_epoch_end`` callback) on the current epoch's
        EMA, so best.pt selection sees this epoch's score with no one-epoch lag. Fitness stays on
        the OOD scale every epoch (``0.0`` when the eval can't run). The base ``validate`` bumps
        ``best_fitness`` on the in-domain scale, so we recompute it on the OOD scale
        (``_ood_best_fitness``) and overwrite — ``save_model`` saves best.pt when
        ``best_fitness == fitness``, i.e. on each new OOD max.
        """
        metrics, fitness = super().validate()
        v2_cfg = getattr(unwrap_model(self.model), "yaml", {}).get("anomaly_v2", {})
        if int(v2_cfg.get("test_val_freq", 3)) > 0 and metrics is not None:
            self._ood_heatmap_map10 = None
            AnomalyV2Trainer._mvtec_ood_eval(self)
            fitness = float(self._ood_heatmap_map10 or 0.0)
            metrics["fitness"] = fitness
            self._ood_best_fitness = fitness if self._ood_best_fitness is None else max(self._ood_best_fitness, fitness)
            self.best_fitness = self._ood_best_fitness
        return metrics, fitness

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
        model = unwrap_model(self.model)
        if isinstance(model.model[-1], AnomalyMCDetect):
            loss_names = ["box_loss", "anom_loss", "dfl_loss", "type_loss"]
        else:
            loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        self.loss_names = tuple(loss_names)
        return yolo.anomaly_v2.YOLOAnomalyValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    @staticmethod
    def _mvtec_ood_eval(trainer: "AnomalyV2Trainer") -> None:
        """Periodic MVTec cross-dataset OOD eval, rank 0 only; called from validate().

        Configured via the model YAML ``anomaly_v2`` block: ``test_val_freq`` (every N epochs;
        default 3, set 0 to disable). Bank-build and heatmap post-processing knobs are baked into
        the model; only the MVTec root and test categories come from the YAML.

        ``test_heatmap_prior`` is forced on (best.pt fitness is ``test_metrics(heatmap_prior)/mAP10``);
        ``test_none_prior`` defaults off.

        Runs on a deepcopy of the EMA so val-time fuse()/bank-build never corrupt the live EMA.
        """
        if RANK not in (-1, 0) or trainer.ema is None:
            return
        v2_cfg = getattr(unwrap_model(trainer.model), "yaml", {}).get("anomaly_v2", {})
        freq = int(v2_cfg.get("test_val_freq", 3))
        if freq <= 0 or (trainer.epoch + 1) % freq != 0:
            return
        root = resolve_mvtec_root(v2_cfg.get("test_root"))
        if root is None:
            LOGGER.warning(
                "MVTec test: dataset root not found (set MVTEC_ROOT or anomaly_v2.test_root); skipping test eval."
            )
            return

        batch = int(v2_cfg.get("test_batch", 8))

        if not bool(v2_cfg.get("test_heatmap_prior", True)):
            LOGGER.warning("anomaly_v2: test_heatmap_prior feeds fitness and cannot be disabled; forcing on.")
        modes = []
        if bool(v2_cfg.get("test_none_prior", False)):
            modes.append("none")
        modes.append("heatmap")
        modes = tuple(modes)

        ema_eval = deepcopy(trainer.ema.ema).eval()
        trainer._ood_heatmap_map10 = None
        try:
            with torch.no_grad():
                rows = run_mvtec_ood_eval(
                    ema_eval,
                    root,
                    categories=v2_cfg.get("test_categories"),
                    modes=modes,
                    imgsz=640,
                    batch=batch,
                    workers=trainer.args.workers,
                    device=trainer.device,
                    save_dir=trainer.save_dir,
                    epoch=trainer.epoch + 1,
                    e2e=False,
                    iou=0.1,
                )
            AnomalyV2Trainer._log_ood_wandb(trainer, rows)
            for r in rows:
                if r["category"] == "AVERAGE" and r["mode"] == "heatmap":
                    map10 = r.get("mAP10", math.nan)
                    if not math.isnan(map10):
                        trainer._ood_heatmap_map10 = float(map10)
                    break
        except Exception as e:
            LOGGER.warning(f"MVTec OOD eval failed at epoch {trainer.epoch + 1}: {type(e).__name__}: {e}")
        finally:
            del ema_eval

    @staticmethod
    def _log_ood_wandb(trainer: "AnomalyV2Trainer", rows: list) -> None:
        """Push OOD metrics to wandb grouped by mode (test_metrics(none_prior/heatmap_prior))."""
        if not rows:
            return
        try:
            from ultralytics.utils.callbacks.wb import wb
        except Exception:
            wb = None
        if wb is None or getattr(wb, "run", None) is None:
            return

        keys = ("mAP10", "mAP25", "mAP50", "mAP50_95", "P", "R")
        mode_to_group = {
            "none": "test_metrics(none_prior)",
            "heatmap": "test_metrics(heatmap_prior)",
        }

        log = {}
        for mode, group_name in mode_to_group.items():
            avg_row = next((r for r in rows if r["category"] == "AVERAGE" and r["mode"] == mode), None)
            if avg_row is not None:
                for k in keys:
                    if avg_row.get(k) is not None and not (
                        isinstance(avg_row.get(k), float) and math.isnan(avg_row[k])
                    ):
                        log[f"{group_name}/{k}"] = avg_row[k]

        if log:
            try:
                wb.run.log(log, step=trainer.epoch + 1)
            except Exception as e:
                LOGGER.warning(f"MVTec OOD: wandb log failed: {type(e).__name__}: {e}")
