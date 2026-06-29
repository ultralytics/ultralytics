# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 trainer.

Thin extension of ``DetectionTrainer``. The differences:
  * ``get_model`` returns a ``YOLOAnomalyV2Model`` instead of a plain ``DetectionModel``.
  * ``get_validator`` returns a ``YOLOAnomalyValidator`` / ``YOLOAnomalySegValidator`` that runs val twice
    (mask-on and mask-off) — see ``val.py``.
  * When the model has a SegBranch (v2.2), an alpha curriculum anneals the fusion
    prior from the GT mask to the predicted heatmap, and ``seg_loss`` is reported.

Everything else (dataset, dataloader, augmentation, loss aggregation, plot,
auto_batch, etc.) is inherited from ``DetectionTrainer`` unchanged. Mask
rendering happens inside ``YOLOAnomalyV2Model.loss`` from ``batch["bboxes"]``;
no special handling needed at the trainer level.
"""

from __future__ import annotations

import math
from copy import copy, deepcopy
from pathlib import Path

import torch

from ultralytics.data import build_yolo_dataset
from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.head import AnomalyMCDetect
from ultralytics.nn.tasks import YOLOAnomalyV2Model, YOLOAnomalyV2SegModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, YAML
from ultralytics.utils.torch_utils import unwrap_model

from .val import resolve_mvtec_root, run_mvtec_ood_eval

# Fit YAML keys -> FeatureDiscriminatorScorer kwargs
_SCORER_YAML_KEYS = {
    "scorer_noise_std": "noise_std", "scorer_steps": "steps", "scorer_hidden": "hidden",
    "scorer_n_noise": "n_noise", "scorer_batch": "batch", "scorer_lr": "lr",
    "scorer_noise_mode": "noise_mode",
}


def _resolve_fit_yaml(cfg_path: str, model_yaml_path: str | None = None) -> dict:
    """Load a fit YAML; tries absolute, relative-to-model-YAML, then ``cfg/models/v2/`` as anchor."""
    p = Path(cfg_path)
    if p.is_file():
        return dict(YAML.load(str(p)))
    # Relative to model YAML's directory (when model_yaml_path is a real path)
    if model_yaml_path:
        m = Path(model_yaml_path)
        if m.parent != Path("."):  # skip basename-only; use the cfg/models/v2 anchor instead
            alt = (m.parent / cfg_path).resolve()
            if alt.is_file():
                return dict(YAML.load(str(alt)))
    # Resolve from <repo_root>/ultralytics/cfg/models/v2/ — the canonical model-YAML location
    cfg_dir = Path(__file__).resolve().parents[3] / "cfg" / "models" / "v2"
    alt = (cfg_dir / cfg_path).resolve()
    if alt.is_file():
        return dict(YAML.load(str(alt)))
    LOGGER.warning(f"MVTec OOD: fit YAML not found at {cfg_path}; using defaults.")
    return {}


class AnomalyV2Trainer(DetectionTrainer):
    """Trainer for YOLOAnomalyV2 (Phase 0 + v2.2 SegBranch)."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.add_callback("on_train_epoch_start", AnomalyV2Trainer._update_seg_alpha)
        # NOTE: _mvtec_ood_eval is NOT a callback — it runs inside validate() so best.pt
        # selection sees the current epoch's OOD score (no one-epoch lag).
        self._ood_best_fitness = None  # running max of the OOD-scale fitness (best.pt selection)

    def validate(self):
        """Run validation; when OOD eval is enabled, use OOD heatmap mAP10 (test_metrics(heatmap_prior)) as fitness.

        The OOD eval runs here (not as an ``on_fit_epoch_end`` callback) on the current epoch's
        EMA, so best.pt selection sees this epoch's score with no one-epoch lag. Fitness stays on
        the OOD scale every epoch (``0.0`` when the eval can't run) — it never falls back to the
        in-domain metric, which is a different scale and would freeze best.pt at epoch 0. The base
        ``validate`` bumps ``best_fitness`` on the in-domain scale, so we recompute it on the OOD
        scale (``_ood_best_fitness``) and overwrite — ``save_model`` saves best.pt when
        ``best_fitness == fitness``, i.e. on each new OOD max.
        """
        metrics, fitness = super().validate()
        v2_cfg = getattr(unwrap_model(self.model), "yaml", {}).get("anomaly_v2", {})
        if int(v2_cfg.get("test_val_freq", 3)) > 0 and metrics is not None:
            self._ood_heatmap_map10 = None  # skipped (freq) epochs -> 0.0, not a best.pt candidate
            AnomalyV2Trainer._mvtec_ood_eval(self)  # sets self._ood_heatmap_map10 (+ wandb log)
            fitness = float(self._ood_heatmap_map10 or 0.0)
            metrics["fitness"] = fitness
            self._ood_best_fitness = (
                fitness if self._ood_best_fitness is None else max(self._ood_best_fitness, fitness)
            )
            self.best_fitness = self._ood_best_fitness
        return metrics, fitness

    def _seg_polygon_active(self) -> bool:
        """True when the SegBranch refiner is supervised against v6 polygon masks (seg_target_polygon)."""
        return bool(getattr(unwrap_model(self.model), "seg_target_polygon", False))

    def _instance_seg_active(self) -> bool:
        """True when the detection head is a ``Segment`` (per-instance mask prediction)."""
        if self.model is None:
            return False
        from ultralytics.nn.modules.head import Segment

        return isinstance(unwrap_model(self.model).model[-1], Segment)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Dataset selection: segment (polygon masks for the SegBranch refiner target /
        per-instance seg) > default detection."""
        if self._seg_polygon_active() or self._instance_seg_active():
            # task="segment" flips YOLODataset.use_segments -> Format(return_mask=True) ->
            # batch["masks"] (overlap-union instance map). Detection head/loss unaffected
            # (bboxes still present). copy() so the persistent self.args stays task="anomaly_v2".
            args = copy(self.args)
            args.task = "segment"
            gs = max(int(unwrap_model(self.model).stride.max()), 32)
            return build_yolo_dataset(
                args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs
            )
        return super().build_dataset(img_path, mode=mode, batch=batch)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Return a YOLOAnomalyV2Model or YOLOAnomalyV2SegModel.

        Auto-detects the head type from the YAML: ``Segment`` head → ``YOLOAnomalyV2SegModel``
        (per-instance mask prediction); ``Detect`` head → ``YOLOAnomalyV2Model`` (standard).

        Args:
            cfg (str, optional): Path to model YAML.
            weights (str, optional): Path to pretrained weights (yolo26m.pt etc.).
            verbose (bool): Verbose info.
        """
        # Read the YAML head section to detect Segment vs Detect head.
        model_cls = YOLOAnomalyV2Model
        if cfg and Path(cfg).is_file():
            yaml_dict = YAML.load(cfg)
            for entry in yaml_dict.get("head", []):
                if len(entry) >= 3 and entry[2] == "Segment":
                    model_cls = YOLOAnomalyV2SegModel
                    break
        model = model_cls(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return the anomaly validator (2-pass, legacy GT mask rendering for training).

        A ``Segment`` head routes to ``YOLOAnomalySegValidator`` (box + per-instance mask metrics);
        a ``Detect`` head to ``YOLOAnomalyValidator`` (box metrics). Mirrors ``get_model``'s
        head-based model selection.
        """
        model = unwrap_model(self.model)
        seg_active = self._instance_seg_active()
        if seg_active:
            # v8SegmentationLoss returns [box, seg, cls, dfl, sem] (5 components).
            loss_names = ["box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"]
        elif isinstance(model.model[-1], AnomalyMCDetect):
            # AnomalyMCLoss returns [box, anom, dfl, type] (decoupled detection + type).
            loss_names = ["box_loss", "anom_loss", "dfl_loss", "type_loss"]
        else:
            loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        if getattr(model, "seg_branch", None) is not None:
            # SegBranch heatmap-predictor BCE+Dice term. Stacking it on a per-instance Segment
            # head is not a supported combo (no config does it), so no name-collision handling.
            loss_names.append("seg_loss")
        # QueryFiLM appends 4 aux components in this exact order (see YOLOAnomalyV2Model.loss).
        if getattr(model, "fusion_mode", None) == "queryfilm":
            loss_names += ["qmask_loss", "qobj_loss", "qovl_loss", "qfg_loss"]
        self.loss_names = tuple(loss_names)
        validator_cls = (
            yolo.anomaly_v2.YOLOAnomalySegValidator if seg_active else yolo.anomaly_v2.YOLOAnomalyValidator
        )
        return validator_cls(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args),
            _callbacks=self.callbacks, prior_mode=None,  # legacy GT bboxes -> renderer
        )

    @staticmethod
    def _mvtec_ood_eval(trainer: "AnomalyV2Trainer") -> None:
        """Periodic MVTec cross-dataset OOD eval, rank 0 only; called from validate().

        Configured via the model YAML ``anomaly_v2`` block: ``test_val_freq`` (every N epochs;
        default 3, set 0 to disable). When ``test_fit_cfg`` is set, ``imgsz``, ``heatmap_mode``,
        scorer settings, ``heat_edge`` / ``heat_norm`` are read from that fit YAML (the single source of
        truth for all fit params). Without it, ``test_imgsz`` (default 320) and the default heatmap
        prior with no scorer / no post-processing are used. Per-prior switches ``test_none_prior`` /
        ``test_heatmap_prior`` / ``test_mask_prior`` pick which modes run; ``test_heatmap_prior`` is
        forced on (best.pt fitness is ``test_metrics(heatmap_prior)/mAP10``).

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
            LOGGER.warning("MVTec test: dataset root not found (set MVTEC_ROOT or anomaly_v2."
                           "test_root); skipping test eval.")
            return

        # -- Fit YAML (optional; single source of truth for imgsz / heatmap_mode / scorer / post) --
        fit_cfg_path = v2_cfg.get("test_fit_cfg")
        fit_yaml = {}
        if fit_cfg_path:
            model_yaml = getattr(unwrap_model(trainer.model), "yaml", {}) or {}
            model_yaml_path = model_yaml.get("yaml_file")
            fit_yaml = _resolve_fit_yaml(str(fit_cfg_path), model_yaml_path)

        imgsz = int(fit_yaml.get("imgsz", v2_cfg.get("test_imgsz", 320)))
        batch = int(v2_cfg.get("test_batch", 8))
        bank_size = int(fit_yaml.get("bb_max_bank_size", v2_cfg.get("test_bank_size", 10000)))

        # Per-prior test switches pick which OOD modes run. test_heatmap_prior is forced on because
        # best.pt fitness is test_metrics(heatmap_prior)/mAP10; test_none_prior (mask_off) and
        # test_mask_prior (mask_on) default off (heatmap-only = ~3x faster).
        heatmap_mode = fit_yaml.get("heatmap_mode")
        _MODE_MAP = {"memory_bank": "heatmap", "learned": "heatmap_learned", "fused": "heatmap_fused"}
        heatmap_variant = _MODE_MAP.get(heatmap_mode, "heatmap")
        if not bool(v2_cfg.get("test_heatmap_prior", True)):
            LOGGER.warning("anomaly_v2: test_heatmap_prior feeds fitness and cannot be disabled; forcing on.")
        modes = []
        if bool(v2_cfg.get("test_none_prior", False)):
            modes.append("mask_off")
        modes.append(heatmap_variant)  # test_heatmap_prior — always on (fitness source)
        if bool(v2_cfg.get("test_mask_prior", False)):
            modes.append("mask_on")
        modes = tuple(modes)

        # Scorer kwargs (learned / fused only)
        scorer_kwargs, scorer_fuse = None, "mean"
        if heatmap_mode in ("learned", "fused"):
            scorer_kwargs = {kwk: fit_yaml[yk] for yk, kwk in _SCORER_YAML_KEYS.items() if yk in fit_yaml}
            scorer_kwargs.setdefault("adaptor", fit_yaml.get("scorer_adaptor", True))
            scorer_kwargs["scorer_weight"] = fit_yaml.get("scorer_weight", 0.5)
            scorer_fuse = fit_yaml.get("scorer_fuse", "mean")

        # Heatmap post-processing (inference-only; from fit YAML)
        heat_edge = bool(fit_yaml.get("heat_edge")) if fit_yaml else None
        heat_edge_sigma = fit_yaml.get("heat_edge_sigma", 1.0)
        heat_norm = fit_yaml.get("heat_norm") or None

        ema_eval = deepcopy(trainer.ema.ema).eval()
        # Apply the fit YAML's bank-build knobs (bb_K / bb_temperature / calibration / bb_layers) onto the
        # eval copy so the OOD bank is built exactly as YOLOA.fit would post-training — the fit YAML is the
        # single source of truth, overriding the model-baked v2_cfg defaults. No-op without a fit YAML.
        if fit_yaml:
            from ultralytics.yoloa import apply_bb_overrides
            apply_bb_overrides(ema_eval, fit_yaml)
        trainer._ood_heatmap_map10 = None  # clear stale; only set on success
        try:
            with torch.no_grad():
                rows = run_mvtec_ood_eval(
                    ema_eval, root,
                    categories=v2_cfg.get("test_categories"),
                    modes=modes,
                    imgsz=imgsz, batch=batch, workers=trainer.args.workers, bank_size=bank_size,
                    device=trainer.device, save_dir=trainer.save_dir, epoch=trainer.epoch + 1,
                    e2e=False, iou=0.1,
                    heatmap_norm=heat_norm,
                    heatmap_edge_weight=(True if heat_edge else None),
                    heatmap_edge_sigma=heat_edge_sigma,
                    scorer_kwargs=scorer_kwargs,
                    scorer_fuse=scorer_fuse,
                )
            AnomalyV2Trainer._log_ood_wandb(trainer, rows)
            # Store OOD heatmap mAP10 (test_metrics(heatmap_prior)) for best.pt selection (fitness override)
            for r in rows:
                if r["category"] == "AVERAGE" and r["mode"] == heatmap_variant:
                    map10 = r.get("mAP10", math.nan)
                    if not math.isnan(map10):
                        trainer._ood_heatmap_map10 = float(map10)
                    break
        except Exception as e:  # never let OOD eval take down a training run
            LOGGER.warning(f"MVTec OOD eval failed at epoch {trainer.epoch + 1}: {type(e).__name__}: {e}")
        finally:
            del ema_eval

    @staticmethod
    def _log_ood_wandb(trainer: "AnomalyV2Trainer", rows: list) -> None:
        """Push OOD metrics to wandb grouped by mode (test_metrics(none_prior/heatmap_prior/mask_prior)).

        For each OOD mode, logs the AVERAGE row (mean across all 15 MVTec categories)
        with metrics {mAP10, mAP25, mAP50, mAP50_95, image_auroc, pixel_auroc}.
        NaN values are skipped. Logged at ``step=epoch+1`` so it merges into the
        same wandb history row the loss/lr already populate this epoch.
        """
        if not rows:
            return
        try:
            from ultralytics.utils.callbacks.wb import wb
        except Exception:
            wb = None
        if wb is None or getattr(wb, "run", None) is None:
            return

        keys = ("mAP10", "mAP25", "mAP50", "mAP50_95", "image_auroc", "pixel_auroc")
        # Map OOD val modes to wandb group names. heatmap_learned / heatmap_fused (scorer modes)
        # also land under test_metrics(heatmap_prior) so they are never silently dropped from wandb.
        mode_to_group = {
            "mask_off": "test_metrics(none_prior)",
            "heatmap": "test_metrics(heatmap_prior)",
            "heatmap_learned": "test_metrics(heatmap_prior)",
            "heatmap_fused": "test_metrics(heatmap_prior)",
            "mask_on": "test_metrics(mask_prior)",
        }

        log = {}
        for mode, group_name in mode_to_group.items():
            # Find AVERAGE row for this mode
            avg_row = next((r for r in rows if r["category"] == "AVERAGE" and r["mode"] == mode), None)
            if avg_row is not None:
                for k in keys:
                    if avg_row.get(k) is not None and not (isinstance(avg_row.get(k), float) and math.isnan(avg_row[k])):
                        log[f"{group_name}/{k}"] = avg_row[k]

        if log:
            try:
                wb.run.log(log, step=trainer.epoch + 1)
            except Exception as e:
                LOGGER.warning(f"MVTec OOD: wandb log failed: {type(e).__name__}: {e}")

    @staticmethod
    def _update_seg_alpha(trainer: "AnomalyV2Trainer") -> None:
        """Set ``seg_alpha`` per ``model.seg_alpha_mode`` (curriculum|pinned_one|pinned_zero).

        Curriculum: alpha = 1 at epoch 0, linear to 0 at ``epochs - close_mosaic``, then 0.
        Pinned modes hold alpha at 1 or 0 throughout — for ablation runs.
        The value is mirrored onto the EMA model, which validation actually runs on.
        """
        model = unwrap_model(trainer.model)
        if getattr(model, "seg_branch", None) is None:
            return
        mode = getattr(model, "seg_alpha_mode", "curriculum")
        if mode == "pinned_one":
            alpha = 1.0
        elif mode == "pinned_zero":
            alpha = 0.0
        else:
            curriculum_end = max(trainer.epochs // 2, trainer.epochs - trainer.args.close_mosaic)
            alpha = max(0.0, 1.0 - trainer.epoch / max(1, curriculum_end))
        model.seg_alpha = alpha
        if trainer.ema is not None:
            unwrap_model(trainer.ema.ema).seg_alpha = alpha
