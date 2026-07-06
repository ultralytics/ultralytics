# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 trainer.

Thin extension of ``DetectionTrainer``. The differences:
  * ``get_model`` returns a ``YOLOAnomalyV2Model`` instead of a plain ``DetectionModel``.
  * ``get_validator`` returns a ``YOLOAnomalyValidator`` that runs val twice
    (mask-on and mask-off) — see ``val.py``.

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
from ultralytics.nn.tasks import YOLOAnomalyV2Model, load_checkpoint
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, YAML
from ultralytics.utils.torch_utils import unwrap_model

from .val import resolve_mvtec_root, run_mvtec_ood_eval


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
    """Trainer for YOLOAnomalyV2."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        # NOTE: _mvtec_ood_eval is NOT a callback — it runs inside validate() so best.pt
        # selection sees the current epoch's OOD score (no one-epoch lag).
        self._ood_best_fitness = None  # running max of the OOD-scale fitness (best.pt selection)

    def validate(self):
        """Run validation; when OOD eval is enabled, use OOD none-prior mAP10_50 (test_metrics(none_prior)) as fitness.

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
            self._ood_none_map1050 = None  # skipped (freq) epochs -> 0.0, not a best.pt candidate
            AnomalyV2Trainer._mvtec_ood_eval(self)  # sets self._ood_none_map1050 (+ wandb log)
            fitness = float(self._ood_none_map1050 or 0.0)
            metrics["fitness"] = fitness
            self._ood_best_fitness = (
                fitness if self._ood_best_fitness is None else max(self._ood_best_fitness, fitness)
            )
            self.best_fitness = self._ood_best_fitness
        return metrics, fitness

    def _polygon_prior_active(self) -> bool:
        """True when the fusion prior uses v6 polygon masks (seg_target_polygon)."""
        return bool(getattr(unwrap_model(self.model), "seg_target_polygon", False))

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build the dataset, forcing polygon-mask loading when polygon-prior mode is on."""
        if self._polygon_prior_active():
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

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Build the standard optimizer, then move fusion params into groups with ``lr * fusion_lr_scale``.

        ``anomaly_v2.fusion_lr_scale`` (default 1.0) multiplies the LR of every ``heatmap_bias_fusion``
        parameter. Splitting AFTER the standard build keeps each moved param's group hyperparameters
        (weight_decay / momentum / use_muon for MuSGD); the scheduler and warmup both scale from the
        group's own lr, so the ratio holds across the whole schedule.
        """
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        m = unwrap_model(model)
        scale = float((getattr(m, "yaml", {}) or {}).get("anomaly_v2", {}).get("fusion_lr_scale", 1.0))
        fusion = getattr(m, "heatmap_bias_fusion", None)
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
            f"anomaly_v2: fusion_lr_scale={scale} -> {n_moved} heatmap_bias_fusion params split into "
            f"scaled groups ({len(optimizer.param_groups)} groups total)"
        )
        return optimizer

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Return a ``YOLOAnomalyV2Model``.

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
        """Return the anomaly validator (2-pass, legacy GT mask rendering for training)."""
        model = unwrap_model(self.model)
        if isinstance(model.model[-1], AnomalyMCDetect):
            # AnomalyMCLoss returns [box, anom, dfl, type] (decoupled detection + type).
            loss_names = ["box_loss", "anom_loss", "dfl_loss", "type_loss"]
        else:
            loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        # QueryFiLM appends 4 aux components in this exact order (see YOLOAnomalyV2Model.loss).
        if getattr(model, "fusion_mode", None) == "queryfilm":
            loss_names += ["qmask_loss", "qobj_loss", "qovl_loss", "qfg_loss"]
        self.loss_names = tuple(loss_names)
        return yolo.anomaly_v2.YOLOAnomalyValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args),
            _callbacks=self.callbacks, prior_mode=None,  # legacy GT bboxes -> renderer
        )

    @staticmethod
    def _ood_eval_setup(trainer: "AnomalyV2Trainer") -> dict | None:
        """Resolve the OOD-eval configuration shared by the per-epoch and final passes.

        Reads the model YAML ``anomaly_v2`` block; when ``test_fit_cfg`` is set, ``imgsz``,
        bank-build knobs and heatmap post-processing come from that fit YAML (the single source
        of truth for all fit params). Without it, ``test_imgsz`` (default 320) and the default
        heatmap prior with no post-processing are used. Returns None when the MVTec root is
        unavailable.
        """
        v2_cfg = getattr(unwrap_model(trainer.model), "yaml", {}).get("anomaly_v2", {})
        root = resolve_mvtec_root(v2_cfg.get("test_root"))
        if root is None:
            LOGGER.warning("MVTec test: dataset root not found (set MVTEC_ROOT or anomaly_v2."
                           "test_root); skipping test eval.")
            return None

        # -- Fit YAML (optional; single source of truth for imgsz / bank knobs / post-processing) --
        fit_cfg_path = v2_cfg.get("test_fit_cfg")
        fit_yaml = {}
        if fit_cfg_path:
            model_yaml = getattr(unwrap_model(trainer.model), "yaml", {}) or {}
            fit_yaml = _resolve_fit_yaml(str(fit_cfg_path), model_yaml.get("yaml_file"))

        # Per-prior test switches pick which OOD modes run. test_none_prior is forced on because
        # best.pt fitness is test_metrics(none_prior)/mAP10_50; test_heatmap_prior defaults on
        # (core fusion signal), test_mask_prior (mask_on) defaults off.
        if not bool(v2_cfg.get("test_none_prior", True)):
            LOGGER.warning("anomaly_v2: test_none_prior feeds fitness and cannot be disabled; forcing on.")
        modes = ["mask_off"]  # test_none_prior — always on (fitness source)
        if bool(v2_cfg.get("test_heatmap_prior", True)):
            modes.append("heatmap")
        if bool(v2_cfg.get("test_mask_prior", False)):
            modes.append("mask_on")

        return {
            "v2_cfg": v2_cfg, "root": root, "fit_yaml": fit_yaml,
            "imgsz": int(fit_yaml.get("imgsz", v2_cfg.get("test_imgsz", 320))),
            "batch": int(v2_cfg.get("test_batch", 8)),
            "bank_size": int(fit_yaml.get("bb_max_bank_size", v2_cfg.get("test_bank_size", 10000))),
            "modes": tuple(modes),
            # Heatmap post-processing (inference-only; from fit YAML)
            "heat_edge": bool(fit_yaml.get("heat_edge")) if fit_yaml else None,
            "heat_edge_sigma": fit_yaml.get("heat_edge_sigma", 1.0),
            "heat_norm": fit_yaml.get("heat_norm") or None,
        }

    @staticmethod
    def _ood_eval_prep_model(model, fit_yaml: dict) -> None:
        """Apply fit-YAML bank-build knobs + heatmap conf gate onto an eval copy of the model.

        The bank knobs (bb_K / bb_temperature / calibration / bb_layers) make the OOD bank build
        exactly as YOLOA.fit would post-training; the gate (p_anom *= blend + (1-blend)*hm) is the
        same knob run_yoloa.py resolves from the same fit YAML — so in-training OOD metrics and
        post-hoc CLI val score the identical graph. No-op without a fit YAML.
        """
        if fit_yaml:
            from ultralytics.yoloa import apply_bb_overrides
            apply_bb_overrides(model, fit_yaml)
        gb = fit_yaml.get("hm_gate_blend") if fit_yaml else None
        if gb is not None and float(gb) < 1.0:
            gb = float(gb)
            model.hm_gate_blend = gb
            for _h in [model.model[-1]] + ([model.head_b] if getattr(model, "two_head", False) else []):
                _h.hm_gate_blend = gb
            LOGGER.info(f"MVTec OOD: hm_gate_blend={gb} (heatmap conf gate ON, from fit yaml)")

    @staticmethod
    def _run_ood_eval(trainer: "AnomalyV2Trainer", model, cfg: dict, epoch) -> list[dict]:
        """Run ``run_mvtec_ood_eval`` with a resolved cfg (shared by per-epoch and final passes)."""
        with torch.no_grad():
            return run_mvtec_ood_eval(
                model, cfg["root"],
                categories=cfg["v2_cfg"].get("test_categories"),
                modes=cfg["modes"],
                imgsz=cfg["imgsz"], batch=cfg["batch"], workers=trainer.args.workers,
                bank_size=cfg["bank_size"],
                device=trainer.device, save_dir=trainer.save_dir, epoch=epoch,
                e2e=False, iou=0.1,
                heatmap_norm=cfg["heat_norm"],
                heatmap_edge_weight=(True if cfg["heat_edge"] else None),
                heatmap_edge_sigma=cfg["heat_edge_sigma"],
            )

    @staticmethod
    def _mvtec_ood_eval(trainer: "AnomalyV2Trainer") -> None:
        """Periodic MVTec cross-dataset OOD eval, rank 0 only; called from validate().

        Configured via the model YAML ``anomaly_v2`` block: ``test_val_freq`` (every N epochs;
        default 3, set 0 to disable); see ``_ood_eval_setup`` for the fit-YAML / mode resolution.
        ``test_none_prior`` is forced on (best.pt fitness is ``test_metrics(none_prior)/mAP10_50``).

        Runs on a deepcopy of the EMA so val-time fuse()/bank-build never corrupt the live EMA.
        """
        if RANK not in (-1, 0) or trainer.ema is None:
            return
        v2_cfg = getattr(unwrap_model(trainer.model), "yaml", {}).get("anomaly_v2", {})
        freq = int(v2_cfg.get("test_val_freq", 3))
        if freq <= 0 or (trainer.epoch + 1) % freq != 0:
            return
        cfg = AnomalyV2Trainer._ood_eval_setup(trainer)
        if cfg is None:
            return
        ema_eval = deepcopy(trainer.ema.ema).eval()
        AnomalyV2Trainer._ood_eval_prep_model(ema_eval, cfg["fit_yaml"])
        trainer._ood_none_map1050 = None  # clear stale; only set on success
        try:
            rows = AnomalyV2Trainer._run_ood_eval(trainer, ema_eval, cfg, trainer.epoch + 1)
            AnomalyV2Trainer._log_ood_wandb(trainer, rows)
            # Store OOD none-prior mAP10_50 (test_metrics(none_prior)) for best.pt selection (fitness override)
            for r in rows:
                if r["category"] == "AVERAGE" and r["mode"] == "mask_off":
                    map1050 = r.get("mAP10_50", math.nan)
                    if not math.isnan(map1050):
                        trainer._ood_none_map1050 = float(map1050)
                    break
        except Exception as e:  # never let OOD eval take down a training run
            LOGGER.warning(f"MVTec OOD eval failed at epoch {trainer.epoch + 1}: {type(e).__name__}: {e}")
        finally:
            del ema_eval

    def final_eval(self):
        """Standard final in-dist val on best.pt, then one MVTec OOD eval on the same weights."""
        super().final_eval()
        self._mvtec_ood_eval_final()

    def _mvtec_ood_eval_final(self) -> None:
        """Re-run the MVTec OOD eval on the saved best.pt after training (epoch label ``final``).

        The per-epoch OOD rows score each epoch's EMA; this pass scores the saved checkpoint
        itself, producing the exact reference row a post-hoc ``run_yoloa.py --mode val`` on the
        pulled best.pt should reproduce (same weights / fit yaml / gate; residual = fp16 ckpt
        round-trip + device numerics). Runs whenever the periodic OOD eval is enabled; disable
        just this pass with ``anomaly_v2.test_final_eval: false``.
        """
        if RANK not in (-1, 0):
            return
        v2_cfg = getattr(unwrap_model(self.model), "yaml", {}).get("anomaly_v2", {})
        if int(v2_cfg.get("test_val_freq", 3)) <= 0 or not bool(v2_cfg.get("test_final_eval", True)):
            return
        ckpt_path = self.best if self.best.exists() else self.last
        if not ckpt_path.exists():
            return
        cfg = AnomalyV2Trainer._ood_eval_setup(self)
        if cfg is None:
            return
        LOGGER.info(f"MVTec OOD final eval on {ckpt_path}...")
        try:
            model, _ = load_checkpoint(str(ckpt_path), device=self.device)  # FP32, eval mode
            AnomalyV2Trainer._ood_eval_prep_model(model, cfg["fit_yaml"])
            AnomalyV2Trainer._run_ood_eval(self, model, cfg, "final")
        except Exception as e:  # never let the final OOD eval mask a finished training
            LOGGER.warning(f"MVTec OOD final eval failed: {type(e).__name__}: {e}")

    @staticmethod
    def _log_ood_wandb(trainer: "AnomalyV2Trainer", rows: list) -> None:
        """Push OOD metrics to wandb grouped by mode (test_metrics(none_prior/heatmap_prior/mask_prior)).

        For each OOD mode, logs the AVERAGE row (mean across all 15 MVTec categories)
        with metrics {mAP10, mAP25, mAP50, mAP10_50, image_auroc, pixel_auroc}.
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

        keys = ("mAP10", "mAP25", "mAP50", "mAP10_50", "image_auroc", "pixel_auroc")
        mode_to_group = {
            "mask_off": "test_metrics(none_prior)",
            "heatmap": "test_metrics(heatmap_prior)",
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
