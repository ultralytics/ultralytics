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

import math
from copy import copy, deepcopy
from pathlib import Path

import torch

from ultralytics.data import build_yolo_dataset
from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import YOLOAnomalyV2Model, YOLOAnomalyV2SegModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, YAML
from ultralytics.utils.torch_utils import unwrap_model

from .val import resolve_mvtec_root, run_mvtec_ood_eval

# Fit YAML keys -> FeatureDiscriminatorScorer kwargs
_SCORER_YAML_KEYS = {
    "scorer_noise_std": "noise_std",
    "scorer_steps": "steps",
    "scorer_hidden": "hidden",
    "scorer_n_noise": "n_noise",
    "scorer_batch": "batch",
    "scorer_lr": "lr",
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
        # insert this so wandb can work correctly
        self.callbacks["on_fit_epoch_end"].insert(0, AnomalyV2Trainer._mvtec_ood_eval)
        self.add_callback("on_train_batch_end", AnomalyV2Trainer._visualize_prior_mask)

    def validate(self):
        """Run validation, overriding fitness with OOD heatmap mAP50 when available."""
        metrics, fitness = super().validate()
        ood_map50 = getattr(self, "_ood_heatmap_map50", None)
        if ood_map50 is not None and ood_map50 > 0 and metrics is not None:
            fitness = ood_map50
            metrics["fitness"] = ood_map50
            if not self.best_fitness or self.best_fitness < ood_map50:
                self.best_fitness = ood_map50
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

    def preprocess_batch(self, batch: dict) -> dict:
        """Store a reference to the preprocessed batch so callbacks can access it.

        The base detection preprocessor only moves tensors to the device and normalizes images;
        we keep the returned batch dict for visualization callbacks without changing behavior.
        """
        batch = super().preprocess_batch(batch)
        self._last_batch = batch
        return batch

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Dataset selection: segment (polygon masks for the SegBranch refiner target / per-instance
        seg) > default detection."""
        if self._seg_polygon_active() or self._instance_seg_active():
            # task="segment" flips YOLODataset.use_segments -> Format(return_mask=True) ->
            # batch["masks"] (overlap-union instance map). Detection head/loss unaffected
            # (bboxes still present). copy() so the persistent self.args stays task="anomaly_v2".
            args = copy(self.args)
            args.task = "segment"
            gs = max(int(unwrap_model(self.model).stride.max()), 32)
            return build_yolo_dataset(args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
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
        """Return an AnomalyV2Validator (single-pass, legacy GT mask rendering for training)."""
        model = unwrap_model(self.model)
        if self._instance_seg_active():
            # v8SegmentationLoss returns [box, seg, cls, dfl, sem] (5 components).
            loss_names = ["box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"]
        else:
            loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        if getattr(model, "seg_branch", None) is not None:
            # Rename to seg_prior_loss when instance seg is active to avoid collision
            # with the per-instance seg_loss above.
            label = "seg_prior_loss" if self._instance_seg_active() else "seg_loss"
            loss_names.append(label)
        # QueryFiLM appends 4 aux components in this exact order (see YOLOAnomalyV2Model.loss).
        if getattr(model, "fusion_mode", None) == "queryfilm":
            loss_names += ["qmask_loss", "qobj_loss", "qovl_loss", "qfg_loss"]
        self.loss_names = tuple(loss_names)
        return yolo.anomaly_v2.AnomalyV2Validator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            prior_mode=None,  # legacy GT bboxes -> renderer
        )

    @staticmethod
    def _mvtec_ood_eval(trainer: "AnomalyV2Trainer") -> None:
        """on_fit_epoch_end: periodic MVTec cross-dataset OOD eval, rank 0 only.

        Configured via the model YAML ``anomaly_v2`` block: ``mvtec_ood_val_freq`` (every N epochs;
        default 3, set 0 to disable). When ``mvtec_ood_fit_cfg`` is set, ``imgsz``, ``heatmap_mode``,
        scorer settings, ``heat_edge`` / ``heat_norm`` are read from that fit YAML (the single source of
        truth for all fit params). Without it, ``mvtec_ood_imgsz`` (default 320) and a plain 3-mode
        (mask_off / heatmap / mask_on) sweep with no scorer / no post-processing are used.

        Runs on a deepcopy of the EMA so val-time fuse()/bank-build never corrupt the live EMA.
        """
        if RANK not in (-1, 0) or trainer.ema is None:
            return
        v2_cfg = getattr(unwrap_model(trainer.model), "yaml", {}).get("anomaly_v2", {})
        freq = int(v2_cfg.get("mvtec_ood_val_freq", 3))
        if freq <= 0 or (trainer.epoch + 1) % freq != 0:
            return
        root = resolve_mvtec_root(v2_cfg.get("mvtec_ood_root"))
        if root is None:
            LOGGER.warning(
                "MVTec OOD: dataset root not found (set MVTEC_ROOT or anomaly_v2.mvtec_ood_root); skipping OOD eval."
            )
            return

        # -- Fit YAML (optional; single source of truth for imgsz / heatmap_mode / scorer / post) --
        fit_cfg_path = v2_cfg.get("mvtec_ood_fit_cfg")
        fit_yaml = {}
        if fit_cfg_path:
            model_yaml = getattr(unwrap_model(trainer.model), "yaml", {}) or {}
            model_yaml_path = model_yaml.get("yaml_file")
            fit_yaml = _resolve_fit_yaml(str(fit_cfg_path), model_yaml_path)

        imgsz = int(fit_yaml.get("imgsz", v2_cfg.get("mvtec_ood_imgsz", 320)))
        batch = int(v2_cfg.get("mvtec_ood_batch", 8))
        bank_size = int(fit_yaml.get("bb_max_bank_size", v2_cfg.get("mvtec_ood_bank_size", 10000)))

        # Modes: fit YAML heatmap_mode -> prior variant; without it = plain 3-mode sweep
        heatmap_mode = fit_yaml.get("heatmap_mode")
        _MODE_MAP = {"memory_bank": "heatmap", "learned": "heatmap_learned", "fused": "heatmap_fused"}
        if heatmap_mode in _MODE_MAP:
            # modes = ("mask_off", _MODE_MAP[heatmap_mode], "mask_on")
            modes = (_MODE_MAP[heatmap_mode],)
        else:
            modes = ("mask_off", "heatmap", "mask_on")

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
        trainer._ood_heatmap_map50 = None  # clear stale; only set on success
        try:
            with torch.no_grad():
                rows = run_mvtec_ood_eval(
                    ema_eval,
                    root,
                    categories=v2_cfg.get("mvtec_ood_categories"),
                    modes=modes,
                    imgsz=imgsz,
                    batch=batch,
                    bank_size=bank_size,
                    device=trainer.device,
                    save_dir=trainer.save_dir,
                    epoch=trainer.epoch + 1,
                    e2e=False,
                    iou=0.1,
                    heatmap_norm=heat_norm,
                    heatmap_edge_weight=(True if heat_edge else None),
                    heatmap_edge_sigma=heat_edge_sigma,
                    scorer_kwargs=scorer_kwargs,
                    scorer_fuse=scorer_fuse,
                )
            AnomalyV2Trainer._log_ood_wandb(trainer, rows)
            # Store OOD heatmap mAP50 for best.pt selection (fitness override)
            heatmap_mode = modes[1] if len(modes) > 1 else modes[0]
            for r in rows:
                if r["category"] == "AVERAGE" and r["mode"] == heatmap_mode:
                    map50 = r.get("mAP50", math.nan)
                    if not math.isnan(map50):
                        trainer._ood_heatmap_map50 = float(map50)
                    break
        except Exception as e:  # never let OOD eval take down a training run
            LOGGER.warning(f"MVTec OOD eval failed at epoch {trainer.epoch + 1}: {type(e).__name__}: {e}")
        finally:
            del ema_eval

    @staticmethod
    def _log_ood_wandb(trainer: "AnomalyV2Trainer", rows: list) -> None:
        """Push per-category + AVERAGE OOD metrics to wandb at the current epoch step (no-op if wandb off).

        Keys are ``ood/<category|AVERAGE>/<mode>/<metric>`` for metric in {mAP10, mAP25, mAP50,
        mAP50_95, image_auroc, pixel_auroc}; NaN values are skipped. Logged at ``step=epoch+1`` so it
        merges into the same wandb history row the loss/lr already populate this epoch.
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
        log = {
            f"ood/{r['category']}/{r['mode']}/{k}": r[k]
            for r in rows
            for k in keys
            if r.get(k) is not None and not (isinstance(r.get(k), float) and math.isnan(r[k]))
        }
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

    @staticmethod
    def _visualize_prior_mask(trainer: "AnomalyV2Trainer") -> None:
        """on_train_batch_end: save GT vs augmented prior-mask grids for the first 3 iterations.

        Visualization is rank 0 only and runs under ``torch.no_grad()``. Images are written to
        ``<save_dir>/prior_vis/epoch{epoch}_iter{count}.png``. The torch (and CUDA) RNG state is
        snapshotted around the visualization so the extra random draws do not change training.
        """
        if RANK not in (-1, 0):
            return

        count = getattr(trainer, "_prior_vis_count", 0)
        if count >= 3:
            return
        trainer._prior_vis_count = count + 1

        model = unwrap_model(trainer.model)
        augmenter = getattr(model, "mask_augmenter", None)
        renderer = getattr(model, "mask_renderer", None)
        if augmenter is None or renderer is None:
            return

        batch = getattr(trainer, "_last_batch", None)
        if batch is None:
            return
        bboxes = batch.get("bboxes")
        batch_idx = batch.get("batch_idx")
        if bboxes is None or batch_idx is None:
            return

        save_dir = Path(trainer.save_dir) / "prior_vis"
        save_path = save_dir / f"epoch{trainer.epoch + 1}_iter{count}.png"

        # Snapshot RNG state so the visualization's random draws do not alter training.
        rng_states = [torch.get_rng_state()]
        if torch.cuda.is_available():
            rng_states.extend(torch.cuda.get_rng_state(d) for d in range(torch.cuda.device_count()))
        try:
            augmenter.visualize(
                renderer=renderer,
                bboxes=bboxes,
                batch_idx=batch_idx,
                batch_size=int(batch["img"].shape[0]),
                save_path=save_path,
            )
        except Exception as e:
            LOGGER.warning(f"Prior-mask visualization failed: {type(e).__name__}: {e}")
        finally:
            torch.set_rng_state(rng_states[0])
            if torch.cuda.is_available():
                for d, state in enumerate(rng_states[1:]):
                    torch.cuda.set_rng_state(state, d)

    def plot_metrics(self):
        """Plot metrics from a CSV file."""
        pass  # skip results plotting for now
