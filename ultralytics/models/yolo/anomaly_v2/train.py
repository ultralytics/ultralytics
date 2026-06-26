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
import random
from copy import copy, deepcopy
from pathlib import Path

import torch

from ultralytics.data import build_yolo_dataset
from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import YOLOAnomalyV2Model, YOLOAnomalyV2SegModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, TQDM, YAML
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
        self.add_callback("on_train_start", AnomalyV2Trainer._mb_initial_fill)
        self.add_callback("on_train_batch_end", AnomalyV2Trainer._mb_step)
        self.add_callback("on_train_start", AnomalyV2Trainer._cached_draw_prior)
        self.add_callback("on_train_batch_end", AnomalyV2Trainer._cached_draw_prior)
        # NOTE: _mvtec_ood_eval is NOT a callback — it runs inside validate() so best.pt
        # selection sees the current epoch's OOD score (no one-epoch lag).
        self._mb_batch = None  # stashed preprocessed batch for the batch-end enqueue
        self._mb_patches_per_step = 0
        self._ood_best_fitness = None  # running max of the OOD-scale fitness (best.pt selection)

    def preprocess_batch(self, batch):
        """Stash tensor refs so the batch-end callback can enqueue this batch's features.

        Snapshot the tensors, not the dict: rank-0 plotting (``plot_training_samples``)
        later replaces the dict's bboxes/cls with numpy arrays in place, which would crash
        the enqueue's mask render.
        """
        batch = super().preprocess_batch(batch)
        if self._mb_active():
            self._mb_batch = {"img": batch["img"], "bboxes": batch["bboxes"], "batch_idx": batch["batch_idx"]}
        return batch

    def validate(self):
        """Run validation; when OOD eval is enabled, use OOD heatmap mAP50 as the fitness.

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
        if int(v2_cfg.get("mvtec_ood_val_freq", 3)) > 0 and metrics is not None:
            self._ood_heatmap_map50 = None  # skipped (freq) epochs -> 0.0, not a best.pt candidate
            AnomalyV2Trainer._mvtec_ood_eval(self)  # sets self._ood_heatmap_map50 (+ wandb log)
            fitness = float(self._ood_heatmap_map50 or 0.0)
            metrics["fitness"] = fitness
            self._ood_best_fitness = (
                fitness if self._ood_best_fitness is None else max(self._ood_best_fitness, fitness)
            )
            self.best_fitness = self._ood_best_fitness
        return metrics, fitness

    def _mb_active(self) -> bool:
        """True when the model trains with the MoCo-style FIFO-queue prior."""
        model = unwrap_model(self.model)
        return getattr(model, "memory_bank", None) is not None and getattr(model, "mb_queue_capacity", 0) > 0

    # ------------------------------------------------------------------
    # Offline cached prior (mb_cached_prior: true)
    # ------------------------------------------------------------------
    def _cached_active(self) -> bool:
        """True when the model trains with the precomputed-heatmap (4th channel) prior."""
        return bool(getattr(unwrap_model(self.model), "mb_cached_prior", False))

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
        """Dataset selection: CachedPriorDataset (4th-ch heatmap) > segment (polygon masks for the
        SegBranch refiner target / per-instance seg) > default detection."""
        if not self._cached_active():
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
        from ultralytics.utils.torch_utils import unwrap_model as _um

        from .dataset import CachedPriorDataset

        p = Path(img_path)
        if p.is_file():  # list-style dataset (<root>/train.txt) -> <root>/heatmaps_v1/train
            prior_dir = p.parent / "heatmaps_v1" / p.stem
        else:  # dir-style dataset (<root>/images/train) -> <root>/heatmaps_v1/train
            prior_dir = p.parents[1] / "heatmaps_v1" / p.name
        gs = max(int(_um(self.model).stride.max()), 32)
        return CachedPriorDataset(
            prior_dir=prior_dir,
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def plot_training_samples(self, batch, ni):
        """Slice the prior channel off before plotting (plot_images expects 3-channel)."""
        if batch["img"].shape[1] == 4:
            batch = {**batch, "img": batch["img"][:, :3]}
        super().plot_training_samples(batch, ni)

    @staticmethod
    def _cached_draw_prior(trainer: "AnomalyV2Trainer") -> None:
        """Per-batch draw for the cached-prior arm: prior_mode='cached' with p=mb_blend_p, else GT.

        Live model only — the validator drives the EMA model itself (prior_mode=None 2-pass).
        """
        if not trainer._cached_active():
            return
        model = unwrap_model(trainer.model)
        mode = "cached" if random.random() < float(getattr(model, "mb_blend_p", 0.5)) else None
        model.set_prior_mode(mode)

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
        else:
            loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        if getattr(model, "seg_branch", None) is not None:
            # Rename to seg_prior_loss when instance seg is active to avoid collision
            # with the per-instance seg_loss above.
            label = "seg_prior_loss" if seg_active else "seg_loss"
            loss_names.append(label)
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
            LOGGER.warning("MVTec OOD: dataset root not found (set MVTEC_ROOT or anomaly_v2."
                           "mvtec_ood_root); skipping OOD eval.")
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
            modes = ("mask_off", _MODE_MAP[heatmap_mode], "mask_on")
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
                    ema_eval, root,
                    categories=v2_cfg.get("mvtec_ood_categories"),
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
            # Store OOD heatmap mAP50 for best.pt selection (fitness override)
            heatmap_mode = modes[1]
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
        """Push OOD metrics to wandb grouped by mode (ood_none, ood_heatmap, ood_mask).

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
        # Map val modes to OOD group names
        mode_to_group = {"mask_off": "ood_none", "heatmap": "ood_heatmap", "mask_on": "ood_mask"}

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

    # ------------------------------------------------------------------
    # MoCo-style FIFO-queue prior (mb_queue_capacity > 0)
    # ------------------------------------------------------------------
    @staticmethod
    def _mb_initial_fill(trainer: "AnomalyV2Trainer") -> None:
        """Build the initial FIFO queue: one pass over the train loader through the EMA encoder.

        At this point the EMA model equals the (pretrained-init) live model, so the queue
        starts as a uniform sample of normal patches under the initial backbone. Per-batch
        ``max_patches`` is sized so one epoch's worth of steps roughly fills the capacity,
        making the steady-state queue span ~1 epoch of history.
        """
        if not trainer._mb_active() or trainer.ema is None:
            return
        model = unwrap_model(trainer.model)
        mb = model.memory_bank
        ema_model = unwrap_model(trainer.ema.ema)
        cap = int(model.mb_queue_capacity)
        steps = max(1, len(trainer.train_loader))
        q = max(16, min(4096, math.ceil(cap / steps)))
        trainer._mb_patches_per_step = q
        LOGGER.info(f"MoCo bank: filling FIFO queue (capacity={cap}, {q} patches/step steady) via EMA encoder...")
        initialized = False
        pushed = 0
        for batch in TQDM(trainer.train_loader, desc="MB initial fill", disable=RANK not in {-1, 0}):
            batch = trainer.preprocess_batch(batch)
            feats = ema_model.encode_bb_feats(batch["img"])
            if not feats:
                break
            if not initialized:
                dim = mb._build_fused_feature(feats).shape[1]
                mb.init_queue(cap, dim, device=trainer.device)
                initialized = True
            excl = model.mask_renderer(batch["bboxes"], batch["batch_idx"], batch["img"].shape[0])
            # Dense intake until the queue first fills (early density), then the steady-state
            # quota so the rest of the walk still rotates in epoch-wide coverage.
            pushed += mb.enqueue(feats, exclude_mask=excl, max_patches=None if pushed < cap else q)
        if not initialized:
            trainer._mb_batch = None
            LOGGER.warning("MoCo bank: no features captured during initial fill; queue disabled.")
            return
        mb.temperature = mb.estimate_temperature()
        n_valid = int((mb.memory_bank.norm(dim=1) > 0).sum())
        LOGGER.info(f"MoCo bank ready: {n_valid}/{cap} slots filled, temperature={mb.temperature:.4f}")
        # Point the EMA model's bank at the live queue (shared tensor) so validation and
        # checkpoint saves (deepcopy of the EMA) carry the current contents.
        ema_mb = getattr(ema_model, "memory_bank", None)
        if ema_mb is not None:
            ema_mb.adopt_queue(mb)
        if RANK in {-1, 0}:
            AnomalyV2Trainer._mb_save_init_snapshot(trainer, model, mb, ema_model)
        trainer._mb_batch = None
        AnomalyV2Trainer._mb_draw_prior(trainer)

    @staticmethod
    def _mb_save_init_snapshot(trainer: "AnomalyV2Trainer", model, mb, ema_model) -> None:
        """Save <save_dir>/mb_bank_init.jpg: last fill batch (top) + its bank heatmap overlay (bottom).

        First-glance sanity check that the freshly built queue localizes anomalies on real
        (augmented) train images before any training happens.
        """
        import cv2
        import numpy as np

        batch = trainer._mb_batch
        if batch is None:
            return
        try:
            img = batch["img"][:8]
            feats = ema_model.encode_bb_feats(img)
            was_training = mb.training
            mb.eval()
            with torch.no_grad():
                hm = mb(feats)  # (B, 1, h, w)
            mb.train(was_training)
            size = 320
            bboxes = batch.get("bboxes")
            batch_idx = batch.get("batch_idx")
            rows_top, rows_bot = [], []
            for b in range(img.shape[0]):
                im = (img[b].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), (size, size))
                # GT boxes (green) so hot regions can be judged against actual defects.
                if bboxes is not None and batch_idx is not None and bboxes.numel():
                    for cx, cy, w, h_ in bboxes[batch_idx == b].cpu().tolist():
                        x1, y1 = int((cx - w / 2) * size), int((cy - h_ / 2) * size)
                        x2, y2 = int((cx + w / 2) * size), int((cy + h_ / 2) * size)
                        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                h = hm[b, 0].float().cpu().numpy()
                cmap = cv2.applyColorMap((h.clip(0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                cmap = cv2.resize(cmap, (size, size), interpolation=cv2.INTER_LINEAR)
                rows_top.append(im)
                rows_bot.append(cv2.addWeighted(im, 0.55, cmap, 0.45, 0))
            grid = np.vstack([np.hstack(rows_top), np.hstack(rows_bot)])
            cv2.putText(grid, f"beta={mb.temperature:.2f} cap={mb._queue_capacity}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out = Path(trainer.save_dir) / "mb_bank_init.jpg"
            cv2.imwrite(str(out), grid)
            LOGGER.info(f"MoCo bank: init heatmap snapshot -> {out}")
        except Exception as e:  # never let a viz failure kill the run
            LOGGER.warning(f"MoCo bank: init snapshot failed: {e}")

    @staticmethod
    def _mb_step(trainer: "AnomalyV2Trainer") -> None:
        """Per-batch: enqueue the just-trained batch via the EMA encoder, redraw the prior mode."""
        if not trainer._mb_active() or trainer.ema is None:
            return
        batch, trainer._mb_batch = trainer._mb_batch, None
        model = unwrap_model(trainer.model)
        mb = model.memory_bank
        if batch is not None and getattr(mb, "_queue_capacity", 0) > 0:
            feats = unwrap_model(trainer.ema.ema).encode_bb_feats(batch["img"])
            if feats:
                excl = model.mask_renderer(batch["bboxes"], batch["batch_idx"], batch["img"].shape[0])
                mb.enqueue(feats, exclude_mask=excl, max_patches=trainer._mb_patches_per_step or None)
        AnomalyV2Trainer._mb_draw_prior(trainer)

    @staticmethod
    def _mb_draw_prior(trainer: "AnomalyV2Trainer") -> None:
        """Draw the next batch's prior source: MB heatmap with p=mb_blend_p, else legacy GT mask.

        Set on the LIVE model only — the validator drives the EMA model's prior itself
        (2-pass GT mask-on/off with ``_prior_mode=None``); a mirrored "heatmap" leftover
        would corrupt that val.
        """
        model = unwrap_model(trainer.model)
        mode = "heatmap" if random.random() < float(getattr(model, "mb_blend_p", 0.5)) else None
        model.set_prior_mode(mode)

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
