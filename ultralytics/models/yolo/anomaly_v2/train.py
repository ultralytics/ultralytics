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
import random
from copy import copy
from pathlib import Path

import torch

from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import YOLOAnomalyV2Model
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, TQDM
from ultralytics.utils.torch_utils import unwrap_model


class AnomalyV2Trainer(DetectionTrainer):
    """Trainer for YOLOAnomalyV2 (Phase 0 + v2.2 SegBranch)."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.add_callback("on_train_epoch_start", AnomalyV2Trainer._update_seg_alpha)
        self.add_callback("on_train_start", AnomalyV2Trainer._mb_initial_fill)
        self.add_callback("on_train_batch_end", AnomalyV2Trainer._mb_step)
        self._mb_batch = None  # stashed preprocessed batch for the batch-end enqueue
        self._mb_patches_per_step = 0

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

    def _mb_active(self) -> bool:
        """True when the model trains with the MoCo-style FIFO-queue prior."""
        model = unwrap_model(self.model)
        return getattr(model, "memory_bank", None) is not None and getattr(model, "mb_queue_capacity", 0) > 0

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
        """Return an AnomalyV2Validator (single-pass, legacy GT mask rendering for training)."""
        model = unwrap_model(self.model)
        loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        if getattr(model, "seg_branch", None) is not None:
            loss_names.append("seg_loss")
        # QueryFiLM appends 4 aux components in this exact order (see YOLOAnomalyV2Model.loss).
        if getattr(model, "fusion_mode", None) == "queryfilm":
            loss_names += ["qmask_loss", "qobj_loss", "qovl_loss", "qfg_loss"]
        self.loss_names = tuple(loss_names)
        return yolo.anomaly_v2.AnomalyV2Validator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args),
            _callbacks=self.callbacks, prior_mode=None,  # legacy GT bboxes -> renderer
        )

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
        if mb.auto_temperature:
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
            rows_top, rows_bot = [], []
            for b in range(img.shape[0]):
                im = (img[b].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), (size, size))
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
