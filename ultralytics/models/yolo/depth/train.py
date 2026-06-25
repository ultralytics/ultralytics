# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depth estimation trainer for YOLO models."""

from __future__ import annotations

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import DepthModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK


class DepthTrainer(yolo.detect.DetectionTrainer):
    """Trainer for YOLO depth estimation models.

    Multi-source training (list of img_paths) is handled transparently by the base DetectionTrainer/BaseDataset.

    Examples:
        >>> from ultralytics.models.yolo.depth import DepthTrainer
        >>> args = dict(model="yolo26s-depth.yaml", data="depth-mixed.yaml", epochs=100)
        >>> trainer = DepthTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize DepthTrainer."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "depth"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a DepthModel initialized with the given config and weights.

        If the dataset YAML declares ``max_depth`` (meters), it overrides the head's output
        range (``sigmoid × max_depth``). Depth range is a property of the data: training on
        GT beyond the head's range is otherwise unrepresentable. The value persists in saved
        checkpoints, so fine-tuned models predict in the new range at inference.
        """
        model = DepthModel(cfg, ch=self.data.get("channels", 3), nc=self.data["nc"], verbose=verbose and RANK in {-1, 0})
        if weights:
            model.load(weights)
        max_depth = self.data.get("max_depth")
        if max_depth is not None:
            head = model.model[-1]
            if getattr(head, "mode", "sigmoid") == "log":
                if RANK in {-1, 0}:
                    LOGGER.info("log-depth head: dataset max_depth ignored (output is unbounded)")
            elif hasattr(head, "max_depth"):
                if RANK in {-1, 0}:
                    LOGGER.info(f"Depth head max_depth: {head.max_depth} → {float(max_depth)} m (from dataset YAML)")
                head.max_depth = float(max_depth)
        return model

    def preprocess_batch(self, batch):
        """Preprocess batch: normalize images and keep depth as float32."""
        batch = super().preprocess_batch(batch)
        if "depth" in batch:
            batch["depth"] = batch["depth"].float()
        return batch

    def get_validator(self):
        """Return a DepthValidator for model validation."""
        self.loss_names = "silog", "grad"
        return yolo.depth.DepthValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """Plot training samples as RGB | GT-depth panels (consistent with val_batch plots).

        The inherited DetectionTrainer version routes the batch through ``plot_images``, which
        blends the depth colormap over the RGB at alpha=0.6 — a confusing "RGB+depth" tint.
        Render side-by-side instead. Batch is already preprocessed here (img in [0,1]).
        """
        try:
            import cv2
            import numpy as np

            from .val import DepthValidator

            imgs, gt = batch["img"], batch["depth"]
            if gt.ndim == 3:
                gt = gt.unsqueeze(1)
            h, w = imgs.shape[-2:]
            rows = []
            for i in range(min(imgs.shape[0], 8)):
                rgb = (imgs[i].detach().float().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                g = gt[i, 0]
                gv = g[g > 0]
                vmin = float(gv.min()) if gv.numel() else 0.0
                vmax = float(gv.max()) if gv.numel() else 1.0
                gc = cv2.resize(DepthValidator._colorize_depth(g, vmin, vmax), (w, h), interpolation=cv2.INTER_NEAREST)
                rows.append(np.hstack([rgb, gc]))
            cv2.imwrite(str(self.save_dir / f"train_batch{ni}.jpg"), np.vstack(rows))
        except Exception as e:
            LOGGER.warning(f"DepthTrainer: failed to plot train_batch{ni}: {e}")

    def final_eval(self):
        """Run the standard final evaluation, then auto-calibrate the saved checkpoints.

        After training, fits the scale-only log-affine (``cal_a``/``cal_b``) on the validation
        set and writes it into best.pt/last.pt, so the model outputs metric-scaled depth out of
        the box. Disable with ``auto_calibrate=False``.
        """
        super().final_eval()
        if RANK not in {-1, 0} or not self.args.auto_calibrate:
            return
        try:
            from .calibrate import calibrate_checkpoint

            LOGGER.info("Auto-calibrating depth output scale on the validation set...")
            for ckpt in (self.best, self.last):
                if ckpt.exists():
                    calibrate_checkpoint(ckpt, self.test_loader, self.device, dist_power=self.args.cal_dist_pw)
        except Exception as e:
            LOGGER.warning(f"Auto-calibration skipped ({type(e).__name__}: {e}); checkpoints left uncalibrated.")
