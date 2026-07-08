# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depth estimation trainer for YOLO models."""

from __future__ import annotations

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import DepthModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plt_settings


class DepthTrainer(yolo.detect.DetectionTrainer):
    """Trainer for YOLO depth estimation models.

    Multi-source training (list of img_paths) is handled transparently by the base DetectionTrainer/BaseDataset.

    Examples:
        >>> from ultralytics.models.yolo.depth import DepthTrainer
        >>> args = dict(model="yolo26s-depth.yaml", data="nyu-depth.yaml", epochs=100)
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
        model = DepthModel(
            cfg, ch=self.data.get("channels", 3), nc=self.data["nc"], verbose=verbose and RANK in {-1, 0}
        )
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

        The inherited DetectionTrainer version routes the batch through ``plot_images``, which has
        no depth rendering. Batch is already preprocessed here (img in [0,1]).
        """
        try:
            from .val import plot_depth_panels

            plot_depth_panels(batch["img"], batch["depth"], [], self.save_dir / f"train_batch{ni}.jpg", max_images=8)
        except Exception as e:
            LOGGER.warning(f"DepthTrainer: failed to plot train_batch{ni}: {e}")

    @plt_settings()
    def plot_training_labels(self):
        """Plot the training-set GT depth distribution to ``labels.jpg``.

        The depth analog of the detection/semantic label plots. The inherited DetectionTrainer
        version concatenates per-image ``bboxes``/``cls`` (all empty for depth) and hands them to
        ``plot_labels``, whose reductions raise "zero-size array to reduction operation maximum
        which has no identity". Instead, sample GT depth maps from the training set and plot a
        histogram of valid (``> 0``) depth values, annotated with basic statistics.
        """
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np

        LOGGER.info(f"Plotting labels to {self.save_dir / 'labels.jpg'}...")
        dataset = self.train_loader.dataset
        n = len(dataset.im_files)
        if n == 0:
            LOGGER.warning("No depth maps found, skipping label plot.")
            return

        sample_size = min(1000, n)
        indices = np.linspace(0, n - 1, sample_size).astype(int)
        per_map_cap = max(1, 1_000_000 // sample_size)  # bound total memory to ~1M values
        values = []
        for idx in indices:
            if dataset._depth_stack is not None:
                d = np.asarray(dataset._depth_stack[idx], dtype=np.float32)
            else:
                df = dataset._depth_path_for(dataset.im_files[idx])
                if not Path(df).exists():
                    continue
                try:
                    d = np.load(df).astype(np.float32)
                except Exception:
                    continue
            v = d[d > 0].ravel()
            if v.size == 0:
                continue
            if v.size > per_map_cap:  # uniform stride keeps the spatial distribution unbiased
                v = v[np.linspace(0, v.size - 1, per_map_cap).astype(int)]
            values.append(v)

        if not values:
            LOGGER.warning("No valid depth values found, skipping label plot.")
            return

        values = np.concatenate(values)
        vmin, vmax = float(values.min()), float(np.percentile(values, 99.5))
        mean, median, std = float(values.mean()), float(np.median(values)), float(values.std())

        _, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
        ax.hist(values, bins=100, range=(vmin, max(vmax, vmin + 1e-6)), color="#3b7dd8")
        ax.axvline(mean, color="#d8643b", linestyle="--", linewidth=1.5, label=f"mean {mean:.2f} m")
        ax.axvline(median, color="#3bd86b", linestyle="--", linewidth=1.5, label=f"median {median:.2f} m")
        ax.set_xlabel("Depth (m)")
        ax.set_ylabel("Pixels")
        ax.set_title("Training Labels Depth Distribution")
        ax.legend(loc="upper right", frameon=False)
        stats = f"images: {sample_size}\nmin: {vmin:.2f} m\nmax: {values.max():.2f} m\nstd: {std:.2f} m"
        ax.text(
            0.98,
            0.7,
            stats,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, edgecolor="none"),
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

        fname = self.save_dir / "labels.jpg"
        plt.savefig(fname, dpi=200)
        plt.close()
        if self.on_plot:
            self.on_plot(fname)

    def final_eval(self):
        """Run the standard final evaluation, then auto-calibrate the saved checkpoints.

        After training, fits the scale-only log-affine (``cal_a``/``cal_b``) on the validation
        set and writes it into best.pt/last.pt, so the model outputs metric-scaled depth out of
        the box. Disable with ``auto_calibrate=False``. When ``plots`` is set, also writes
        ``val_batch{ni}_calibrated.jpg`` (RGB | GT | raw | calibrated) comparison panels.
        """
        super().final_eval()
        if RANK not in {-1, 0} or not self.args.auto_calibrate:
            return
        try:
            from .calibrate import calibrate_checkpoint

            LOGGER.info("Auto-calibrating depth output scale on the validation set...")
            # Calibrated comparison plots come from the checkpoint that represents the run:
            # best.pt, or last.pt when best was never saved. Each checkpoint is fitted separately.
            plot_ckpt = self.best if self.best.exists() else self.last
            for ckpt in (self.best, self.last):
                if ckpt.exists():
                    plot_dir = self.save_dir if self.args.plots and ckpt == plot_ckpt else None
                    calibrate_checkpoint(
                        ckpt, self.test_loader, self.device, dist_power=self.args.cal_dist_pw, plot_dir=plot_dir
                    )
        except Exception as e:
            LOGGER.warning(f"Auto-calibration skipped ({type(e).__name__}: {e}); checkpoints left uncalibrated.")
