# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depth estimation validator for YOLO models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.metrics import DepthMetrics
from ultralytics.utils.plotting import plot_images


class DepthValidator(DetectionValidator):
    """Validator for YOLO depth estimation models.

    Computes standard depth metrics: delta1, abs_rel, rmse, silog. Uses validation loss as the primary training signal.
    """

    def __init__(
        self,
        dataloader=None,
        save_dir: str | Path | None = None,
        args=None,
        _callbacks: dict | None = None,
    ) -> None:
        """Initialize DepthValidator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "depth"

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize the DepthMetrics accumulator with the dataset's depth range."""
        self.metrics = DepthMetrics(max_depth=self.data.get("max_depth") or 100.0)
        self.metrics.clear_stats()

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch — move to device, normalize images, and keep depth as float32."""
        batch = super().preprocess(batch)
        batch["depth"] = batch["depth"].float()
        return batch

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """No NMS needed for depth — return predictions as-is."""
        return preds

    def update_metrics(self, preds: torch.Tensor, batch: dict[str, Any]) -> None:
        """Accumulate depth metrics for a batch."""
        gt_depth = batch["depth"]
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        if preds.ndim == 3:
            preds = preds.unsqueeze(1)
        if preds.shape[-2:] != gt_depth.shape[-2:]:
            preds = F.interpolate(preds.float(), size=gt_depth.shape[-2:], mode="bilinear", align_corners=True)
        self.metrics.update_stats(preds, gt_depth)

    def get_stats(self) -> dict[str, float]:
        """Finalize and return the metrics dict.

        Cross-rank metric reduction is handled by gather_stats() (called before this on all ranks);
        this runs on rank 0 with the already-summed accumulators.
        """
        self.metrics.process()
        return self.metrics.results_dict

    def gather_stats(self) -> None:
        """Sum depth metric accumulators across DDP ranks onto rank 0.

        Validation is sharded (ContiguousDistributedSampler gives each rank a distinct chunk of the
        val set), so each rank holds only its shard's summed statistics. All-reduce the sums so
        rank 0's get_stats() computes metrics over the full val set instead of a single shard.
        Overrides DetectionValidator.gather_stats(), which reduces detection-specific stats/box
        attributes that DepthMetrics does not have.
        """
        if RANK == -1 or not dist.is_initialized():
            return
        totals = self.metrics._totals
        totals = (
            totals.to(self.device) if totals is not None else torch.zeros(7, dtype=torch.float64, device=self.device)
        )
        count = torch.tensor([self.metrics._count], dtype=torch.float64, device=self.device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        self.metrics._totals = totals
        self.metrics._count = float(count.item())

    def print_results(self) -> None:
        """Log the headline depth metrics in the detection-style aligned table format.

        Columns line up with get_desc(): Class, Images, delta1, abs_rel, rmse, silog.
        Uses "depth_val" as the row label (depth has no classes, where detection prints "all").
        """
        r = self.metrics.results_dict
        n_images = len(self.dataloader.dataset) if self.dataloader is not None else (self.seen or 0)
        pf = "%22s" + "%11i" + "%11.4g" * 4  # label, Images, delta1, abs_rel, rmse, silog
        LOGGER.info(
            pf
            % (
                "depth_val",
                n_images,
                r.get("metrics/delta1", 0.0),
                r.get("metrics/abs_rel", 0.0),
                r.get("metrics/rmse", 0.0),
                r.get("metrics/silog", 0.0),
            )
        )

    def finalize_metrics(self) -> None:
        """Set final values for metrics speed."""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir

    def get_desc(self) -> str:
        """Return description for progress bar."""
        return ("%22s" + "%11s" * 5) % ("Class", "Images", "delta1", "abs_rel", "rmse", "silog")

    def plot_predictions(self, batch: dict[str, Any], preds: torch.Tensor, ni: int) -> None:
        """Save predicted depth overlays to val_batch{ni}_pred.jpg.

        Depth has no boxes/classes, so the detection-style plotter is replaced with a depth heatmap overlay
        through the shared ``plot_images`` path, matching the semantic-segmentation visualization style.
        """
        plot_images(
            labels={"depth": preds},
            images=batch["img"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
