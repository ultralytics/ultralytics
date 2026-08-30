# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Detection training over a corpus of merged sources that keep separate label spaces.

Each source is its own `YOLODataset` holding the joined 807-name list but only its own files, so mosaic and copy_paste
draw from one source by construction. Batches are dataset-pure, which is what makes the batch-global
`target_scores_sum` in `v8DetectionLoss` well defined, and it lets the cls BCE see only the batch's owning class slice
instead of teaching every other source's classes as absent.

Sampling is a two-stage draw per step: a source by localization hardness, then images within it by repeat factor.
The first epoch samples sources uniformly. Later epochs use the preceding epoch's weighted L1 losses.
"""

from __future__ import annotations

import math
import shutil
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torchvision.ops.misc import FrozenBatchNorm2d

from ultralytics.data import YOLOConcatDataset, build_dataloader
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model

DETECTION_AREA_RANGES = {
    "tiny": [0, 10**2],
    "small": [0, 32**2],
    "medium": [32**2, 96**2],
    "large": [96**2, 1e5**2],
}


def _replace_batch_norm(module: torch.nn.Module, frozen: bool) -> int:
    """Replace descendant BatchNorm layers with frozen normalization or GroupNorm.

    Args:
        module (torch.nn.Module): Root module whose descendants are converted.
        frozen (bool): Whether to use FrozenBatchNorm2d instead of GroupNorm.

    Returns:
        (int): Number of replaced layers.
    """
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            norm = (
                FrozenBatchNorm2d(child.num_features, eps=child.eps)
                if frozen
                else torch.nn.GroupNorm(math.gcd(32, child.num_features), child.num_features, eps=child.eps)
            ).to(child.weight.device, child.weight.dtype)
            with torch.no_grad():
                norm.weight.copy_(child.weight)
                norm.bias.copy_(child.bias)
                if frozen:
                    norm.running_mean.copy_(child.running_mean)
                    norm.running_var.copy_(child.running_var)
            setattr(module, name, norm)
            replaced += 1
        else:
            replaced += _replace_batch_norm(child, frozen)
    return replaced


SOURCE_METRIC_KEYS = (
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/mAR(B)",
    *(f"metrics/m{kind}_{size}(B)" for size in DETECTION_AREA_RANGES for kind in ("AP", "AR")),
)
LOCALIZATION_METRIC_KEY = "metrics/localization_mAR(B)"


class PartitionedDetectionLoss:
    """Route each source-pure batch to a criterion with its native class count."""

    def __init__(self, model: torch.nn.Module, slices: dict[str, tuple[int, int]], updates: int = 0) -> None:
        """Build one criterion per source label space.

        Args:
            model (torch.nn.Module): Detection model used to construct native criteria.
            slices (dict[str, tuple[int, int]]): Ordered source class bounds.
            updates (int): Completed end-to-end loss schedule updates.
        """
        self.criteria = {}
        for index, source in enumerate(slices):
            model.set_class_source(index)
            criterion = model.init_criterion()
            if hasattr(criterion, "updates"):
                criterion.updates = updates - 1
                criterion.update()
            self.criteria[source] = criterion
        model.set_class_source(None)
        self.source = next(iter(slices))

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate loss in the active source label space."""
        return self.criteria[self.source](preds, batch)

    def update(self) -> None:
        """Advance every source criterion through the shared epoch schedule."""
        for criterion in self.criteria.values():
            if hasattr(criterion, "update"):
                criterion.update()


# Dataset-pure batches follow UniDet's source buckets:
# https://github.com/xingyizhou/UniDet/blob/94cd0e8612e558c1dff64d2928bc969856c9a802/unidet/data/multi_dataset_dataloader.py#L203-L224
class QuotaBatchSampler(Sampler):
    """Yield dataset-pure index batches, source by quota and images by repeat factor.

    Every rank draws the same source for a given step from a shared generator, then takes its own slice of the global
    batch. Ranks therefore agree on the owning class slice while still seeing different images.

    Attributes:
        starts (list[int]): Index offset of each source inside the concatenated dataset.
        quota (np.ndarray): Per-source draw probability.
        cdfs (list[np.ndarray]): Per-source cumulative image weights, sampled by inverse transform.
        batches (int): Steps per epoch, one epoch being one pass worth of images.
        epoch (int): Passes taken so far, added to the seed so repeat passes differ.
    """

    def __init__(
        self,
        counts: list[int],
        weights: list[np.ndarray],
        batch: int,
        world_size: int,
        rank: int,
        seed: int,
    ):
        """Initialize the sampler from per-source image counts and per-image weights, at the global batch size.

        Args:
            counts (list[int]): Image count per source, in concatenation order.
            weights (list[np.ndarray]): Per-image draw probability inside each source.
            batch (int): Global batch size, summed over ranks.
            world_size (int): Number of ranks sharing a global batch.
            rank (int): This process's rank.
            seed (int): Base seed, shared by every rank so they agree on the source per step.
        """
        self.starts = np.cumsum([0, *counts[:-1]]).tolist()
        self.counts = np.array(counts, dtype=np.float64)
        self.quota = np.full(len(counts), 1 / len(counts))
        self.cdfs = [np.cumsum(w) for w in weights]  # sampling a p vector directly recumsums it every step
        self.batch, self.world_size, self.rank, self.seed = batch, world_size, rank, seed
        self.batches = max(1, sum(counts) // batch)
        self.epoch = 0

    def __len__(self) -> int:
        """Return steps per epoch."""
        return self.batches

    def set_epoch(self, epoch: int) -> None:
        """Set the pass index every rank draws from, so DDP ranks stay on the same source per step."""
        self.epoch = epoch

    def update_loss_quota(self, losses: np.ndarray) -> None:
        """Set source probabilities from the latest per-source L1 losses."""
        # Plain-Det paper Eq. 9 uses source mass L * sqrt(N):
        # https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00763.pdf#page=9
        weights = losses * np.sqrt(self.counts)
        self.quota = weights / weights.sum()

    def __iter__(self):
        """Draw a source then a dataset-pure batch of images, advancing the epoch so repeat passes differ."""
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        per_rank = self.batch // self.world_size
        for _ in range(self.batches):
            d = rng.choice(len(self.quota), p=self.quota)
            # With replacement, since a batch is a vanishing fraction of a source
            idx = np.searchsorted(self.cdfs[d], rng.random(self.batch) * self.cdfs[d][-1])
            yield (idx[self.rank * per_rank : (self.rank + 1) * per_rank] + self.starts[d]).tolist()


def source_stats(dataset, lo: int, hi: int, t: float) -> np.ndarray:
    """Return per-image draw probabilities for one source.

    f(c) is the image frequency of class c inside this source, r(c) = max(1, sqrt(t / f(c))), and an image takes the
    largest r(c) over the classes it holds, so the rarest class sets the weight. Backgrounds keep r = 1.

    Args:
        dataset (YOLODataset): The source dataset, labels already loaded.
        lo (int): First class id of the source's slice in the joined label space.
        hi (int): One past the last class id of the slice.
        t (float): Repeat-factor threshold, classes rarer than this are oversampled.

    Returns:
        (np.ndarray): Per-image draw probability, summing to 1.
    """
    labels = dataset.labels
    counts = np.zeros(hi - lo, dtype=np.float64)
    for lb in labels:
        counts[np.unique(lb["cls"].astype(np.int64).ravel() - lo)] += 1
    if not t:
        return np.full(len(labels), 1 / len(labels))
    # LVIS repeat-factor sampling:
    # https://github.com/facebookresearch/detectron2/blob/b4a4a3bd136852dae5fb1de37978dee412653e31/detectron2/data/samplers/distributed_sampler.py#L159-L209
    r_c = np.maximum(1.0, np.sqrt(t / np.maximum(counts / len(labels), 1e-12)))
    r = np.fromiter(
        (r_c[np.unique(lb["cls"].astype(np.int64).ravel() - lo)].max(initial=1.0) for lb in labels),
        dtype=np.float64,
        count=len(labels),
    )
    return r / r.sum()


class EnterpriseDetMetrics(DetMetrics):
    """Calculate source metrics and use their unweighted mean as the primary detection metric."""

    def __init__(self, sources: list[str]) -> None:
        """Initialize aggregate and per-source detection metrics.

        Args:
            sources (list[str]): Source names in dataset order.
        """
        super().__init__()
        self.source_metrics = {name: DetMetrics() for name in sources}
        self.coco_results = {}
        self.localization_results = {}

    @property
    def keys(self) -> list[str]:
        """Return macro and per-source metric keys."""
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            *SOURCE_METRIC_KEYS,
            LOCALIZATION_METRIC_KEY,
            "metrics/macro_mAP50-95(B)",
            *(f"metrics/{name}/{key[8:]}" for name in self.source_metrics for key in SOURCE_METRIC_KEYS),
            *(f"metrics/{name}/{LOCALIZATION_METRIC_KEY[8:]}" for name in self.source_metrics),
        ]

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """Process aggregate and source statistics."""
        stats = super().process(save_dir, plot, on_plot)
        for metric in self.source_metrics.values():
            metric.process()
        return stats

    def mean_results(self) -> list[float]:
        """Return the unweighted mean of source-level results."""
        source_mean = np.mean([metric.mean_results() for metric in self.source_metrics.values()], axis=0)
        if self.coco_results:
            source_mean[2:] = [self._macro(key) for key in SOURCE_METRIC_KEYS[:2]]
        return source_mean.tolist()

    def _macro(self, key: str) -> float:
        """Return the source mean for a defined COCO metric."""
        values = [metrics[key] for metrics in self.coco_results.values() if metrics[key] >= 0]
        return float(np.mean(values)) if values else -1.0

    @property
    def fitness(self) -> float:
        """Return source-macro mAP50-95 as fitness."""
        return self.mean_results()[-1]

    @property
    def results_dict(self) -> dict[str, float]:
        """Return macro detection metrics and each source's AP."""
        precision, recall, _, _ = self.mean_results()
        macro = {key: self._macro(key) for key in SOURCE_METRIC_KEYS}
        return {
            "metrics/precision(B)": precision,
            "metrics/recall(B)": recall,
            **macro,
            LOCALIZATION_METRIC_KEY: float(np.mean(list(self.localization_results.values()))),
            "metrics/macro_mAP50-95(B)": macro["metrics/mAP50-95(B)"],
            **{
                f"metrics/{name}/{key[8:]}": value
                for name, metrics in self.coco_results.items()
                for key, value in metrics.items()
            },
            **{
                f"metrics/{name}/{LOCALIZATION_METRIC_KEY[8:]}": value
                for name, value in self.localization_results.items()
            },
            "fitness": macro["metrics/mAP50-95(B)"],
        }


class EnterpriseDetectionValidator(DetectionValidator):
    """Evaluate every Enterprise image only against its source label space."""

    def __init__(self, *args, slices: dict[str, tuple[int, int]], **kwargs) -> None:
        """Initialize source-aware validation.

        Args:
            *args (Any): Positional arguments forwarded to DetectionValidator.
            slices (dict[str, tuple[int, int]]): Source names mapped to class-id bounds.
            **kwargs (Any): Keyword arguments forwarded to DetectionValidator.
        """
        super().__init__(*args, **kwargs)
        self.slices = slices
        self.metrics = EnterpriseDetMetrics(list(slices))

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics, source paths, and the validation criterion."""
        self.args.save_json = True
        super().init_metrics(model)
        self.source_model = model
        self.source_indices = {source: index for index, source in enumerate(self.slices)}
        check_requirements("faster-coco-eval>=1.7.0")
        for metric in self.metrics.source_metrics.values():
            metric.names = model.names
            metric.clear_stats()
            metric.clear_image_metrics()
        roots = [Path(path) for path in self.data[self.args.split]]
        self.source_roots = dict(zip(self.slices, roots))
        assert len(self.source_roots) == len(self.slices), "Enterprise offsets and validation dirs disagree"
        datasets = getattr(self.dataloader.dataset, "datasets", (self.dataloader.dataset,))
        self.source_image_ids = {name: [] for name in self.slices}
        for image_id, path in enumerate(path for dataset in datasets for path in dataset.im_files):
            name = self._source(str(path))
            self.source_image_ids[name].append(image_id)
        if self.training:
            self.model = model
            if getattr(model, "criterion", None) is None:
                model.criterion = model.init_criterion()

    def finalize_metrics(self) -> None:
        """Finalize source metrics and release merged evaluation classifiers."""
        super().finalize_metrics()
        if self.training:
            unwrap_model(self.model).model[-1].clear_class_source_cache()

    def _source(self, im_file: str) -> str:
        """Return the source owning an image path."""
        path = Path(im_file)
        return next(name for name, root in self.source_roots.items() if root == path.parent or root in path.parents)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply each image's complete source slice to validation classification loss."""
        batch = super().preprocess(batch)
        self.batch_sources = [self._source(path) for path in batch["im_file"]]
        self.source_model.set_class_source([self.source_indices[name] for name in self.batch_sources])
        if self.training:
            weights = torch.zeros(len(self.batch_sources), 1, self.nc, device=self.device)
            for i, name in enumerate(self.batch_sources):
                lo, hi = self.slices[name]
                weights[i, :, lo:hi] = 1.0
            self.model.criterion.class_weights = weights
        return batch

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Run NMS on each image using only classes from its source."""
        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        if self.end2end:
            for i, name in enumerate(self.batch_sources):
                lo, hi = self.slices[name]
                preds[i, (preds[i, :, 5] < lo) | (preds[i, :, 5] >= hi), 4] = 0
            return super().postprocess(preds)
        outputs = [None] * len(self.batch_sources)
        for name in dict.fromkeys(self.batch_sources):
            indices = [i for i, source in enumerate(self.batch_sources) if source == name]
            lo, hi = self.slices[name]
            source_preds = torch.cat((preds[indices, :4], preds[indices, 4 + lo : 4 + hi]), 1)
            for i, output in zip(indices, super().postprocess(source_preds)):
                output["cls"] += lo
                outputs[i] = output
        return outputs

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Update aggregate and owning-source metrics from filtered predictions."""
        start = len(self.metrics.stats["tp"])
        super().update_metrics(preds, batch)
        for i, name in enumerate(self.batch_sources):
            self.metrics.source_metrics[name].update_stats(
                {
                    **{key: values[start + i] for key, values in self.metrics.stats.items()},
                    "im_name": f"{name}/{Path(batch['im_file'][i]).name}",
                }
            )
        for values in self.metrics.stats.values():
            del values[start:]
        self.metrics.clear_image_metrics()

    def gather_stats(self) -> None:
        """Gather aggregate and source statistics from every validation rank."""
        super().gather_stats()
        source_stats = {
            name: {"stats": metric.stats, "image_metrics": metric.box.image_metrics}
            for name, metric in self.metrics.source_metrics.items()
        }
        if RANK == 0:
            gathered = [None] * dist.get_world_size()
            dist.gather_object(source_stats, gathered, 0)
            for name, metric in self.metrics.source_metrics.items():
                metric.stats = {
                    key: [item for rank_stats in gathered for item in rank_stats[name]["stats"][key]]
                    for key in metric.stats
                }
                metric.clear_image_metrics()
                for rank_stats in gathered:
                    metric.box.image_metrics.update(rank_stats[name]["image_metrics"])
        elif RANK > 0:
            dist.gather_object(source_stats, None, 0)
            for metric in self.metrics.source_metrics.values():
                metric.clear_stats()
                metric.clear_image_metrics()
        if RANK in {-1, 0}:
            self.metrics.stats = {
                key: [item for metric in self.metrics.source_metrics.values() for item in metric.stats[key]]
                for key in self.metrics.stats
            }
            self.metrics.box.image_metrics = {
                image: values
                for metric in self.metrics.source_metrics.values()
                for image, values in metric.box.image_metrics.items()
            }

    def get_stats(self) -> dict[str, Any]:
        """Return source-aware metrics without merged-dataset COCO evaluation."""
        from faster_coco_eval import COCO

        self.coco_gt = COCO(self.gdict)
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.metrics.coco_results = {}
        self.metrics.localization_results = {}
        predictions = self.coco_gt.loadRes(self._prepare_coco_predictions()) if self.jdict else self.jdict
        for name, (lo, hi) in self.slices.items():
            image_ids = self.source_image_ids[name]
            source = self.coco_evaluate(
                {},
                predictions,
                self.coco_gt,
                image_ids=image_ids,
                category_ids=list(range(lo + 1, hi + 1)),
                ranges=DETECTION_AREA_RANGES,
            )
            self.metrics.coco_results[name] = {key: source[key] for key in SOURCE_METRIC_KEYS}
            self.metrics.localization_results[name] = (
                self.coco_evaluate({}, predictions, self.coco_gt, image_ids=image_ids, class_agnostic=True)[
                    "metrics/mAR(B)"
                ]
                if self.jdict
                else 0.0
            )
        stats = self.metrics.results_dict
        self.metrics.clear_stats()
        for metric in self.metrics.source_metrics.values():
            metric.clear_stats()
            metric.clear_image_metrics()
        return stats

    def print_results(self) -> None:
        """Print source-macro results followed by AP for every source."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * 4
        LOGGER.info(pf % ("all", self.seen, self.metrics.nt_per_class.sum(), *self.metrics.mean_results()))
        for name, metrics in self.metrics.coco_results.items():
            LOGGER.info(
                f"{name:>22s} mAP50-95 {metrics['metrics/mAP50-95(B)']:.4g} "
                f"localization AR100 {self.metrics.localization_results[name]:.4g}"
            )


class EnterpriseDetectionTrainer(DetectionTrainer):
    """Detection trainer using dataset-pure batches and one classifier per source label space.

    Attributes:
        slices (dict): Source name to (lo, hi) class-id bounds, from the `offsets` block of the corpus data.yaml.
        source_of (dict): Image directory to source name, how a batch reports which slice owns it.
    """

    def set_model_attributes(self) -> None:
        """Set source heads, normalization, and loss-aware sampling."""
        super().set_model_attributes()
        model = unwrap_model(self.model)
        backbone_layers = len(model.yaml["backbone"])
        frozen_norms = sum(_replace_batch_norm(layer, True) for layer in model.model[:backbone_layers])
        group_norms = sum(_replace_batch_norm(layer, False) for layer in model.model[backbone_layers:])
        LOGGER.info(f"Enterprise normalization: {frozen_norms} frozen backbone BN, {group_norms} detector GN")
        bounds = [*sorted(self.data["offsets"].values()), self.data["nc"]]
        self.slices = {k: (lo, hi) for (k, lo), hi in zip(self.data["offsets"].items(), bounds[1:])}
        self.source_indices = {source: index for index, source in enumerate(self.slices)}
        model.model[-1].partition_classifiers(list(self.slices.values()))
        self.source_loss_stats = torch.zeros(2, len(self.slices), device=self.device)
        self.source_l1_losses = torch.ones(len(self.slices), device=self.device)
        self.sampling_metrics = {}
        self.add_callback("on_train_batch_end", self._record_source_loss)
        self.add_callback("on_train_epoch_end", self._update_loss_quota)

    def build_dataset(self, img_path: str | list, mode: str = "train", batch: int | None = None):
        """Build one dataset per source in train mode, and the plain merged dataset in val mode."""
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        paths = img_path if isinstance(img_path, list) else [img_path]
        return YOLOConcatDataset([super(EnterpriseDetectionTrainer, self).build_dataset(p, mode, batch) for p in paths])

    def get_validator(self):
        """Return the source-aware Enterprise validator."""
        return EnterpriseDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            slices=self.slices,
        )

    def resume_training(self, ckpt: dict | None) -> None:
        """Restore source-specific criteria after the generic end-to-end resume setup.

        Args:
            ckpt (dict | None): Training checkpoint state.
        """
        super().resume_training(ckpt)
        if ckpt is not None and self.resume:
            losses = np.array([ckpt["train_metrics"][f"sampling/{name}/l1_loss"] for name in self.slices])
            self.source_l1_losses.copy_(torch.from_numpy(losses).to(self.device))
            self.quota_sampler.update_loss_quota(losses)
        model = unwrap_model(self.model)
        updates = ckpt.get("epoch", -1) + 1 if ckpt is not None and self.resume else 0
        model.criterion = PartitionedDetectionLoss(model, self.slices, updates)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Return the quota-sampled dataset-pure loader for training, and the default loader for validation."""
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        assert len(self.slices) == len(dataset.datasets), "corpus data.yaml offsets and train dirs disagree"
        repeat_sources = set(filter(None, getattr(self.args, "repeat_sources", "").split(",")))
        assert repeat_sources <= self.slices.keys(), (
            f"unknown repeat_sources: {sorted(repeat_sources - self.slices.keys())}"
        )
        self.source_of, weights = {}, []
        for name, d in zip(self.slices, dataset.datasets):
            self.source_of[str(Path(d.im_files[0]).parent)] = name
            weights.append(
                source_stats(
                    d, *self.slices[name], self.args.repeat_t if not repeat_sources or name in repeat_sources else 0
                )
            )

        world = self.world_size or 1
        sampler = QuotaBatchSampler(
            [len(d) for d in dataset.datasets],
            weights,
            batch_size * world,
            world,
            max(rank, 0),  # rank is LOCAL_RANK, which is -1 outside DDP
            self.args.seed,
        )
        self.quota_sampler = sampler
        LOGGER.info(
            f"{len(dataset.datasets)} sources, {len(dataset):,} images, {sampler.batches:,} steps/epoch, "
            f"quota {np.round(sampler.quota, 4).tolist()}, RFS {sorted(repeat_sources) if repeat_sources else 'all'}"
        )
        return build_dataloader(
            dataset, batch_size, self.args.workers, rank=rank, device=self.device, batch_sampler=sampler
        )

    def _record_source_loss(self, _) -> None:
        """Accumulate the active source weighted L1 loss after each training batch."""
        index = self.source_indices[self.current_source]
        self.source_loss_stats[0, index] += self.loss_items["l1_loss"].detach()
        self.source_loss_stats[1, index] += 1

    def _update_loss_quota(self, _) -> None:
        """Update every DDP rank with the same Plain-Det source quota."""
        batches = self.source_loss_stats[1].cpu().numpy().copy()
        if dist.is_initialized():
            dist.all_reduce(self.source_loss_stats)
        seen = self.source_loss_stats[1].bool()
        self.source_l1_losses[seen] = self.source_loss_stats[0, seen] / self.source_loss_stats[1, seen]
        losses = self.source_l1_losses.cpu().numpy()
        quota = self.quota_sampler.quota.copy()
        self.sampling_metrics = {
            key: value
            for name, loss, probability, count in zip(self.slices, losses, quota, batches)
            for key, value in {
                f"sampling/{name}/l1_loss": loss,
                f"sampling/{name}/target_probability": probability,
                f"sampling/{name}/realized_batches": count,
                f"sampling/{name}/realized_probability": count / batches.sum(),
            }.items()
        }
        self.quota_sampler.update_loss_quota(losses)
        self.source_loss_stats.zero_()
        LOGGER.info(
            f"source L1 loss {np.round(losses, 4).tolist()}, next quota "
            f"{np.round(self.quota_sampler.quota, 4).tolist()}"
        )

    def save_metrics(self, metrics: dict[str, float]) -> None:
        """Save validation and source-sampling metrics."""
        self.metrics.update(self.sampling_metrics)
        super().save_metrics({**metrics, **self.sampling_metrics})

    def save_model(self) -> bool:
        """Save the regular checkpoint and fixed Enterprise transfer snapshots."""
        saved = super().save_model()
        if self.epoch + 1 in {2, 10, 20}:
            shutil.copyfile(self.last, self.wdir / f"epoch{self.epoch + 1}.pt")
        return saved

    def preprocess_batch(self, batch: dict) -> dict:
        """Point the cls loss at this batch's owning class slice, then preprocess as usual."""
        self.current_source = self.source_of[str(Path(batch["im_file"][0]).parent)]
        name = self.current_source
        lo, _ = self.slices[name]
        batch["cls"].sub_(lo)
        model = unwrap_model(self.model)
        model.set_class_source(self.source_indices[name])
        batch = super().preprocess_batch(batch)
        model.criterion.source = name
        return batch

    def plot_training_labels(self):
        """Skip the label histogram, unreadable over 807 classes and 14.7M boxes and minutes of pandas work."""
