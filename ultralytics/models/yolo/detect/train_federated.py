# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Detection training over a corpus of merged sources that keep separate label spaces.

Each source is its own `YOLODataset` holding the joined 807-name list but only its own files, so mosaic and copy_paste
draw from one source by construction. Batches are dataset-pure, which is what makes the batch-global
`target_scores_sum` in `v8DetectionLoss` well defined, and it lets the cls BCE see only the batch's owning class slice
instead of teaching every other source's classes as absent.

Sampling is a two-stage draw per step: a source by quota `p(d) = N_d^quota_alpha`, then images within it by repeat
factor. `quota_alpha`, `repeat_t` and `fed_k` come from the recipe profile, not from this file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler

from ultralytics.data import YOLOConcatDataset, build_dataloader
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


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
        alpha: float,
        world_size: int,
        rank: int,
        seed: int,
    ):
        """Initialize the sampler from per-source image counts and per-image weights, at the global batch size.

        Args:
            counts (list[int]): Image count per source, in concatenation order.
            weights (list[np.ndarray]): Per-image draw probability inside each source.
            batch (int): Global batch size, summed over ranks.
            alpha (float): Quota temper exponent.
            world_size (int): Number of ranks sharing a global batch.
            rank (int): This process's rank.
            seed (int): Base seed, shared by every rank so they agree on the source per step.
        """
        self.starts = np.cumsum([0, *counts[:-1]]).tolist()
        tempered = np.array(counts, dtype=np.float64) ** alpha
        self.quota = tempered / tempered.sum()
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


def source_stats(dataset, lo: int, hi: int, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Return per-image draw probabilities and per-class image counts for one source.

    f(c) is the image frequency of class c inside this source, r(c) = max(1, sqrt(t / f(c))), and an image takes the
    largest r(c) over the classes it holds, so the rarest class sets the weight. Backgrounds keep r = 1.

    Args:
        dataset (YOLODataset): The source dataset, labels already loaded.
        lo (int): First class id of the source's slice in the joined label space.
        hi (int): One past the last class id of the slice.
        t (float): Repeat-factor threshold, classes rarer than this are oversampled.

    Returns:
        (np.ndarray): Per-image draw probability, summing to 1.
        (np.ndarray): Per-class image count over the slice, length hi - lo.
    """
    labels = dataset.labels
    counts = np.zeros(hi - lo, dtype=np.float64)
    for lb in labels:
        counts[np.unique(lb["cls"].astype(np.int64).ravel() - lo)] += 1
    r_c = np.maximum(1.0, np.sqrt(t / np.maximum(counts / len(labels), 1e-12)))
    r = np.fromiter(
        (r_c[np.unique(lb["cls"].astype(np.int64).ravel() - lo)].max(initial=1.0) for lb in labels),
        dtype=np.float64,
        count=len(labels),
    )
    return r / r.sum(), counts


class FederatedDetectionTrainer(DetectionTrainer):
    """Detection trainer whose train loader emits dataset-pure batches and whose cls loss sees only the owning slice.

    The mask rides the existing per-class `class_weights` multiply in `v8DetectionLoss`, recomputed every step. That
    hook is exclusive here because the corpus recipe sets `cls_pw: 0.0`, so `set_class_weights` writes nothing.

    Attributes:
        slices (dict): Source name to (lo, hi) class-id bounds, from the `offsets` block of the corpus data.yaml.
        source_of (dict): Image directory to source name, how a batch reports which slice owns it.
        neg_weights (dict): Source name to per-class sampling weight for federated negatives, image count^0.5.
    """

    def build_dataset(self, img_path: str | list, mode: str = "train", batch: int | None = None):
        """Build one dataset per source in train mode, and the plain merged dataset in val mode."""
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        paths = img_path if isinstance(img_path, list) else [img_path]
        return YOLOConcatDataset([super(FederatedDetectionTrainer, self).build_dataset(p, mode, batch) for p in paths])

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Return the quota-sampled dataset-pure loader for training, and the default loader for validation."""
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        assert self.args.cls_pw == 0.0, "federated cls masking owns class_weights, so cls_pw must stay 0.0"
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        bounds = [*sorted(self.data["offsets"].values()), self.data["nc"]]
        self.slices = {k: (lo, hi) for (k, lo), hi in zip(self.data["offsets"].items(), bounds[1:])}
        assert len(self.slices) == len(dataset.datasets), "corpus data.yaml offsets and train dirs disagree"
        self.source_of, self.neg_weights, weights = {}, {}, []
        for name, d in zip(self.slices, dataset.datasets):
            self.source_of[str(Path(d.im_files[0]).parent)] = name
            w, counts = source_stats(d, *self.slices[name], self.args.repeat_t)
            weights.append(w)
            self.neg_weights[name] = np.sqrt(counts)

        world = self.world_size or 1
        sampler = QuotaBatchSampler(
            [len(d) for d in dataset.datasets],
            weights,
            batch_size * world,
            self.args.quota_alpha,
            world,
            max(rank, 0),  # rank is LOCAL_RANK, which is -1 outside DDP
            self.args.seed,
        )
        self.rng = np.random.default_rng(self.args.seed)
        LOGGER.info(
            f"{len(dataset.datasets)} sources, {len(dataset):,} images, {sampler.batches:,} steps/epoch, "
            f"quota {np.round(sampler.quota, 4).tolist()}"
        )
        return build_dataloader(
            dataset, batch_size, self.args.workers, rank=rank, device=self.device, batch_sampler=sampler
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """Point the cls loss at this batch's owning class slice, then preprocess as usual."""
        mask = self._fed_mask(batch)
        batch = super().preprocess_batch(batch)
        model = unwrap_model(self.model)
        if getattr(model, "criterion", None) is None:
            model.criterion = model.init_criterion()
        model.criterion.class_weights = mask.to(self.device).view(1, 1, -1)
        return batch

    def _fed_mask(self, batch: dict) -> torch.Tensor:
        """Return an nc-long CPU mask over the batch's owning slice, subsampled when the slice exceeds `fed_k`.

        Every class present in the batch is always kept, topped up with negatives drawn without replacement at image
        count^0.5 until the kept set reaches `fed_k`. The kept count lands above `fed_k` when the batch alone holds
        that many classes, and below it when the slice runs short of drawable negatives. Columns outside the kept set
        get no gradient. The draw is per rank, as in Detic, so a class masked on one rank still learns from another.
        """
        name = self.source_of[str(Path(batch["im_file"][0]).parent)]
        lo, hi = self.slices[name]
        mask = torch.zeros(self.data["nc"])
        if hi - lo <= self.args.fed_k:
            mask[lo:hi] = 1.0
            return mask
        present = torch.unique(batch["cls"].int()).numpy()  # still on cpu, super() has not moved the batch yet
        w = self.neg_weights[name].copy()
        w[present - lo] = 0.0
        mask[present] = 1.0
        k = min(self.args.fed_k - len(present), int((w > 0).sum()))
        if k > 0:
            mask[self.rng.choice(len(w), k, replace=False, p=w / w.sum()) + lo] = 1.0
        return mask

    def plot_training_labels(self):
        """Skip the label histogram, unreadable over 807 classes and 14.7M boxes and minutes of pandas work."""
