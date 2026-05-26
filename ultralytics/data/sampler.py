# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from torch.utils.data import Sampler

if TYPE_CHECKING:
    from ultralytics.data.dataset import YOLOConcatDataset


def iter_dataset_labels(dataset) -> list[dict]:
    """Return label dicts for a YOLODataset or YOLOConcatDataset."""
    from ultralytics.data.dataset import YOLOConcatDataset

    if isinstance(dataset, YOLOConcatDataset):
        labels = []
        for d in dataset.datasets:
            labels.extend(d.labels)
        return labels
    return dataset.labels


def get_concat_index_pools(dataset) -> list[list[int]]:
    """Return global index lists for each sub-dataset in a YOLOConcatDataset."""
    from ultralytics.data.dataset import YOLOConcatDataset

    if not isinstance(dataset, YOLOConcatDataset):
        raise TypeError(f"Expected YOLOConcatDataset, got {type(dataset).__name__}")
    pools, offset = [], 0
    for d in dataset.datasets:
        n = len(d)
        pools.append(list(range(offset, offset + n)))
        offset += n
    return pools


def normalize_fractions(fractions: list[float]) -> list[float]:
    """Normalize dataset fractions to sum to 1.0."""
    total = float(sum(fractions))
    if total <= 0:
        raise ValueError(f"dataset_fractions must sum to > 0, got {fractions}")
    return [float(f) / total for f in fractions]


def allocate_batch_counts(fractions: list[float], batch_size: int) -> list[int]:
    """Split ``batch_size`` into per-dataset counts proportional to ``fractions`` (sum equals batch_size)."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    fracs = normalize_fractions(fractions)
    counts = [batch_size * f for f in fracs]
    ints = [int(c) for c in counts]
    remainder = batch_size - sum(ints)
    if remainder:
        order = sorted(range(len(fracs)), key=lambda i: counts[i] - ints[i], reverse=True)
        for i in order[:remainder]:
            ints[i] += 1
    return ints


class ProportionalBatchSampler(Sampler[list[int]]):
    """Sample each batch from multiple datasets by configurable fractions.

    Batch size is fixed per run (e.g. set by Katib as the ``batch`` hyperparameter). Each sub-dataset is sampled
    with replacement so small sources can be oversampled when their fraction is high.

    Args:
        index_pools (list[list[int]]): Global dataset indices per source (e.g. from ``YOLOConcatDataset``).
        fractions (list[float]): Relative amount per source; normalized to sum to 1.
        batch_size (int): Images per batch for this training run.
        seed (int): Base random seed; combined with epoch in ``set_epoch``.
        rank (int): DDP rank.
        world_size (int): DDP world size.
    """

    def __init__(
        self,
        index_pools: list[list[int]],
        fractions: list[float],
        batch_size: int = 16,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if len(index_pools) != len(fractions):
            raise ValueError(
                f"dataset_fractions length {len(fractions)} must match number of train datasets {len(index_pools)}"
            )
        empty = [i for i, p in enumerate(index_pools) if not p]
        if empty:
            raise ValueError(f"Empty dataset index pool for dataset index(es) {empty}")
        self.index_pools = index_pools
        self.fractions = normalize_fractions(fractions)
        self.batch_size = int(batch_size)
        self.seed = seed
        self.rank = rank
        self.world_size = max(world_size, 1)
        self.epoch = 0
        self.dataset_size = sum(len(p) for p in index_pools)

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling across DDP ranks."""
        self.epoch = epoch

    def _rng(self) -> random.Random:
        return random.Random(self.seed + self.epoch)

    def _sample_batch(self, rng: random.Random) -> list[int]:
        """Build one batch with per-source counts from ``fractions``."""
        counts = allocate_batch_counts(self.fractions, self.batch_size)
        batch = []
        for pool, n in zip(self.index_pools, counts):
            if n:
                batch.extend(rng.choices(pool, k=n))
        rng.shuffle(batch)
        return batch

    def __iter__(self):
        """Yield batch index lists indefinitely (for InfiniteDataLoader)."""
        rng = self._rng()
        batch_count = 0
        while True:
            batch = self._sample_batch(rng)
            if batch_count % self.world_size == self.rank:
                yield batch
            batch_count += 1

    def __len__(self) -> int:
        """Approximate batches per epoch per rank (progress bar / warmup)."""
        return max(1, math.ceil(self.dataset_size / self.batch_size / self.world_size))
