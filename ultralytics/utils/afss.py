# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from copy import copy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.utils import LOGGER, LOCAL_RANK, RANK


class AFSSScheduler:
    """Schedule training images by their per-image learning sufficiency."""

    # Constants from AFSS: https://arxiv.org/abs/2603.17684
    EASY_THRESHOLD = 0.85
    MODERATE_THRESHOLD = 0.55
    EASY_RATE = 0.02
    MODERATE_RATE = 0.40
    EASY_REVIEW_GAP = 10
    MODERATE_REVIEW_GAP = 3
    FULL_FINAL_EPOCHS = 10

    def __init__(self, num_images: int, warmup_epochs: float = 3.0, seed: int = 0):
        """Initialize an AFSS schedule with one compact state vector per image."""
        self.num_images = num_images
        self.warmup_epochs = warmup_epochs
        self.seed = seed
        self.precision = np.zeros(num_images, dtype=np.float32)
        self.recall = np.zeros(num_images, dtype=np.float32)
        self.last_seen = np.full(num_images, -1, dtype=np.int32)

    def sample_indices(self, epoch: int) -> list[int]:
        """Return hard images and a budgeted review set for one epoch."""
        rng = np.random.default_rng(self.seed + epoch)
        score = np.minimum(self.precision, self.recall)
        easy = np.flatnonzero(score > self.EASY_THRESHOLD)
        moderate = np.flatnonzero((score >= self.MODERATE_THRESHOLD) & (score <= self.EASY_THRESHOLD))
        hard = np.flatnonzero(score < self.MODERATE_THRESHOLD)
        selected = hard.tolist()

        easy_budget = max(1, round(self.EASY_RATE * len(easy))) if len(easy) else 0
        forced_easy = easy[(epoch - 1 - self.last_seen[easy]) >= self.EASY_REVIEW_GAP]
        forced_easy_quota = min(len(forced_easy), easy_budget // 2)
        if forced_easy_quota:
            selected.extend(rng.choice(forced_easy, forced_easy_quota, replace=False).tolist())
        remaining_easy = np.setdiff1d(easy, selected, assume_unique=False)
        random_easy_quota = min(easy_budget - forced_easy_quota, len(remaining_easy))
        if random_easy_quota:
            selected.extend(rng.choice(remaining_easy, random_easy_quota, replace=False).tolist())

        moderate_budget = max(1, round(self.MODERATE_RATE * len(moderate))) if len(moderate) else 0
        forced_moderate = moderate[(epoch - 1 - self.last_seen[moderate]) >= self.MODERATE_REVIEW_GAP]
        selected.extend(forced_moderate.tolist())
        remaining_moderate = np.setdiff1d(moderate, forced_moderate, assume_unique=False)
        random_moderate_quota = min(max(moderate_budget - len(forced_moderate), 0), len(remaining_moderate))
        if random_moderate_quota:
            selected.extend(rng.choice(remaining_moderate, random_moderate_quota, replace=False).tolist())

        selected = sorted(set(selected))
        if not selected:
            selected = list(range(self.num_images))
        LOGGER.info(
            f"AFSS epoch {epoch}: {len(selected)}/{self.num_images} images "
            f"(hard={len(hard)}, moderate={len(moderate)}, easy={len(easy)})"
        )
        return selected

    def update_last_seen(self, indices: list[int], epoch: int) -> None:
        """Record the epoch in which each selected image was trained."""
        if indices:
            self.last_seen[np.asarray(indices, dtype=np.intp)] = epoch

    def update_metrics(self, image_metrics: dict[str, dict], index_by_name: dict[str, int]) -> None:
        """Refresh precision and recall from the validator's per-image metrics."""
        for filename, metrics in image_metrics.items():
            index = index_by_name.get(str(filename))
            if index is None:
                index = index_by_name.get(Path(filename).name)
            if index is not None:
                self.precision[index] = metrics.get("precision", 0.0)
                self.recall[index] = metrics.get("recall", 0.0)

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return the scheduler state for checkpoint sidecar storage."""
        return {"precision": self.precision, "recall": self.recall, "last_seen": self.last_seen}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Restore scheduler state when resuming a run."""
        if not all(np.asarray(state[key]).shape == (self.num_images,) for key in ("precision", "recall", "last_seen")):
            raise ValueError("AFSS state does not match the current dataset")
        self.precision = np.asarray(state["precision"], dtype=np.float32)
        self.recall = np.asarray(state["recall"], dtype=np.float32)
        self.last_seen = np.asarray(state["last_seen"], dtype=np.int32)


def _unwrap_dataset(dataset):
    """Return the underlying image dataset from a loader wrapper."""
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def afss_on_epoch_start(trainer):
    """Select the next AFSS training subset before batches are requested."""
    if not hasattr(trainer, "afss_scheduler"):
        dataset = _unwrap_dataset(trainer.train_loader.dataset)
        trainer.afss_scheduler = AFSSScheduler(len(dataset), trainer.args.warmup_epochs, trainer.args.seed)
        trainer.afss_current_indices = list(range(len(dataset)))
        state_path = trainer.wdir / "afss_state.pt"
        if state_path.exists():
            try:
                state = torch.load(state_path, map_location="cpu", weights_only=False)
            except TypeError:  # PyTorch < 2.0
                state = torch.load(state_path, map_location="cpu")
            try:
                trainer.afss_scheduler.load_state_dict(state)
            except (KeyError, ValueError):
                LOGGER.warning("AFSS state does not match the current dataset; starting a new schedule.")

    epoch = trainer.epoch
    if epoch < trainer.afss_scheduler.warmup_epochs:
        return
    selected = (
        list(range(trainer.afss_scheduler.num_images))
        if trainer.epochs - epoch <= trainer.afss_scheduler.FULL_FINAL_EPOCHS
        else trainer.afss_scheduler.sample_indices(epoch)
    )

    if trainer.world_size > 1:
        payload = [selected if RANK == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        selected = payload[0]

    if selected != trainer.afss_current_indices:
        if trainer.world_size > 1:
            old_loader = trainer.train_loader
            if hasattr(old_loader, "close"):
                old_loader.close()
            trainer.train_loader = trainer.get_dataloader(
                trainer.data["train"],
                batch_size=trainer.batch_size // trainer.world_size,
                rank=LOCAL_RANK,
                mode="train",
                active_indices=selected,
            )
        else:
            dataset = _unwrap_dataset(trainer.train_loader.dataset)
            dataset.active_indices = selected
            trainer.train_loader.reset()
        trainer.afss_current_indices = selected

    trainer.nb = len(trainer.train_loader)
    LOGGER.info(f"AFSS training set: {len(selected)}/{trainer.afss_scheduler.num_images} images")


def afss_on_epoch_end(trainer):
    """Refresh image difficulty periodically and persist the selected-image clock."""
    if not hasattr(trainer, "afss_scheduler"):
        return
    epoch = trainer.epoch
    trainer.afss_scheduler.update_last_seen(trainer.afss_current_indices, epoch)
    warmup = math.ceil(trainer.afss_scheduler.warmup_epochs)
    if epoch >= warmup and (epoch - warmup) % 5 == 0 and trainer.epochs - epoch > trainer.afss_scheduler.FULL_FINAL_EPOCHS:
        afss_refresh_metrics(trainer)


def afss_refresh_metrics(trainer):
    """Run a quiet training-set validation pass to update AFSS difficulty scores."""
    loader = trainer.get_dataloader(
        trainer.data["train"],
        batch_size=trainer.batch_size // max(trainer.world_size, 1),
        rank=LOCAL_RANK,
        mode="val",
    )
    args = copy(trainer.args)
    args.plots = args.save_json = args.save_txt = args.verbose = False
    validator = trainer.get_validator().__class__(loader, save_dir=trainer.save_dir / "afss_train_eval", args=args)
    validator(trainer)
    if hasattr(loader, "close"):
        loader.close()

    if RANK in {-1, 0}:
        dataset = _unwrap_dataset(loader.dataset)
        index_by_name = {}
        for index, filename in enumerate(dataset.im_files):
            index_by_name.setdefault(str(filename), index)
            index_by_name.setdefault(Path(filename).name, index)
        trainer.afss_scheduler.update_metrics(validator.metrics.box.image_metrics, index_by_name)
        LOGGER.info(f"AFSS refreshed {len(validator.metrics.box.image_metrics)} image scores")
    if trainer.world_size > 1:
        payload = [trainer.afss_scheduler.state_dict() if RANK == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        trainer.afss_scheduler.load_state_dict(payload[0])


def afss_save_state(trainer):
    """Save AFSS state alongside model checkpoints."""
    if hasattr(trainer, "afss_scheduler") and RANK in {-1, 0}:
        torch.save(trainer.afss_scheduler.state_dict(), trainer.wdir / "afss_state.pt")
