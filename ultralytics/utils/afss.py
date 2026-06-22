# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
from copy import copy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.utils import LOGGER, LOCAL_RANK, RANK


class AFSSScheduler:
    """Anti-Forgetting Sampling Strategy (AFSS) scheduler for YOLO training.

    This scheduler partitions images into easy, moderate, and hard sets based on
    per-image precision and recall, then samples from each set according to a
    budget policy that prevents forgetting by periodically forcing long-unseen
    images back into the training batch.

    Attributes:
        num_images (int): Total number of images in the dataset.
        seed (int): Random seed for deterministic sampling.
        state (dict[int, dict]): Per-image state containing precision, recall, and last_seen_epoch.
    """

    def __init__(
        self,
        num_images: int,
        seed: int = 0,
        easy_thr: float = 0.85,
        easy_ratio: float = 0.02,
        moderate_thr: float = 0.55,
        moderate_ratio: float = 0.4,
        obj_counts: list[int] | None = None,
        min_objects: int = 1,
    ):
        """Initialize AFSSScheduler with the given dataset size and warmup configuration.

        Args:
            num_images (int): Total number of images in the dataset.
            seed (int): Random seed for deterministic sampling.
            easy_thr (float): Threshold above which an image is classified as easy (min(P, R) > easy_thr).
            easy_ratio (float): Fraction of easy images reviewed each epoch (continuous-review budget).
            moderate_thr (float): Threshold above which an image is classified as moderate (min(P, R) >= moderate_thr).
            moderate_ratio (float): Fraction of moderate images sampled each epoch.
            obj_counts (list[int], optional): Per-image GT object counts, indexed like the dataset. Used together
                with min_objects to force always-train images.
            min_objects (int): Images with <= this many GT objects are always trained (never skipped). Their
                per-image P/R is too coarse to trust (with 0-1 objects, recall is effectively binary), so they are
                easily misclassified as easy and dropped. Set to -1 to disable.
        """
        self.num_images = num_images
        self.seed = seed
        self.easy_thr = easy_thr
        self.easy_ratio = easy_ratio
        self.moderate_thr = moderate_thr
        self.moderate_ratio = moderate_ratio
        self.min_objects = min_objects
        # Indices that bypass difficulty-based sampling and are trained every epoch.
        if obj_counts is not None and min_objects >= 0:
            self.always_train = {i for i, c in enumerate(obj_counts) if c <= min_objects}
        else:
            self.always_train = set()
        self.state = {i: {"precision": 0.0, "recall": 0.0, "last_seen_epoch": -1} for i in range(num_images)}

    def sample_indices(self, epoch: int) -> list[int]:
        """Sample image indices for the given epoch according to AFSS policy.

        Args:
            epoch (int): Current training epoch.

        Returns:
            (list[int]): List of selected image indices.
        """
        rng = np.random.RandomState(epoch + self.seed)
        selected = []

        easy_set = set()
        moderate_set = set()
        hard_set = []

        for i, st in self.state.items():
            if i in self.always_train:  # few-object images bypass difficulty sampling and train every epoch
                continue
            s_i = min(st["precision"], st["recall"])
            if s_i > self.easy_thr:
                easy_set.add(i)
            elif s_i >= self.moderate_thr:
                moderate_set.add(i)
            else:
                hard_set.append(i)

        # Always-train (few-object) images: included unconditionally.
        selected.extend(self.always_train)
        n_always = len(self.always_train)

        # Per-bucket sampled counts, tracked for diagnostics (compare against paper Fig. 3).
        n_easy_sampled = 0
        n_moderate_sampled = 0

        # Hard set: include all hard images
        selected.extend(hard_set)

        # Easy set
        if easy_set:
            forced_easy = [i for i in easy_set if (epoch - 1 - self.state[i]["last_seen_epoch"]) >= 10]
            easy_budget = round(self.easy_ratio * len(easy_set))
            forced_easy_quota = min(len(forced_easy), math.floor(0.5 * easy_budget))
            random_easy_quota = easy_budget - forced_easy_quota

            if easy_budget > 0:
                forced_easy_sample = []
                if forced_easy_quota > 0 and forced_easy:
                    forced_easy_sample = rng.choice(forced_easy, size=forced_easy_quota, replace=False).tolist()
                selected.extend(forced_easy_sample)
                n_easy_sampled += len(forced_easy_sample)

                remaining_easy = [i for i in easy_set if i not in set(forced_easy_sample)]
                if random_easy_quota > 0 and remaining_easy:
                    random_easy_sample = rng.choice(remaining_easy, size=random_easy_quota, replace=False).tolist()
                    selected.extend(random_easy_sample)
                    n_easy_sampled += len(random_easy_sample)

        # Moderate set
        if moderate_set:
            forced_moderate = [i for i in moderate_set if (epoch - 1 - self.state[i]["last_seen_epoch"]) >= 3]
            moderate_budget = round(self.moderate_ratio * len(moderate_set))
            selected.extend(forced_moderate)
            n_moderate_sampled += len(forced_moderate)

            random_moderate_quota = moderate_budget - len(forced_moderate)
            remaining_moderate = [i for i in moderate_set if i not in set(forced_moderate)]
            if random_moderate_quota > 0 and remaining_moderate:
                random_moderate_sample = rng.choice(
                    remaining_moderate, size=random_moderate_quota, replace=False
                ).tolist()
                selected.extend(random_moderate_sample)
                n_moderate_sampled += len(random_moderate_sample)

        # Diagnostic: difficulty distribution and per-bucket usage for this epoch.
        # "max unused" is the largest gap (in epochs) since any easy image was last trained — a growing value
        # signals the continuous-review budget cannot keep up and easy images are being forgotten.
        max_unused_easy = max((epoch - 1 - self.state[i]["last_seen_epoch"] for i in easy_set), default=0)
        n_selected = n_easy_sampled + n_moderate_sampled + len(hard_set) + n_always
        LOGGER.info(
            f"AFSS epoch {epoch} difficulty: "
            f"easy={len(easy_set)} (sampled {n_easy_sampled}), "
            f"moderate={len(moderate_set)} (sampled {n_moderate_sampled}), "
            f"hard={len(hard_set)} (sampled {len(hard_set)}), "
            f"always-train={n_always} | "
            f"total selected={n_selected}/{self.num_images} | "
            f"easy max-unused={max_unused_easy} epochs"
        )

        selected_indices = sorted(selected)
        if not selected_indices:
            LOGGER.warning(f"AFSS sampled zero images for epoch {epoch}; falling back to full dataset.")
            selected_indices = list(range(self.num_images))
        return selected_indices

    def update_last_seen(self, indices: list[int], epoch: int) -> None:
        """Update last_seen_epoch for the given indices.

        Args:
            indices (list[int]): List of image indices that were seen this epoch.
            epoch (int): Current training epoch.
        """
        for i in indices:
            self.state[i]["last_seen_epoch"] = epoch

    def update_metrics(self, image_metrics: dict[str, dict], filename_to_idx: dict[str, int]) -> None:
        """Update per-image precision and recall from validator metrics.

        Args:
            image_metrics (dict[str, dict]): Dict keyed by image filename with precision/recall values.
            filename_to_idx (dict[str, int]): Mapping from filename to dataset index.
        """
        for filename, metrics in image_metrics.items():
            idx = filename_to_idx.get(Path(filename).name)
            if idx is None:
                continue
            self.state[idx]["precision"] = float(metrics.get("precision", 0.0))
            self.state[idx]["recall"] = float(metrics.get("recall", 0.0))


def _unwrap_dataset(dataset):
    """Unwrap a dataset from any dataloader wrapper layers."""
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def afss_on_epoch_start(trainer):
    """AFSS callback: sample active indices at the start of each epoch after warmup."""
    if not hasattr(trainer, "afss_scheduler"):
        # Lazy init on first epoch
        dataset = _unwrap_dataset(trainer.train_loader.dataset)

        obj_counts = [len(lb["cls"]) for lb in dataset.labels]  # per-image GT object count, in dataset index order
        trainer.afss_scheduler = AFSSScheduler(
            len(dataset),
            seed=trainer.args.seed,
            easy_thr=trainer.args.afss_easy_thr,
            easy_ratio=trainer.args.afss_easy_ratio,
            moderate_thr=trainer.args.afss_moderate_thr,
            moderate_ratio=trainer.args.afss_moderate_ratio,
            obj_counts=obj_counts,
            min_objects=trainer.args.afss_min_objects,
        )
        n_always = len(trainer.afss_scheduler.always_train)
        if n_always:
            LOGGER.info(
                f"AFSS: {n_always}/{len(dataset)} images have <= {trainer.args.afss_min_objects} GT objects; "
                f"these will be trained every epoch."
            )
        trainer.afss_current_indices = list(range(len(dataset)))

        # Resume: restore scheduler state if available
        afss_path = trainer.wdir / "afss_state.pt"
        if afss_path.exists():
            state = torch.load(afss_path, weights_only=False)
            if len(state) == trainer.afss_scheduler.num_images:
                trainer.afss_scheduler.state = state
            else:
                LOGGER.warning(
                    f"AFSS resume state mismatch: expected {trainer.afss_scheduler.num_images} images, "
                    f"got {len(state)}. Starting with fresh AFSS state."
                )

    epoch = trainer.epoch
    if epoch < trainer.args.warmup_epochs:  # do not use afss during warmup
        return

    full_final = trainer.epochs - epoch <= trainer.args.afss_full_final
    if full_final:
        # Full-coverage consolidation phase: train on every image for the final afss_full_final epochs to recover
        # drift on long-skipped images before the final model is selected.
        selected_indices = list(range(trainer.afss_scheduler.num_images))
    else:
        selected_indices = trainer.afss_scheduler.sample_indices(epoch)

    # DDP broadcast
    if trainer.world_size > 1:
        if RANK == 0:
            broadcast_list = [selected_indices]
        else:
            broadcast_list = [None]
        dist.broadcast_object_list(broadcast_list, src=0)
        selected_indices = broadcast_list[0]

    old_nb = trainer.nb
    if selected_indices == trainer.afss_current_indices:
        pass  # active set unchanged (e.g. consecutive full-coverage epochs); keep the current loader and workers
    elif trainer.world_size > 1:
        # Rebuild loader for DDP so DistributedSampler sees new length
        batch_size = trainer.batch_size // trainer.world_size
        old_loader = trainer.train_loader
        trainer.train_loader = trainer.get_dataloader(
            trainer.data["train"],
            batch_size=batch_size,
            rank=LOCAL_RANK,
            mode="train",
            active_indices=selected_indices,
        )
        del old_loader

        new_dataset = _unwrap_dataset(trainer.train_loader.dataset)

        if trainer.args.close_mosaic and epoch >= (trainer.epochs - trainer.args.close_mosaic):
            new_dataset.close_mosaic(hyp=copy(trainer.args))
    else:
        dataset = _unwrap_dataset(trainer.train_loader.dataset)
        dataset.active_indices = selected_indices
        trainer.train_loader.reset()

    trainer.afss_current_indices = selected_indices
    trainer.nb = len(trainer.train_loader)
    # Adjust last_opt_step so optimizer stepping continues correctly when nb changes
    if old_nb != trainer.nb:
        trainer.last_opt_step -= epoch * (old_nb - trainer.nb)
    LOGGER.info(
        f"AFSS epoch {epoch}: training on {len(selected_indices)}/{trainer.afss_scheduler.num_images} images"
        + (" (full-coverage final phase)" if full_final else "")
    )


def afss_on_epoch_end(trainer):
    """AFSS callback: update last seen and refresh metrics at the end of each epoch."""
    if not hasattr(trainer, "afss_scheduler"):
        return
    epoch = trainer.epoch
    trainer.afss_scheduler.update_last_seen(trainer.afss_current_indices, epoch)
    if trainer.epochs - epoch - 1 <= trainer.args.afss_full_final:
        return  # every remaining epoch trains on the full dataset, so refreshing per-image metrics is wasted compute
    if epoch >= trainer.args.warmup_epochs and (epoch - math.ceil(trainer.args.warmup_epochs)) % 5 == 0:
        afss_refresh_metrics(trainer)


def afss_refresh_metrics(trainer):
    """Run validation on the training set to refresh per-image precision/recall for AFSS."""
    if not hasattr(trainer, "afss_validator"):
        batch_size = trainer.batch_size // max(trainer.world_size, 1)
        train_eval_loader = trainer.get_dataloader(
            trainer.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="val"
        )
        trainer.afss_validator = trainer.get_validator().__class__(
            train_eval_loader,
            save_dir=trainer.save_dir / "afss_train_eval",
            args=copy(trainer.args),
            _callbacks=trainer.callbacks,
        )

    trainer.afss_validator(trainer)

    if RANK in {-1, 0}:
        image_metrics = trainer.afss_validator.metrics.box.image_metrics
        dataset = _unwrap_dataset(trainer.train_loader.dataset)
        filename_to_idx = {Path(f).name: i for i, f in enumerate(dataset.im_files)}
        trainer.afss_scheduler.update_metrics(image_metrics, filename_to_idx)
        LOGGER.info(f"AFSS: refreshed metrics for {len(image_metrics)} images")

    if trainer.world_size > 1:
        state_list = [trainer.afss_scheduler.state if RANK == 0 else None]
        dist.broadcast_object_list(state_list, src=0)
        trainer.afss_scheduler.state = state_list[0]


def afss_save_state(trainer):
    """Save AFSS scheduler state to a sidecar checkpoint file."""
    if hasattr(trainer, "afss_scheduler") and RANK in {-1, 0}:
        torch.save(trainer.afss_scheduler.state, trainer.wdir / "afss_state.pt")
