# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import os
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, dataloader, distributed

from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.data.dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset
from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.checks import check_file
from ultralytics.utils.torch_utils import TORCH_2_0


class InfiniteDataLoader(dataloader.DataLoader):
    """DataLoader that reuses workers for infinite iteration.

    This dataloader extends the PyTorch DataLoader to provide infinite recycling of workers, which improves efficiency
    for training loops that need to iterate through the dataset multiple times without recreating workers.

    Attributes:
        batch_sampler (_RepeatSampler): A sampler that repeats indefinitely.
        iterator (Iterator): The iterator from the parent DataLoader.

    Methods:
        __len__: Return the length of the batch sampler's sampler.
        __iter__: Yield batches from the underlying iterator.
        __del__: Ensure workers are properly terminated.
        reset: Reset the iterator, useful when modifying dataset settings during training.

    Examples:
        Create an infinite DataLoader for training
        >>> dataset = YOLODataset(...)
        >>> dataloader = InfiniteDataLoader(dataset, batch_size=16, shuffle=True)
        >>> for batch in dataloader:  # Infinite iteration
        >>>     train_step(batch)
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the InfiniteDataLoader with the same arguments as DataLoader."""
        if not TORCH_2_0:
            kwargs.pop("prefetch_factor", None)  # not supported by earlier versions
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        """Return the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self) -> Iterator:
        """Create an iterator that yields indefinitely from the underlying iterator."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def __del__(self):
        """Ensure that workers are properly terminated when the DataLoader is deleted."""
        try:
            if not hasattr(self.iterator, "_workers"):
                return
            for w in self.iterator._workers:  # force terminate
                if w.is_alive():
                    w.terminate()
            self.iterator._shutdown_workers()  # cleanup
        except Exception:
            pass

    def reset(self):
        """Reset the iterator to allow modifications to the dataset during training."""
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """Sampler that repeats forever for infinite iteration.

    This sampler wraps another sampler and yields its contents indefinitely, allowing for infinite iteration over a
    dataset without recreating the sampler.

    Attributes:
        sampler (torch.utils.data.Sampler): The sampler to repeat.
    """

    def __init__(self, sampler: Any):
        """Initialize the _RepeatSampler with a sampler to repeat indefinitely."""
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        """Iterate over the sampler indefinitely, yielding its contents."""
        while True:
            yield from iter(self.sampler)


class ContiguousDistributedSampler(torch.utils.data.Sampler):
    """Distributed sampler that assigns contiguous batch-aligned chunks of the dataset to each GPU.

    Unlike PyTorch's DistributedSampler which distributes samples in a round-robin fashion (GPU 0 gets indices
    [0,2,4,...], GPU 1 gets [1,3,5,...]), this sampler gives each GPU contiguous batches of the dataset (GPU 0 gets
    batches [0,1,2,...], GPU 1 gets batches [k,k+1,...], etc.). This preserves any ordering or grouping in the original
    dataset, which is critical when samples are organized by similarity (e.g., images sorted by size to enable efficient
    batching without padding when using rect=True).

    The sampler handles uneven batch counts by distributing remainder batches to the first few ranks, ensuring all
    samples are covered exactly once across all GPUs.

    Args:
        dataset (Dataset): Dataset to sample from. Must implement __len__.
        num_replicas (int, optional): Number of distributed processes. Defaults to world size.
        batch_size (int, optional): Batch size used by dataloader. Defaults to dataset.batch_size or 1.
        rank (int, optional): Rank of current process. Defaults to current rank.
        shuffle (bool, optional): Whether to shuffle indices within each rank's chunk. Defaults to False. When True,
            shuffling is deterministic and controlled by set_epoch() for reproducibility.

    Examples:
        >>> # For validation with size-grouped images
        >>> sampler = ContiguousDistributedSampler(val_dataset, batch_size=32, shuffle=False)
        >>> loader = DataLoader(val_dataset, batch_size=32, sampler=sampler)
        >>> # For training with shuffling
        >>> sampler = ContiguousDistributedSampler(train_dataset, batch_size=32, shuffle=True)
        >>> for epoch in range(num_epochs):
        ...     sampler.set_epoch(epoch)
        ...     for batch in loader:
        ...         ...
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        batch_size: int | None = None,
        rank: int | None = None,
        shuffle: bool = False,
    ) -> None:
        """Initialize the sampler with dataset and distributed training parameters."""
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if batch_size is None:
            batch_size = getattr(dataset, "batch_size", 1)

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.total_size = len(dataset)
        # ensure all ranks have a sample if batch size >= total size; degenerates to round-robin sampler
        self.batch_size = 1 if batch_size >= self.total_size else batch_size
        self.num_batches = math.ceil(self.total_size / self.batch_size)

    def _get_rank_indices(self) -> tuple[int, int]:
        """Calculate the start and end sample indices for this rank."""
        # Calculate which batches this rank handles
        batches_per_rank_base = self.num_batches // self.num_replicas
        remainder = self.num_batches % self.num_replicas

        # This rank gets an extra batch if rank < remainder
        batches_for_this_rank = batches_per_rank_base + (1 if self.rank < remainder else 0)

        # Calculate starting batch: base position + number of extra batches given to earlier ranks
        start_batch = self.rank * batches_per_rank_base + min(self.rank, remainder)
        end_batch = start_batch + batches_for_this_rank

        # Convert batch indices to sample indices
        start_idx = start_batch * self.batch_size
        end_idx = min(end_batch * self.batch_size, self.total_size)

        return start_idx, end_idx

    def __iter__(self) -> Iterator:
        """Generate indices for this rank's contiguous chunk of the dataset."""
        start_idx, end_idx = self._get_rank_indices()
        indices = list(range(start_idx, end_idx))

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = [indices[i] for i in torch.randperm(len(indices), generator=g).tolist()]

        return iter(indices)

    def __len__(self) -> int:
        """Return the number of samples in this rank's chunk."""
        start_idx, end_idx = self._get_rank_indices()
        return end_idx - start_idx

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler to ensure different shuffling patterns across epochs.

        Args:
            epoch (int): Epoch number to use as the random seed for shuffling.
        """
        self.epoch = epoch


class BalancedDistributedSampler(torch.utils.data.Sampler):
    """
    Distributed sampler with strict per-class balanced sampling.

    Pre-builds a class-to-image mapping (cls → [img_indices]).  At every epoch a fixed
    number of images is drawn *per class* (with replacement when a class has fewer images
    than the quota), so every class contributes equally to the epoch regardless of how many
    bboxes an image contains.  The resulting global index list is sharded across ranks,
    giving each GPU a non-overlapping, contiguous slice.

    total_size  = samples_per_class × num_classes
    samples_per_class = max(1, len(dataset) // num_classes)

    Args:
        dataset: Dataset with a ``labels`` attribute — a list of dicts, each having a ``cls``
            key whose value is a numpy array of shape ``(N, 1)`` containing class IDs.
        num_replicas (int, optional): Number of distributed processes. Defaults to world size.
        rank (int, optional): Rank of the current process. Defaults to current rank.
        shuffle (bool): If True, additionally shuffle each rank's local slice every epoch.
        save_histogram (bool): If True, rank-0 saves a class-distribution PNG after each epoch.

    Usage::

        sampler = BalancedDistributedSampler(train_dataset)
        loader = DataLoader(train_dataset, sampler=sampler, batch_size=32)
        for epoch in range(epochs):
            sampler.set_epoch(epoch)   # must call every epoch
            for batch in loader:
                ...
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, save_histogram=False):
        """Initialize the sampler, building class-to-image mapping."""
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.save_histogram = save_histogram
        self.epoch = 0

        # Log dataset info
        dataset_name = getattr(dataset, '__class__', 'Unknown').__name__
        dataset_size = len(dataset)
        print(f"\n🔍 BalancedDistributedSampler.__init__:")
        print(f"  Dataset class: {dataset_name}")
        print(f"  Dataset size: {dataset_size} images")

        # cls_to_imgs: {class_id: [img_idx, ...]}
        self.cls_to_imgs = self._build_cls_to_imgs()
        self.num_classes = len(self.cls_to_imgs)
        self.samples_per_class = max(1, len(dataset) // max(self.num_classes, 1))
        self.total_size = self.samples_per_class * self.num_classes

    def _build_cls_to_imgs(self) -> dict:
        """Build mapping {class_id: [image_indices]} from dataset labels.
        
        Also validates that each class has at least one bbox annotation.
        """
        labels = getattr(self.dataset, "labels", None)
        cls_to_imgs: dict = {}
        cls_bbox_count: dict = {}  # Count bboxes per class
        
        if labels is None:
            return cls_to_imgs
            
        for i, lb in enumerate(labels):
            cls = lb.get("cls", [])
            if len(cls):
                for c in np.unique(cls.flatten().astype(int)):
                    c_int = int(c)
                    cls_to_imgs.setdefault(c_int, []).append(i)
                    cls_bbox_count[c_int] = cls_bbox_count.get(c_int, 0) + len(cls[cls == c])
        
        # Check for classes with 0 bboxes
        if cls_bbox_count:
            classes_with_zero = [c for c in cls_bbox_count if cls_bbox_count[c] == 0]
            if classes_with_zero:
                print(f"⚠️  Classes with 0 bboxes (should not happen): {classes_with_zero[:10]}...")
            
            # Print summary
            total_classes_declared = max(cls_bbox_count.keys()) + 1 if cls_bbox_count else 0
            total_classes_found = len(cls_to_imgs)
            total_bboxes = sum(cls_bbox_count.values())
            
            print(f"\n📊 BalancedDistributedSampler class distribution:")
            print(f"  Total classes in data: {total_classes_found}")
            print(f"  Min class ID: {min(cls_to_imgs.keys()) if cls_to_imgs else 'N/A'}")
            print(f"  Max class ID: {max(cls_to_imgs.keys()) if cls_to_imgs else 'N/A'}")
            print(f"  Total bboxes: {total_bboxes}")
            print(f"  Avg bboxes/class: {total_bboxes / max(total_classes_found, 1):.1f}")
            
            # Show class bbox distribution (top 10 and bottom 10)
            sorted_counts = sorted(cls_bbox_count.items(), key=lambda x: x[1], reverse=True)
            print(f"  Top 5 classes by bbox count:")
            for cls_id, count in sorted_counts[:5]:
                print(f"    cls {cls_id}: {count} bboxes")
            print(f"  Bottom 5 classes by bbox count:")
            for cls_id, count in sorted_counts[-5:]:
                print(f"    cls {cls_id}: {count} bboxes")
        return cls_to_imgs

    def __iter__(self):
        """Yield indices for this rank's slice of the class-balanced sample."""
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # 1. For every class draw exactly `samples_per_class` image indices (with replacement)
        indices = []
        for cls_id in sorted(self.cls_to_imgs.keys()):
            img_list = self.cls_to_imgs[cls_id]
            n = len(img_list)
            sampled = torch.randint(0, n, (self.samples_per_class,), generator=g).tolist()
            indices.extend(img_list[p] for p in sampled)

        # 2. Optionally save class-distribution histogram (all ranks)
        if self.save_histogram and self.rank == 0:
            self._save_cls_histogram(indices)

        # 3. Shard across ranks (contiguous slice)
        per_rank = self.total_size // self.num_replicas
        remainder = self.total_size % self.num_replicas
        start = self.rank * per_rank + min(self.rank, remainder)
        end = start + per_rank + (1 if self.rank < remainder else 0)
        rank_indices = indices[start:end]

        # 4. Optionally shuffle within this rank's slice
        if self.shuffle:
            perm = torch.randperm(len(rank_indices), generator=g).tolist()
            rank_indices = [rank_indices[i] for i in perm]

        return iter(rank_indices)

    def _save_cls_histogram(self, indices: list) -> None:
        """Save a PNG histogram and CSV of the class distribution across sampled indices."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        labels = getattr(self.dataset, "labels", None)
        if labels is None:
            return

        # Count how many times each class appears across all sampled images
        class_counts: dict = {}
        for idx in indices:
            if idx >= len(labels):
                print(f"⚠️  Index {idx} out of bounds! labels size: {len(labels)}")
                continue
            cls = labels[idx].get("cls", [])
            if len(cls):
                for c in cls.flatten().astype(int):
                    class_counts[int(c)] = class_counts.get(int(c), 0) + 1

        if not class_counts:
            return

        sorted_cls = sorted(class_counts.keys())
        counts = [class_counts[c] for c in sorted_cls]
        n_cls = len(sorted_cls)

        # Debug: print histogram info
        print(f"\n📈 _save_cls_histogram (rank {self.rank}, epoch {self.epoch}):")
        print(f"  Num indices sampled: {len(indices)}")
        print(f"  Num classes in histogram: {n_cls}")
        print(f"  Class ID range: {min(sorted_cls)}-{max(sorted_cls)}")
        print(f"  Total bboxes in sampled: {sum(counts)}")

        # Scale figure width with number of classes; cap bar width to keep plot readable
        fig_w = max(24, n_cls // 20)
        fig, ax = plt.subplots(figsize=(fig_w, 6))
        bar_w = max(0.4, min(1.0, 800 / n_cls))
        ax.bar(range(n_cls), counts, width=bar_w, color="steelblue", edgecolor="none")
        ax.set_xlabel("Class ID", fontsize=11)
        ax.set_ylabel("Bbox Count in Sampled Images", fontsize=11)
        ax.set_title(
            f"Class Distribution — epoch {self.epoch}  "
            f"({n_cls} classes, {self.samples_per_class} imgs/cls)",
            fontsize=12,
        )

        # Show ~50 tick labels regardless of class count
        step = max(1, n_cls // 50)
        tick_pos = list(range(0, n_cls, step))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([str(sorted_cls[i]) for i in tick_pos], rotation=60, ha="right", fontsize=7)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        plt.tight_layout()
        save_dir = Path("runs") / "balanced_sampler"
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"epoch{self.epoch:04d}_rank{self.rank}"
        save_path = save_dir / f"cls_dist_{suffix}.png"
        plt.savefig(save_path, dpi=120)
        plt.close(fig)

        
        # Save CSV: cls_id, bbox_count — one row per class
        print("Saved class distribution histogram to", save_path)
        csv_path = save_dir / f"cls_dist_{suffix}.csv"
        with open(csv_path, "w") as f:
            f.write("cls_id,bbox_count\n")
            for c, cnt in zip(sorted_cls, counts):
                f.write(f"{c},{cnt}\n")
        print("Saved class distribution CSV to", csv_path)

    def __len__(self) -> int:
        """Return number of samples for this rank."""
        per_rank = self.total_size // self.num_replicas
        remainder = self.total_size % self.num_replicas
        return per_rank + (1 if self.rank < remainder else 0)

    def set_epoch(self, epoch: int) -> None:
        """Set epoch so each epoch uses a different sampling seed."""
        self.epoch = epoch


def seed_worker(worker_id: int) -> None:
    """Set dataloader worker seed for reproducibility across worker processes."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_yolo_dataset(
    cfg: IterableSimpleNamespace,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    multi_modal: bool = False,
) -> Dataset:
    """Build and return a YOLO dataset based on configuration parameters."""
    dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_grounding(
    cfg: IterableSimpleNamespace,
    img_path: str,
    json_file: str,
    batch: int,
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    max_samples: int = 80,
) -> Dataset:
    """Build and return a GroundingDataset based on configuration parameters."""
    return GroundingDataset(
        img_path=img_path,
        json_file=json_file,
        max_samples=max_samples,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_dataloader(
    dataset,
    batch: int,
    workers: int,
    shuffle: bool = True,
    rank: int = -1,
    drop_last: bool = False,
    pin_memory: bool = True,
    balanced: bool = True,
) -> InfiniteDataLoader:
    """Create and return an InfiniteDataLoader for training or validation.

    Args:
        dataset (Dataset): Dataset to load data from.
        batch (int): Batch size for the dataloader.
        workers (int): Number of worker processes for data loading.
        shuffle (bool, optional): Whether to shuffle the dataset.
        rank (int, optional): Process rank in distributed training. -1 for single-GPU training.
        drop_last (bool, optional): Whether to drop the last incomplete batch.
        pin_memory (bool, optional): Whether to use pinned memory for dataloader.
        balanced (bool, optional): Whether to use class-balanced sampling.

    Returns:
        (InfiniteDataLoader): A dataloader that can be used for training or validation.

    Examples:
        Create a dataloader for training
        >>> dataset = YOLODataset(...)
        >>> dataloader = build_dataloader(dataset, batch=16, workers=4, shuffle=True)
    """
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    if balanced and (not shuffle):
        print("⚠️  Warning: balanced sampling is only effective when shuffle=True. Setting balanced=False.")

    if rank == -1:
        # Single-GPU: use BalancedDistributedSampler when balanced sampling is requested
        sampler = BalancedDistributedSampler(dataset, num_replicas=1, rank=0, shuffle=shuffle) if (balanced and shuffle) else None
    else:
        # Multi-GPU distributed training
        sampler = BalancedDistributedSampler(dataset, shuffle=shuffle) if (balanced and shuffle) else distributed.DistributedSampler(dataset, shuffle=shuffle) if shuffle else ContiguousDistributedSampler(dataset)
      
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        prefetch_factor=4 if nw > 0 else None,  # increase over default 2
        pin_memory=nd > 0 and pin_memory,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last and len(dataset) % batch != 0,
    )


def check_source(
    source: str | int | Path | list | tuple | np.ndarray | Image.Image | torch.Tensor,
) -> tuple[Any, bool, bool, bool, bool, bool]:
    """Check the type of input source and return corresponding flag values.

    Args:
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | torch.Tensor): The input source to check.

    Returns:
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | torch.Tensor): The processed source.
        webcam (bool): Whether the source is a webcam.
        screenshot (bool): Whether the source is a screenshot.
        from_img (bool): Whether the source is an image or list of images.
        in_memory (bool): Whether the source is an in-memory object.
        tensor (bool): Whether the source is a torch.Tensor.

    Examples:
        Check a file path source
        >>> source, webcam, screenshot, from_img, in_memory, tensor = check_source("image.jpg")

        Check a webcam source
        >>> source, webcam, screenshot, from_img, in_memory, tensor = check_source(0)
    """
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)
        source_lower = source.lower()
        is_url = source_lower.startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        is_file = (urlsplit(source_lower).path if is_url else source_lower).rpartition(".")[-1] in (
            IMG_FORMATS | VID_FORMATS
        )
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source_lower == "screen"
        if is_url and is_file:
            source = check_file(source)  # download
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # convert all list elements to PIL or np arrays
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError("Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict")

    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(
    source: str | int | Path | list | tuple | np.ndarray | Image.Image | torch.Tensor,
    batch: int = 1,
    vid_stride: int = 1,
    buffer: bool = False,
    channels: int = 3,
):
    """Load an inference source for object detection and apply necessary transformations.

    Args:
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | torch.Tensor): The input source for
            inference.
        batch (int, optional): Batch size for dataloaders.
        vid_stride (int, optional): The frame interval for video sources.
        buffer (bool, optional): Whether stream frames will be buffered.
        channels (int, optional): The number of input channels for the model.

    Returns:
        (Dataset): A dataset object for the specified input source with attached source_type attribute.

    Examples:
        Load an image source for inference
        >>> dataset = load_inference_source("image.jpg", batch=1)

        Load a video stream source
        >>> dataset = load_inference_source("rtsp://example.com/stream", vid_stride=2)
    """
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)

    # DataLoader
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer, channels=channels)
    elif screenshot:
        dataset = LoadScreenshots(source, channels=channels)
    elif from_img:
        dataset = LoadPilAndNumpy(source, channels=channels)
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride, channels=channels)

    # Attach source types to the dataset
    setattr(dataset, "source_type", source_type)

    return dataset
