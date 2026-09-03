# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import os
import random
from collections.abc import Iterator
from copy import copy
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, dataloader, distributed

from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.data.dataset import (
    DepthDataset,
    GroundingDataset,
    PolygonSemanticDataset,
    SemanticDataset,
    YOLODataset,
    YOLOMultiModalDataset,
)
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
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS, get_split_fraction
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.checks import check_file
from ultralytics.utils.torch_utils import TORCH_1_13, TORCH_2_7, get_torch_device_backend


class InfiniteDataLoader(dataloader.DataLoader):
    """DataLoader that repeats its sampler forever and reuses one worker iterator across epochs.

    Prefetching runs straight through epoch boundaries, which keeps small-dataset training fast. `close_dataloader`
    shuts the workers down; the next iteration then spawns fresh ones that see any dataset changes.

    Attributes:
        batch_sampler (_RepeatSampler): A sampler that repeats indefinitely.

    Examples:
        Create an infinite DataLoader for training
        >>> dataset = YOLODataset(...)
        >>> dataloader = InfiniteDataLoader(dataset, batch_size=16, shuffle=True)
        >>> for batch in dataloader:  # one epoch per loop, workers kept between loops
        >>>     train_step(batch)
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the InfiniteDataLoader with the same arguments as DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))

    def __len__(self) -> int:
        """Return the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self) -> Iterator:
        """Yield one epoch of batches from the persistent iterator."""
        if self._iterator is None:
            self._iterator = self._get_iterator()
        for _ in range(len(self)):
            yield next(self._iterator)

    def __del__(self):
        """Ensure that workers are properly terminated when the DataLoader is deleted."""
        try:
            for w in getattr(self._iterator, "_workers", ()):  # force terminate
                if w.is_alive():
                    w.terminate()
            close_dataloader(self)
        except Exception:
            pass


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
            sampler = getattr(self.sampler, "sampler", None)  # BatchSampler wraps the index sampler
            if hasattr(sampler, "set_epoch"):
                # The next pass starts prefetching before the trainer's set_epoch, so reshuffle it here
                sampler.set_epoch(sampler.epoch + 1)


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
        # Use unit batches when one input batch would span the dataset.
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
        start_idx = min(start_batch * self.batch_size, self.total_size)
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
    fraction: float | None = None,
) -> Dataset:
    """Build and return a YOLO dataset based on configuration parameters."""
    pad = 0.0 if mode == "train" else 0.5
    rect = cfg.rect or rect
    if cfg.task == "depth":
        dataset = DepthDataset
        pad, rect = 0.0, rect and mode == "train"  # depth val letterbox stretches, so pad and rect_shape are ignored
    elif cfg.task == "semantic":
        dataset = SemanticDataset if data.get("masks_dir") else PolygonSemanticDataset
        pad = 0.0  # no pad for semantic
    elif multi_modal:
        dataset = YOLOMultiModalDataset
    else:
        dataset = YOLODataset

    if data.get("complete"):
        fraction = 1.0  # already limited during dataset download
    elif fraction is None:
        fraction = get_split_fraction(cfg.fraction, mode)
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=copy(cfg),
        rect=rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=pad,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=fraction,
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
        hyp=copy(cfg),
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        fraction=get_split_fraction(cfg.fraction, mode),
    )


def build_dataloader(
    dataset,
    batch: int,
    workers: int,
    shuffle: bool = True,
    rank: int = -1,
    drop_last: bool = False,
    pin_memory: bool = True,
    device: torch.device | str = "cuda",
    infinite: bool = True,
) -> dataloader.DataLoader:
    """Create and return a DataLoader for training or validation.

    Args:
        dataset (Dataset): Dataset to load data from.
        batch (int): Batch size for the dataloader.
        workers (int): Number of worker processes for data loading.
        shuffle (bool, optional): Whether to shuffle the dataset.
        rank (int, optional): Process rank in distributed training. -1 for single-GPU training.
        drop_last (bool, optional): Whether to drop the last incomplete batch.
        pin_memory (bool, optional): Whether to use pinned memory for dataloader.
        device (torch.device | str, optional): Device used by the dataloader consumer.
        infinite (bool, optional): Prefetch across epoch boundaries with `InfiniteDataLoader`; a finite loader instead
            drains between passes so validation holds no queued batches while training resumes.

    Returns:
        (torch.utils.data.DataLoader): A dataloader with persistent workers for training or validation.

    Examples:
        Create a dataloader for training
        >>> dataset = YOLODataset(...)
        >>> dataloader = build_dataloader(dataset, batch=16, workers=4, shuffle=True)
    """
    dataset_len = len(dataset)
    batch = min(batch, dataset_len)
    seed = torch.initial_seed() - RANK - 1
    sampler = (
        None
        if rank == -1
        else distributed.DistributedSampler(dataset, shuffle=shuffle, seed=seed)
        if shuffle
        else ContiguousDistributedSampler(dataset)
    )
    samples = len(sampler) if sampler is not None else dataset_len
    drop_last = drop_last and bool(batch) and dataset_len % batch != 0
    batches = (samples // batch if drop_last else math.ceil(samples / batch)) if batch else 0
    device_type = getattr(device, "type", str(device).split(":")[0])
    nd = get_torch_device_backend(device).device_count() if device_type not in {"cpu", "mps"} else 0
    # Do not create more worker processes than final loader batches. Single-batch loaders run in-process to avoid
    # persistent DataLoader worker pools that add overhead and can stall tiny datasets while holding CUDA context.
    nw = min(os.cpu_count() // max(nd, 1), workers, 0 if batches <= 1 else batches)  # number of workers
    generator = torch.Generator()
    generator.manual_seed((6148914691236517205 + RANK + seed) % (1 << 64))
    pin_memory = nd > 0 and pin_memory
    pin_memory_device = (
        device_type if pin_memory and device_type in {"npu", "xpu"} and TORCH_1_13 and not TORCH_2_7 else None
    )
    return (InfiniteDataLoader if infinite else dataloader.DataLoader)(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=pin_memory,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last,
        persistent_workers=nw > 0,  # reuse workers across epochs
        **({"prefetch_factor": 4} if nw > 0 else {}),  # increase over default 2; only accepted with workers
        **({"pin_memory_device": pin_memory_device} if pin_memory_device else {}),
    )


def close_dataloader(loader) -> None:
    """Shut down a DataLoader's persistent workers so they neither outlive their use nor serve stale dataset state."""
    iterator = getattr(loader, "_iterator", None)  # set once an iterator persists across passes
    if iterator is not None:
        if loader.num_workers:
            iterator._shutdown_workers()  # joins workers and calls torch._C._remove_worker_pids
        loader._iterator = None  # the next iteration spawns fresh workers


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
    dataset.source_type = source_type

    return dataset
