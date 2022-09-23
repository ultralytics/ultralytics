from torch.utils.data import DataLoader, dataloader, distributed
from .dataset import YOLODataset
from .dataset_wrappers import MixAndRectDataset
from .utils import PIN_MEMORY
from ..utils.general import LOGGER
from ..utils.torch_utils import torch_distributed_zero_first
import torch
import os
import numpy as np
import random


class InfiniteDataLoader(dataloader.DataLoader):
    """Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# TODO: we can inject most args from a config file
def build_dataloader(
    img_path,
    img_size,
    batch_size,
    stride=32,
    label_path=None,
    single_cls=False,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    image_weights=False,
    prefix="",
    shuffle=False,
    use_segments=False,
    use_keypoints=False,
):
    if rect and shuffle:
        LOGGER.warning("WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = YOLODataset(
            img_path=img_path,
            img_size=img_size,
            batch_size=batch_size,
            label_path=label_path,
            augment=augment,  # augmentation
            hyp=hyp,
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            prefix=prefix,
            use_segments=use_segments,
            use_keypoints=use_keypoints,
        )

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    # generator = torch.Generator()
    # generator.manual_seed(0)
    return (
        loader(
            # TODO: we can remove this once we don't need a data wrapper
            dataset = MixAndRectDataset(dataset),
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            num_workers=nw,
            sampler=sampler,
            pin_memory=PIN_MEMORY,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            # generator=generator,
        ),
        dataset,
    )
