# Ultralytics YOLO 🚀, GPL-3.0 license

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, dataloader, distributed

from ..utils import LOGGER, colorstr
from ..utils.torch_utils import torch_distributed_zero_first
from .dataset import ClassificationDataset, YOLODataset
from .utils import PIN_MEMORY, RANK


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


def build_dataloader(cfg, batch_size, img_path, stride=32, label_path=None, rank=-1, mode="train"):
    assert mode in ["train", "val"]
    shuffle = mode == "train"
    if cfg.rect and shuffle:
        LOGGER.warning("WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = YOLODataset(
            img_path=img_path,
            label_path=label_path,
            imgsz=cfg.imgsz,
            batch_size=batch_size,
            augment=mode == "train",  # augmentation
            hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
            rect=cfg.rect if mode == "train" else True,  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            use_segments=cfg.task == "segment",
            use_keypoints=cfg.task == "keypoint")

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    workers = cfg.workers if mode == "train" else cfg.workers * 2
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if cfg.image_weights or cfg.close_mosaic else InfiniteDataLoader  # allow attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return loader(dataset=dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn=getattr(dataset, "collate_fn", None),
                  worker_init_fn=seed_worker,
                  generator=generator), dataset


# build classification
# TODO: using cfg like `build_dataloader`
def build_classification_dataloader(path,
                                    imgsz=224,
                                    batch_size=16,
                                    augment=True,
                                    cache=False,
                                    rank=-1,
                                    workers=8,
                                    shuffle=True):
    # Returns Dataloader object to be used with YOLOv5 Classifier
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = ClassificationDataset(root=path, imgsz=imgsz, augment=augment, cache=cache)
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle and sampler is None,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              worker_init_fn=seed_worker,
                              generator=generator)  # or DataLoader(persistent_workers=True)
