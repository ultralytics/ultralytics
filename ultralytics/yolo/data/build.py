# Ultralytics YOLO 🚀, GPL-3.0 license

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, dataloader, distributed

from ultralytics.yolo.data.dataloaders.stream_loaders import (LOADERS, LoadImages, LoadPilAndNumpy, LoadScreenshots,
                                                              LoadStreams, LoadTensor, SourceTypes, autocast_list)
from ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.yolo.utils.checks import check_file

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
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
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


def seed_worker(worker_id):  # noqa
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(cfg, batch, img_path, stride=32, rect=False, names=None, rank=-1, mode='train'):
    assert mode in ['train', 'val']
    shuffle = mode == 'train'
    if cfg.rect and shuffle:
        LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = YOLODataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == 'train',  # augmentation
            hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
            rect=cfg.rect or rect,  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(stride),
            pad=0.0 if mode == 'train' else 0.5,
            prefix=colorstr(f'{mode}: '),
            use_segments=cfg.task == 'segment',
            use_keypoints=cfg.task == 'keypoint',
            names=names,
            classes=cfg.classes)

    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    workers = cfg.workers if mode == 'train' else cfg.workers * 2
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if cfg.image_weights or cfg.close_mosaic else InfiniteDataLoader  # allow attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return loader(dataset=dataset,
                  batch_size=batch,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn=getattr(dataset, 'collate_fn', None),
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


def check_source(source):
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('https://', 'http://', 'rtsp://', 'rtmp://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download
    elif isinstance(source, tuple(LOADERS)):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # convert all list elements to PIL or np arrays
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError('Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict')

    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(source=None, transforms=None, imgsz=640, vid_stride=1, stride=32, auto=True):
    """
    TODO: docs
    """
    source, webcam, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(webcam, screenshot, from_img, tensor)

    # Dataloader
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif webcam:
        dataset = LoadStreams(source,
                              imgsz=imgsz,
                              stride=stride,
                              auto=auto,
                              transforms=transforms,
                              vid_stride=vid_stride)

    elif screenshot:
        dataset = LoadScreenshots(source, imgsz=imgsz, stride=stride, auto=auto, transforms=transforms)
    elif from_img:
        dataset = LoadPilAndNumpy(source, imgsz=imgsz, stride=stride, auto=auto, transforms=transforms)
    else:
        dataset = LoadImages(source,
                             imgsz=imgsz,
                             stride=stride,
                             auto=auto,
                             transforms=transforms,
                             vid_stride=vid_stride)

    setattr(dataset, 'source_type', source_type)  # attach source types

    return dataset
