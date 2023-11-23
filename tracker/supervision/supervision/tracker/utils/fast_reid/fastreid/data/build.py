# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import os

import torch
from torch._six import string_classes
from collections import Mapping

from supervision.tracker.utils.fast_reid.fastreid.config import configurable
from supervision.tracker.utils.fast_reid.fastreid.utils import comm
from . import samplers
from .common import CommDataset
from .data_utils import DataLoaderX
from .datasets import DATASET_REGISTRY
from .transforms import build_transforms


__all__ = [
    "build_reid_train_loader",
    "build_reid_test_loader"
]

# _root = os.getenv("FASTREID_DATASETS", "datasets")
_root = os.getenv("FASTREID_DATASETS", "fast_reid/datasets")


def _train_loader_from_config(cfg, *, train_set=None, transforms=None, sampler=None, **kwargs):

    if transforms is None:
        transforms = build_transforms(cfg, is_train=True)

    if train_set is None:
        train_items = list()
        for d in cfg.DATASETS.NAMES:
            data = DATASET_REGISTRY.get(d)(root=_root, **kwargs)
            if comm.is_main_process():
                data.show_train()
            train_items.extend(data.train)

        train_set = CommDataset(train_items, transforms, relabel=True)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        num_instance = cfg.DATALOADER.NUM_INSTANCE
        mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()

        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = samplers.TrainingSampler(len(train_set))
        elif sampler_name == "NaiveIdentitySampler":
            sampler = samplers.NaiveIdentitySampler(train_set.img_items, mini_batch_size, num_instance)
        elif sampler_name == "BalancedIdentitySampler":
            sampler = samplers.BalancedIdentitySampler(train_set.img_items, mini_batch_size, num_instance)
        elif sampler_name == "SetReWeightSampler":
            set_weight = cfg.DATALOADER.SET_WEIGHT
            sampler = samplers.SetReWeightSampler(train_set.img_items, mini_batch_size, num_instance, set_weight)
        elif sampler_name == "ImbalancedDatasetSampler":
            sampler = samplers.ImbalancedDatasetSampler(train_set.img_items)
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "train_set": train_set,
        "sampler": sampler,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


@configurable(from_config=_train_loader_from_config)
def build_reid_train_loader(
        train_set, *, sampler=None, total_batch_size, num_workers=0,
):
    """
    Build a dataloader for object re-identification with some default features.
    This interface is experimental.

    Returns:
        torch.utils.data.DataLoader: a dataloader.
    """

    mini_batch_size = total_batch_size // comm.get_world_size()

    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, mini_batch_size, True)

    train_loader = DataLoaderX(
        comm.get_local_rank(),
        dataset=train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )

    return train_loader


def _test_loader_from_config(cfg, *, dataset_name=None, test_set=None, num_query=0, transforms=None, **kwargs):
    if transforms is None:
        transforms = build_transforms(cfg, is_train=False)

    if test_set is None:
        assert dataset_name is not None, "dataset_name must be explicitly passed in when test_set is not provided"
        data = DATASET_REGISTRY.get(dataset_name)(root=_root, **kwargs)
        if comm.is_main_process():
            data.show_test()
        test_items = data.query + data.gallery
        test_set = CommDataset(test_items, transforms, relabel=False)

        # Update query number
        num_query = len(data.query)

    return {
        "test_set": test_set,
        "test_batch_size": cfg.TEST.IMS_PER_BATCH,
        "num_query": num_query,
    }


@configurable(from_config=_test_loader_from_config)
def build_reid_test_loader(test_set, test_batch_size, num_query, num_workers=4):
    """
    Similar to `build_reid_train_loader`. This sampler coordinates all workers to produce
    the exact set of all samples
    This interface is experimental.

    Args:
        test_set:
        test_batch_size:
        num_query:
        num_workers:

    Returns:
        DataLoader: a torch DataLoader, that loads the given reid dataset, with
        the test-time transformation.

    Examples:
    ::
        data_loader = build_reid_test_loader(test_set, test_batch_size, num_query)
        # or, instantiate with a CfgNode:
        data_loader = build_reid_test_loader(cfg, "my_test")
    """

    mini_batch_size = test_batch_size // comm.get_world_size()
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, mini_batch_size, False)
    test_loader = DataLoaderX(
        comm.get_local_rank(),
        dataset=test_set,
        batch_sampler=batch_sampler,
        num_workers=num_workers,  # save some memory
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return test_loader, num_query


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out

    elif isinstance(elem, Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs
