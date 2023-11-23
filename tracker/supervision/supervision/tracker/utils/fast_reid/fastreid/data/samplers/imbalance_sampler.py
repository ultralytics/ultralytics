# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

# based on:
# https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py


import itertools
from typing import Optional, List, Callable

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

from supervision.tracker.utils.fast_reid.fastreid.utils import comm


class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        data_source: a list of data items
        size: number of samples to draw
    """

    def __init__(self, data_source: List, size: int = None, seed: Optional[int] = None,
                 callback_get_label: Callable = None):
        self.data_source = data_source
        # consider all elements in the dataset
        self.indices = list(range(len(data_source)))
        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self._size = len(self.indices) if size is None else size
        self.callback_get_label = callback_get_label

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(data_source, idx)
            label_to_count[label] = label_to_count.get(label, 0) + 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(data_source, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        else:
            return dataset[idx][1]

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            for i in torch.multinomial(self.weights, self._size, replacement=True):
                yield self.indices[i]
