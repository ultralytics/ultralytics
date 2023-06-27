# Ultralytics YOLO 🚀, AGPL-3.0 license

import collections
from copy import deepcopy

from .augment import LetterBox


class MixAndRectDataset:
    """
    A dataset class that applies mosaic and mixup transformations as well as rectangular training.

    Attributes:
        dataset: The base dataset.
        imgsz: The size of the images in the dataset.
    """

    def __init__(self, dataset):
        """
        Args:
            dataset (BaseDataset): The base dataset to apply transformations to.
        """
        self.dataset = dataset
        self.imgsz = dataset.imgsz

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Applies mosaic, mixup and rectangular training transformations to an item in the dataset.

        Args:
            index (int): Index of the item in the dataset.

        Returns:
            (dict): A dictionary containing the transformed item data.
        """
        labels = deepcopy(self.dataset[index])
        for transform in self.dataset.transforms.tolist():
            # Mosaic and mixup
            if hasattr(transform, 'get_indexes'):
                indexes = transform.get_indexes(self.dataset)
                if not isinstance(indexes, collections.abc.Sequence):
                    indexes = [indexes]
                labels['mix_labels'] = [deepcopy(self.dataset[index]) for index in indexes]
            if self.dataset.rect and isinstance(transform, LetterBox):
                transform.new_shape = self.dataset.batch_shapes[self.dataset.batch[index]]
            labels = transform(labels)
            if 'mix_labels' in labels:
                labels.pop('mix_labels')
        return labels
