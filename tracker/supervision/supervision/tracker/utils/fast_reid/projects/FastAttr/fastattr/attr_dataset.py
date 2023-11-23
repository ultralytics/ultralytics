# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch.utils.data import Dataset

from supervision.tracker.utils.fast_reid.fastreid.data.data_utils import read_image


class AttrDataset(Dataset):
    """Image Person Attribute Dataset"""

    def __init__(self, img_items, transform, attr_dict):
        self.img_items = img_items
        self.transform = transform
        self.attr_dict = attr_dict

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, labels = self.img_items[index]
        img = read_image(img_path)

        if self.transform is not None: img = self.transform(img)

        labels = torch.as_tensor(labels)

        return {
            "images": img,
            "targets": labels,
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        return len(self.attr_dict)

    @property
    def sample_weights(self):
        sample_weights = torch.zeros(self.num_classes, dtype=torch.float32)
        for _, attr in self.img_items:
            sample_weights += torch.as_tensor(attr)
        sample_weights /= len(self)
        return sample_weights
