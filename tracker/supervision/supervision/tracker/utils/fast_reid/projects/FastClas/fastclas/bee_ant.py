# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import glob
import os

from supervision.tracker.utils.fast_reid.fastreid.data.datasets import DATASET_REGISTRY
from supervision.tracker.utils.fast_reid.fastreid.data.datasets.bases import ImageDataset


__all__ = ["Hymenoptera"]


@DATASET_REGISTRY.register()
class Hymenoptera(ImageDataset):
    """This is a demo dataset for smoke test, you can refer to
    https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    """
    dataset_dir = 'hymenoptera_data'
    dataset_name = "hyt"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        train_dir = os.path.join(self.dataset_dir, "train")
        val_dir = os.path.join(self.dataset_dir, "val")

        required_files = [
            self.dataset_dir,
            train_dir,
            val_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(train_dir)
        val = self.process_dir(val_dir)

        super().__init__(train, val, [], **kwargs)

    def process_dir(self, data_dir):
        data = []
        all_dirs = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        for dir_name in all_dirs:
            all_imgs = glob.glob(os.path.join(data_dir, dir_name, "*.jpg"))
            for img_name in all_imgs:
                data.append([img_name, dir_name, '0'])
        return data
