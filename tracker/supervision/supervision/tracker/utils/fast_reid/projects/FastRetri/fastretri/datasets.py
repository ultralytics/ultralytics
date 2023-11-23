# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os

from supervision.tracker.utils.fast_reid.fastreid.data.datasets import DATASET_REGISTRY
from supervision.tracker.utils.fast_reid.fastreid.data.datasets.bases import ImageDataset

__all__ = ["Cars196", "CUB", "SOP", "InShop"]


@DATASET_REGISTRY.register()
class Cars196(ImageDataset):
    dataset_dir = 'Cars_196'
    dataset_name = "cars"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        train_file = os.path.join(self.dataset_dir, "train.txt")
        test_file = os.path.join(self.dataset_dir, "test.txt")

        required_files = [
            self.dataset_dir,
            train_file,
            test_file,
        ]
        self.check_before_run(required_files)

        train = self.process_label_file(train_file, is_train=True)
        query = self.process_label_file(test_file, is_train=False)

        super(Cars196, self).__init__(train, query, [], **kwargs)

    def process_label_file(self, file, is_train):
        data_list = []
        with open(file, 'r') as f:
            lines = f.read().splitlines()

        for line in lines:
            img_name, label = line.split(',')
            if is_train:
                label = self.dataset_name + '_' + str(label)

            data_list.append((os.path.join(self.dataset_dir, img_name), label, '0'))

        return data_list


@DATASET_REGISTRY.register()
class CUB(Cars196):
    dataset_dir = "CUB_200_2011"
    dataset_name = "cub"


@DATASET_REGISTRY.register()
class SOP(Cars196):
    dataset_dir = "Stanford_Online_Products"
    dataset_name = "sop"


@DATASET_REGISTRY.register()
class InShop(Cars196):
    dataset_dir = "InShop"
    dataset_name = "inshop"

    def __init__(self, root="datasets", **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        train_file = os.path.join(self.dataset_dir, "train.txt")
        query_file = os.path.join(self.dataset_dir, "test_query.txt")
        gallery_file = os.path.join(self.dataset_dir, "test_gallery.txt")

        required_files = [
            train_file,
            query_file,
            gallery_file,
        ]
        self.check_before_run(required_files)

        train = self.process_label_file(train_file, True)
        query = self.process_label_file(query_file, False)
        gallery = self.process_label_file(gallery_file, False)

        super(Cars196, self).__init__(train, query, gallery, **kwargs)
