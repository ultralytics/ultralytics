# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import glob
import os
from collections import defaultdict

from supervision.tracker.utils.fast_reid.fastreid.data.datasets import DATASET_REGISTRY
from supervision.tracker.utils.fast_reid.fastreid.data.datasets.bases import ImageDataset

__all__ = ["NAIC20_R2", "NAIC20_R2CNV", "NAIC20_R1", "NAIC20_R1CNV", "NAIC19", "NAIC20_R2A", ]


@DATASET_REGISTRY.register()
class NAIC20_R2(ImageDataset):
    dataset_name = "naic20_r2"
    dataset_dir = "naic/2020_NAIC/fusai/train"

    def __init__(self, root="datasets", rm_lt=False, **kwargs):
        self.root = root

        self.data_path = os.path.join(self.root, self.dataset_dir, "images")
        self.train_label = os.path.join(self.root, self.dataset_dir, "naic20r2_train_list_clean.txt")
        self.query_label = os.path.join(self.root, self.dataset_dir, "val_query.txt")
        self.gallery_label = os.path.join(self.root, self.dataset_dir, "val_gallery.txt")

        required_files = [self.train_label, self.query_label, self.gallery_label]
        self.check_before_run(required_files)

        all_train = self.process_train(self.train_label)

        # fmt: off
        if rm_lt: train = self.remove_longtail(all_train)
        else:     train = all_train
        # fmt: on

        query, gallery = self.process_test(self.query_label, self.gallery_label)

        super().__init__(train, query, gallery, **kwargs)

    def process_train(self, label_path):
        with open(label_path, 'r') as f:
            data_list = [i.strip('\n') for i in f.readlines()]

        img_paths = []
        for data_info in data_list:
            img_name, pid = data_info.split(":")
            img_path = os.path.join(self.data_path, img_name)
            pid = self.dataset_name + "_" + pid
            camid = self.dataset_name + '_0'
            img_paths.append([img_path, pid, camid])

        return img_paths

    def process_test(self, query_path, gallery_path):
        with open(query_path, 'r') as f:
            query_list = [i.strip('\n') for i in f.readlines()]

        with open(gallery_path, 'r') as f:
            gallery_list = [i.strip('\n') for i in f.readlines()]

        query_paths = []
        for data in query_list:
            img_name, pid = data.split(':')
            img_path = os.path.join(self.data_path, img_name)
            camid = '0'
            query_paths.append([img_path, int(pid), camid])

        gallery_paths = []
        for data in gallery_list:
            img_name, pid = data.split(':')
            img_path = os.path.join(self.data_path, img_name)
            camid = '1'
            gallery_paths.append([img_path, int(pid), camid])

        return query_paths, gallery_paths

    @classmethod
    def remove_longtail(cls, all_train):
        # 建立 id 到 image 的字典
        pid2data = defaultdict(list)
        for item in all_train:
            pid2data[item[1]].append(item)

        train = []
        for pid, data in pid2data.items():
            # 如果 id 只有一张图片，去掉这个 id
            if len(data) == 1: continue
            train.extend(data)

        return train


@DATASET_REGISTRY.register()
class NAIC20_R2CNV(NAIC20_R2, ImageDataset):
    dataset_name = 'naic20_r2cnv'
    dataset_dir = "naic/2020_NAIC/fusai/train"

    def __init__(self, root="datasets", rm_lt=False, **kwargs):
        self.root = root

        self.data_path = os.path.join(self.root, self.dataset_dir, "images_convert")
        self.train_label = os.path.join(self.root, self.dataset_dir, "naic20r2_train_list_clean.txt")
        self.query_label = os.path.join(self.root, self.dataset_dir, "val_query.txt")
        self.gallery_label = os.path.join(self.root, self.dataset_dir, "val_gallery.txt")

        required_files = [self.train_label, self.query_label, self.gallery_label]
        self.check_before_run(required_files)

        all_train = self.process_train(self.train_label)[:53000]

        # fmt: off
        if rm_lt: train = self.remove_longtail(all_train)
        else:     train = all_train
        # fmt: on

        ImageDataset.__init__(self, train, query=[], gallery=[], **kwargs)


@DATASET_REGISTRY.register()
class NAIC20_R1(NAIC20_R2):
    dataset_name = "naic20_r1"
    dataset_dir = 'naic/2020_NAIC/chusai/train'

    def __init__(self, root="datasets", rm_lt=False, **kwargs):
        self.root = root

        self.data_path = os.path.join(self.root, self.dataset_dir, "images")
        self.train_label = os.path.join(self.root, self.dataset_dir, "label.txt")

        required_files = [self.train_label]
        self.check_before_run(required_files)

        all_train = self.process_train(self.train_label)[:40188]

        # fmt: off
        if rm_lt: train = self.remove_longtail(all_train)
        else:     train = all_train
        # fmt: on

        super(NAIC20_R2, self).__init__(train, [], [], **kwargs)


@DATASET_REGISTRY.register()
class NAIC20_R1CNV(NAIC20_R2):
    dataset_name = 'naic20_r1cnv'
    dataset_dir = "naic/2020_NAIC/chusai/train"

    def __init__(self, root="datasets", rm_lt=False, **kwargs):
        self.root = root

        self.data_path = os.path.join(self.root, self.dataset_dir, "images_convert")
        self.train_label = os.path.join(self.root, self.dataset_dir, "label.txt")

        required_files = [self.train_label]
        self.check_before_run(required_files)

        all_train = self.process_train(self.train_label)[:40188]

        # fmt: off
        if rm_lt: train = self.remove_longtail(all_train)
        else:     train = all_train
        # fmt: on

        super(NAIC20_R2, self).__init__(train, [], [], **kwargs)


@DATASET_REGISTRY.register()
class NAIC19(NAIC20_R2):
    dataset_name = "naic19"
    dataset_dir = "naic/2019_NAIC/fusai"

    def __init__(self, root='datasets', rm_lt=False, **kwargs):
        self.root = root

        self.data_path = os.path.join(self.root, self.dataset_dir)
        self.train_label = os.path.join(self.root, self.dataset_dir, 'train_list_clean.txt')

        required_files = [self.train_label]
        self.check_before_run(required_files)

        all_train = self.process_train(self.train_label)

        # fmt: off
        if rm_lt: train = self.remove_longtail(all_train)
        else:     train = all_train
        # fmt: on

        super(NAIC20_R2, self).__init__(train, [], [], **kwargs)

    def process_train(self, label_path):
        with open(label_path, 'r') as f:
            data_list = [i.strip('\n') for i in f.readlines()]

        img_paths = []
        for data_info in data_list:
            img_name, pid = data_info.split(" ")
            img_path = os.path.join(self.data_path, img_name)
            pid = self.dataset_name + "_" + pid
            camid = self.dataset_name + '_0'
            img_paths.append([img_path, pid, camid])

        return img_paths


@DATASET_REGISTRY.register()
class NAIC20_R2A(ImageDataset):
    dataset_name = "naic20_b"
    dataset_dir = 'naic/round2/image_A'

    def __init__(self, root='datasets', **kwargs):
        self.root = root

        self.query_path = os.path.join(self.root, self.dataset_dir, "query")
        self.gallery_path = os.path.join(self.root, self.dataset_dir, "gallery")

        query = self.process_test(self.query_path)
        gallery = self.process_test(self.gallery_path)

        super().__init__([], query, gallery)

    def process_test(self, test_path):
        img_paths = glob.glob(os.path.join(test_path, "*.png"))

        data = []
        for img_path in img_paths:
            img_name = img_path.split("/")[-1]
            data.append([img_path, img_name, "naic_0"])
        return data
