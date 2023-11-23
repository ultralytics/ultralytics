# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import json
import logging
import os

from supervision.tracker.utils.fast_reid.fastreid.data.build import _root
from supervision.tracker.utils.fast_reid.fastreid.data.build import build_reid_train_loader, build_reid_test_loader
from supervision.tracker.utils.fast_reid.fastreid.data.datasets import DATASET_REGISTRY
from supervision.tracker.utils.fast_reid.fastreid.data.transforms import build_transforms
from supervision.tracker.utils.fast_reid.fastreid.engine import DefaultTrainer
from supervision.tracker.utils.fast_reid.fastreid.evaluation.clas_evaluator import ClasEvaluator
from supervision.tracker.utils.fast_reid.fastreid.utils import comm
from supervision.tracker.utils.fast_reid.fastreid.utils.checkpoint import PathManager
from .dataset import ClasDataset


class ClasTrainer(DefaultTrainer):
    idx2class = None

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`fastreid.data.build_reid_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        logger = logging.getLogger("fastreid.clas_dataset")
        logger.info("Prepare training set")

        train_items = list()
        for d in cfg.DATASETS.NAMES:
            data = DATASET_REGISTRY.get(d)(root=_root)
            if comm.is_main_process():
                data.show_train()
            train_items.extend(data.train)
        transforms = build_transforms(cfg, is_train=True)
        train_set = ClasDataset(train_items, transforms)
        cls.idx2class = train_set.idx_to_class

        data_loader = build_reid_train_loader(cfg, train_set=train_set)
        return data_loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`fastreid.data.build_reid_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        data = DATASET_REGISTRY.get(dataset_name)(root=_root)
        if comm.is_main_process():
            data.show_test()
        transforms = build_transforms(cfg, is_train=False)

        test_set = ClasDataset(data.query, transforms, cls.idx2class)
        data_loader, _ = build_reid_test_loader(cfg, test_set=test_set)
        return data_loader

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        data_loader = cls.build_test_loader(cfg, dataset_name)
        return data_loader, ClasEvaluator(cfg, output_dir)

    @staticmethod
    def auto_scale_hyperparams(cfg, num_classes):
        cfg = DefaultTrainer.auto_scale_hyperparams(cfg, num_classes)

        # Save index to class dictionary
        output_dir = cfg.OUTPUT_DIR
        if comm.is_main_process() and output_dir:
            path = os.path.join(output_dir, "idx2class.json")
            with PathManager.open(path, "w") as f:
                json.dump(ClasTrainer.idx2class, f)

        return cfg
