# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""
import logging
import sys

sys.path.append('.')

from supervision.tracker.utils.fast_reid.fastreid.config import get_cfg
from supervision.tracker.utils.fast_reid.fastreid.engine import DefaultTrainer
from supervision.tracker.utils.fast_reid.fastreid.engine import default_argument_parser, default_setup, launch
from supervision.tracker.utils.fast_reid.fastreid.utils.checkpoint import Checkpointer
from supervision.tracker.utils.fast_reid.fastreid.data.datasets import DATASET_REGISTRY
from supervision.tracker.utils.fast_reid.fastreid.data.build import _root, build_reid_train_loader, build_reid_test_loader
from supervision.tracker.utils.fast_reid.fastreid.data.transforms import build_transforms
from supervision.tracker.utils.fast_reid.fastreid.utils import comm

from fastattr import *


class AttrTrainer(DefaultTrainer):
    sample_weights = None

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`fastreid.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = DefaultTrainer.build_model(cfg)
        if cfg.MODEL.LOSSES.BCE.WEIGHT_ENABLED and \
                AttrTrainer.sample_weights is not None:
            setattr(model, "sample_weights", AttrTrainer.sample_weights.to(model.device))
        else:
            setattr(model, "sample_weights", None)
        return model

    @classmethod
    def build_train_loader(cls, cfg):

        logger = logging.getLogger("fastreid.attr_dataset")
        train_items = list()
        attr_dict = None
        for d in cfg.DATASETS.NAMES:
            dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL)
            if comm.is_main_process():
                dataset.show_train()
            if attr_dict is not None:
                assert attr_dict == dataset.attr_dict, f"attr_dict in {d} does not match with previous ones"
            else:
                attr_dict = dataset.attr_dict
            train_items.extend(dataset.train)

        train_transforms = build_transforms(cfg, is_train=True)
        train_set = AttrDataset(train_items, train_transforms, attr_dict)

        data_loader = build_reid_train_loader(cfg, train_set=train_set)
        AttrTrainer.sample_weights = data_loader.dataset.sample_weights
        return data_loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        dataset = DATASET_REGISTRY.get(dataset_name)(root=_root)
        attr_dict = dataset.attr_dict
        if comm.is_main_process():
            dataset.show_test()
        test_items = dataset.test

        test_transforms = build_transforms(cfg, is_train=False)
        test_set = AttrDataset(test_items, test_transforms, attr_dict)
        data_loader, _ = build_reid_test_loader(cfg, test_set=test_set)
        return data_loader

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        data_loader = cls.build_test_loader(cfg, dataset_name)
        return data_loader, AttrEvaluator(cfg, output_folder)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_attr_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = AttrTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = AttrTrainer.test(cfg, model)
        return res

    trainer = AttrTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
