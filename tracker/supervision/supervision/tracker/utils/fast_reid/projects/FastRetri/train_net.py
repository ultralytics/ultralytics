#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys

sys.path.append('.')

from supervision.tracker.utils.fast_reid.fastreid.config import get_cfg
from supervision.tracker.utils.fast_reid.fastreid.engine import default_argument_parser, default_setup, launch
from supervision.tracker.utils.fast_reid.fastreid.utils.checkpoint import Checkpointer
from supervision.tracker.utils.fast_reid.fastreid.engine.defaults import DefaultTrainer

from fastretri import *


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        data_loader, num_query = cls.build_test_loader(cfg, dataset_name)
        return data_loader, RetriEvaluator(cfg, num_query, output_dir)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_retri_config(cfg)
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
        model = Trainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)

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
