#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import json
import logging
import os
import sys

sys.path.append('.')

from supervision.tracker.utils.fast_reid.fastreid.config import get_cfg
from supervision.tracker.utils.fast_reid.fastreid.engine import default_argument_parser, default_setup, launch
from supervision.tracker.utils.fast_reid.fastreid.utils.checkpoint import Checkpointer, PathManager

from fastclas import *


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
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
        model = ClasTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        try:
            output_dir = os.path.dirname(cfg.MODEL.WEIGHTS)
            path = os.path.join(output_dir, "idx2class.json")
            with PathManager.open(path, 'r') as f:
                idx2class = json.load(f)
            ClasTrainer.idx2class = idx2class
        except:
            logger = logging.getLogger("fastreid.fastclas")
            logger.info(f"Cannot find idx2class dict in {os.path.dirname(cfg.MODEL.WEIGHTS)}")

        res = ClasTrainer.test(cfg, model)
        return res

    trainer = ClasTrainer(cfg)

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
