# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.
The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""

import argparse
import logging
import os
import sys
from collections import OrderedDict

import torch
from torch.nn.parallel import DistributedDataParallel

from supervision.tracker.utils.fast_reid.fastreid.data import build_reid_test_loader, build_reid_train_loader
from supervision.tracker.utils.fast_reid.fastreid.evaluation import (ReidEvaluator,
                                 inference_on_dataset, print_csv_format)
from supervision.tracker.utils.fast_reid.fastreid.modeling.meta_arch import build_model
from supervision.tracker.utils.fast_reid.fastreid.solver import build_lr_scheduler, build_optimizer
from supervision.tracker.utils.fast_reid.fastreid.utils import comm
from supervision.tracker.utils.fast_reid.fastreid.utils.checkpoint import Checkpointer
from supervision.tracker.utils.fast_reid.fastreid.utils.collect_env import collect_env_info
from supervision.tracker.utils.fast_reid.fastreid.utils.env import seed_all_rng
from supervision.tracker.utils.fast_reid.fastreid.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from supervision.tracker.utils.fast_reid.fastreid.utils.file_io import PathManager
from supervision.tracker.utils.fast_reid.fastreid.utils.logger import setup_logger
from . import hooks
from .train_loop import TrainerBase, AMPTrainer, SimpleTrainer

__all__ = ["default_argument_parser", "default_setup", "DefaultPredictor", "DefaultTrainer"]


def default_argument_parser():
    """
    Create a parser with some common arguments used by fastreid users.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="fastreid Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:
    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory
    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    # setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, PathManager.open(args.config_file, "r").read()
            )
        )

    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng()

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config.
    The predictor takes an BGR image, resizes it to the specified resolution,
    runs the model and produces a dict of predictions.
    This predictor takes care of model loading and input preprocessing for you.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.
    Attributes:
    Examples:
    .. code-block:: python
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.cfg.defrost()
        self.cfg.MODEL.BACKBONE.PRETRAIN = False
        self.model = build_model(self.cfg)
        self.model.eval()

        Checkpointer(self.model).load(cfg.MODEL.WEIGHTS)

    def __call__(self, image):
        """
        Args:
            image (torch.tensor): an image tensor of shape (B, C, H, W).
        Returns:
            predictions (torch.tensor): the output features of the model
        """
        inputs = {"images": image.to(self.model.device)}
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            predictions = self.model(inputs)
        return predictions.cpu()


class DefaultTrainer(TrainerBase):
    """
    A trainer with default training logic. Compared to `SimpleTrainer`, it
    contains the following logic in addition:
    1. Create model, optimizer, scheduler, dataloader from the given config.
    2. Load a checkpoint or `cfg.MODEL.WEIGHTS`, if exists.
    3. Register a few common hooks.
    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`SimpleTrainer` are too much for research.
    The code of this class has been annotated about restrictive assumptions it mades.
    When they do not work for you, you're encouraged to:
    1. Overwrite methods of this class, OR:
    2. Use :class:`SimpleTrainer`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.
    Also note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in fastreid.
    To obtain more stable behavior, write your own training logic with other public APIs.
    Attributes:
        scheduler:
        checkpointer:
        cfg (CfgNode):
    Examples:
    .. code-block:: python
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """

        super().__init__()

        logger = logging.getLogger("fastreid")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for fastreid
            setup_logger()

        # Assume these objects must be constructed in this order.
        data_loader = self.build_train_loader(cfg)
        cfg = self.auto_scale_hyperparams(cfg, data_loader.dataset.num_classes)
        model = self.build_model(cfg)
        optimizer, param_wrapper = self.build_optimizer(cfg, model)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            # ref to https://github.com/pytorch/pytorch/issues/22049 to set `find_unused_parameters=True`
            # for part of the parameters is not updated.
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            )

        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer, param_wrapper
        )

        self.iters_per_epoch = len(data_loader.dataset) // cfg.SOLVER.IMS_PER_BATCH
        self.scheduler = self.build_lr_scheduler(cfg, optimizer, self.iters_per_epoch)

        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = Checkpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            save_to_disk=comm.is_main_process(),
            optimizer=optimizer,
            **self.scheduler,
        )

        self.start_epoch = 0
        self.max_epoch = cfg.SOLVER.MAX_EPOCH
        self.max_iter = self.max_epoch * self.iters_per_epoch
        self.warmup_iters = cfg.SOLVER.WARMUP_ITERS
        self.delay_epochs = cfg.SOLVER.DELAY_EPOCHS
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)

        if resume and self.checkpointer.has_checkpoint():
            self.start_epoch = checkpoint.get("epoch", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        logger = logging.getLogger(__name__)
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
        cfg.DATASETS.NAMES = tuple([cfg.TEST.PRECISE_BN.DATASET])  # set dataset name for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
        ]

        if cfg.TEST.PRECISE_BN.ENABLED and hooks.get_bn_modules(self.model):
            logger.info("Prepare precise BN dataset")
            ret.append(hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            ))

        if len(cfg.MODEL.FREEZE_LAYERS) > 0 and cfg.SOLVER.FREEZE_ITERS > 0:
            ret.append(hooks.LayerFreeze(
                self.model,
                cfg.MODEL.FREEZE_LAYERS,
                cfg.SOLVER.FREEZE_ITERS,
            ))

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation before checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), 200))

        return ret

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.
        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        It is now implemented by:
        .. code-block:: python
            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]
        """
        # Assume the default print/log frequency.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    def train(self):
        """
        Run training.
        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_epoch, self.max_epoch, self.iters_per_epoch)
        if comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            return self._last_eval_results

    def run_step(self):
        self._trainer.iter = self.iter
        self._trainer.run_step()

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`fastreid.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        It now calls :func:`fastreid.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer, iters_per_epoch):
        """
        It now calls :func:`fastreid.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer, iters_per_epoch)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`fastreid.data.build_reid_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        logger = logging.getLogger(__name__)
        logger.info("Prepare training set")
        return build_reid_train_loader(cfg, combineall=cfg.DATASETS.COMBINEALL)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`fastreid.data.build_reid_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_reid_test_loader(cfg, dataset_name=dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        data_loader, num_query = cls.build_test_loader(cfg, dataset_name)
        return data_loader, ReidEvaluator(cfg, num_query, output_dir)

    @classmethod
    def test(cls, cfg, model):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TESTS):
            logger.info("Prepare testing set")
            try:
                data_loader, evaluator = cls.build_evaluator(cfg, dataset_name)
            except NotImplementedError:
                logger.warn(
                    "No evaluator found. implement its `build_evaluator` method."
                )
                results[dataset_name] = {}
                continue
            results_i = inference_on_dataset(model, data_loader, evaluator, flip_test=cfg.TEST.FLIP.ENABLED)
            results[dataset_name] = results_i

            if comm.is_main_process():
                assert isinstance(
                    results, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                results_i['dataset'] = dataset_name
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]

        return results

    @staticmethod
    def auto_scale_hyperparams(cfg, num_classes):
        r"""
        This is used for auto-computation actual training iterations,
        because some hyper-param, such as MAX_ITER, means training epochs rather than iters,
        so we need to convert specific hyper-param to training iterations.
        """
        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        # If you don't hard-code the number of classes, it will compute the number automatically
        if cfg.MODEL.HEADS.NUM_CLASSES == 0:
            output_dir = cfg.OUTPUT_DIR
            cfg.MODEL.HEADS.NUM_CLASSES = num_classes
            logger = logging.getLogger(__name__)
            logger.info(f"Auto-scaling the num_classes={cfg.MODEL.HEADS.NUM_CLASSES}")

            # Update the saved config file to make the number of classes valid
            if comm.is_main_process() and output_dir:
                # Note: some of our scripts may expect the existence of
                # config.yaml in output directory
                path = os.path.join(output_dir, "config.yaml")
                with PathManager.open(path, "w") as f:
                    f.write(cfg.dump())

        if frozen: cfg.freeze()

        return cfg


# Access basic attributes from the underlying trainer
for _attr in ["model", "data_loader", "optimizer", "grad_scaler"]:
    setattr(DefaultTrainer, _attr, property(lambda self, x=_attr: getattr(self._trainer, x, None)))
