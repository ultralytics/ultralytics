# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""
import logging
import os
import time

from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

from supervision.tracker.utils.fast_reid.fastreid.data.build import _root, build_reid_test_loader, build_reid_train_loader
from supervision.tracker.utils.fast_reid.fastreid.data.datasets import DATASET_REGISTRY
from supervision.tracker.utils.fast_reid.fastreid.data.transforms import build_transforms
from supervision.tracker.utils.fast_reid.fastreid.engine import hooks
from supervision.tracker.utils.fast_reid.fastreid.engine.defaults import DefaultTrainer, TrainerBase
from supervision.tracker.utils.fast_reid.fastreid.engine.train_loop import SimpleTrainer, AMPTrainer
from supervision.tracker.utils.fast_reid.fastreid.solver import build_optimizer
from supervision.tracker.utils.fast_reid.fastreid.utils import comm
from supervision.tracker.utils.fast_reid.fastreid.utils.checkpoint import Checkpointer
from supervision.tracker.utils.fast_reid.fastreid.utils.logger import setup_logger
from supervision.tracker.utils.fast_reid.fastreid.utils.params import ContiguousParams
from .face_data import MXFaceDataset
from .face_data import TestFaceDataset
from .face_evaluator import FaceEvaluator
from .modeling import PartialFC
from .pfc_checkpointer import PfcPeriodicCheckpointer, PfcCheckpointer
from .utils_amp import MaxClipGradScaler


class FaceTrainer(DefaultTrainer):
    def __init__(self, cfg):
        TrainerBase.__init__(self)

        logger = logging.getLogger('fastreid.partial-fc.trainer')
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for fastreid
            setup_logger()

        # Assume these objects must be constructed in this order.
        data_loader = self.build_train_loader(cfg)
        cfg = self.auto_scale_hyperparams(cfg, data_loader.dataset.num_classes)
        model = self.build_model(cfg)
        optimizer, param_wrapper = self.build_optimizer(cfg, model)

        if cfg.MODEL.HEADS.PFC.ENABLED:
            # fmt: off
            feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
            embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
            num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
            sample_rate   = cfg.MODEL.HEADS.PFC.SAMPLE_RATE
            cls_type      = cfg.MODEL.HEADS.CLS_LAYER
            scale         = cfg.MODEL.HEADS.SCALE
            margin        = cfg.MODEL.HEADS.MARGIN
            # fmt: on
            # Partial-FC module
            embedding_size = embedding_dim if embedding_dim > 0 else feat_dim
            self.pfc_module = PartialFC(embedding_size, num_classes, sample_rate, cls_type, scale, margin)
            self.pfc_optimizer, _ = build_optimizer(cfg, self.pfc_module, False)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            # ref to https://github.com/pytorch/pytorch/issues/22049 to set `find_unused_parameters=True`
            # for part of the parameters is not updated.
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            )

        if cfg.MODEL.HEADS.PFC.ENABLED:
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()
            grad_scaler = MaxClipGradScaler(mini_batch_size, 128 * mini_batch_size, growth_interval=100)
            self._trainer = PFCTrainer(model, data_loader, optimizer, param_wrapper,
                                       self.pfc_module, self.pfc_optimizer, cfg.SOLVER.AMP.ENABLED, grad_scaler)
        else:
            self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
                model, data_loader, optimizer, param_wrapper
            )

        self.iters_per_epoch = len(data_loader.dataset) // cfg.SOLVER.IMS_PER_BATCH
        self.scheduler = self.build_lr_scheduler(cfg, optimizer, self.iters_per_epoch)
        if cfg.MODEL.HEADS.PFC.ENABLED:
            self.pfc_scheduler = self.build_lr_scheduler(cfg, self.pfc_optimizer, self.iters_per_epoch)

        self.checkpointer = Checkpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            save_to_disk=comm.is_main_process(),
            optimizer=optimizer,
            **self.scheduler,
        )

        if cfg.MODEL.HEADS.PFC.ENABLED:
            self.pfc_checkpointer = PfcCheckpointer(
                self.pfc_module,
                cfg.OUTPUT_DIR,
                optimizer=self.pfc_optimizer,
                **self.pfc_scheduler,
            )

        self.start_epoch = 0
        self.max_epoch = cfg.SOLVER.MAX_EPOCH
        self.max_iter = self.max_epoch * self.iters_per_epoch
        self.warmup_iters = cfg.SOLVER.WARMUP_ITERS
        self.delay_epochs = cfg.SOLVER.DELAY_EPOCHS
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def build_hooks(self):
        ret = super().build_hooks()

        if self.cfg.MODEL.HEADS.PFC.ENABLED:
            # Make sure checkpointer is after writer
            ret.insert(
                len(ret) - 1,
                PfcPeriodicCheckpointer(self.pfc_checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD)
            )
            # partial fc scheduler hook
            ret.append(
                hooks.LRScheduler(self.pfc_optimizer, self.pfc_scheduler)
            )
        return ret

    def resume_or_load(self, resume=True):
        # Backbone loading state_dict
        super().resume_or_load(resume)
        # Partial-FC loading state_dict
        if self.cfg.MODEL.HEADS.PFC.ENABLED:
            self.pfc_checkpointer.resume_or_load('', resume=resume)

    @classmethod
    def build_train_loader(cls, cfg):
        path_imgrec = cfg.DATASETS.REC_PATH
        if path_imgrec != "":
            transforms = build_transforms(cfg, is_train=True)
            train_set = MXFaceDataset(path_imgrec, transforms)
            return build_reid_train_loader(cfg, train_set=train_set)
        else:
            return DefaultTrainer.build_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        dataset = DATASET_REGISTRY.get(dataset_name)(root=_root)
        test_set = TestFaceDataset(dataset.carray, dataset.is_same)
        data_loader, _ = build_reid_test_loader(cfg, test_set=test_set)
        return data_loader, test_set.labels

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(cfg.OUTPUT_DIR, "visualization")
        data_loader, labels = cls.build_test_loader(cfg, dataset_name)
        return data_loader, FaceEvaluator(cfg, labels, dataset_name, output_dir)


class PFCTrainer(SimpleTrainer):
    """
    Author: {Xiang An, Yang Xiao, XuHan Zhu} in DeepGlint,
    Partial FC: Training 10 Million Identities on a Single Machine
    See the original paper:
    https://arxiv.org/abs/2010.05222
    code based on:
    https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/partial_fc.py
    """

    def __init__(self, model, data_loader, optimizer, param_wrapper, pfc_module, pfc_optimizer, amp_enabled,
                 grad_scaler):
        super().__init__(model, data_loader, optimizer, param_wrapper)

        self.pfc_module = pfc_module
        self.pfc_optimizer = pfc_optimizer
        self.amp_enabled = amp_enabled

        self.grad_scaler = grad_scaler

    def run_step(self):
        assert self.model.training, "[PFCTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        features, targets = self.model(data)

        # Partial-fc backward
        f_grad, loss_v = self.pfc_module.forward_backward(features, targets, self.pfc_optimizer)

        if self.amp_enabled:
            features.backward(self.grad_scaler.scale(f_grad))
            self.grad_scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), max_norm=5, norm_type=2)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            features.backward(f_grad)
            clip_grad_norm_(self.model.parameters(), max_norm=5, norm_type=2)
            self.optimizer.step()

        loss_dict = {"loss_cls": loss_v}
        self._write_metrics(loss_dict, data_time)

        self.pfc_optimizer.step()
        self.pfc_module.update()
        self.optimizer.zero_grad()
        self.pfc_optimizer.zero_grad()
        if isinstance(self.param_wrapper, ContiguousParams):
            self.param_wrapper.assert_buffer_is_valid()
