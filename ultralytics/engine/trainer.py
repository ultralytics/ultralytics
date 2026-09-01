# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolo26n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

from __future__ import annotations

import csv
import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics import __version__
from ultralytics.cfg import _YOLO_CLI_COMMAND, get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset, convert_ndjson_to_yolo_if_needed
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.optim import MuSGD
from ultralytics.utils import (
    DEFAULT_CFG,
    GIT,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.torch_utils import (
    TORCH_1_11,
    TORCH_2_0,
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    attempt_compile,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    get_torch_device_backend,
    init_seeds,
    one_cycle,
    parse_device,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
    unwrap_model,
)


class BaseTrainer:
    """A base class for creating trainers.

    This class provides the foundation for training YOLO models, handling the training loop, validation, checkpointing,
    and various training utilities. It supports both single-GPU and multi-GPU distributed training.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Whether Automatic Mixed Precision is enabled.
        scaler (torch.amp.GradScaler): Gradient scaler for AMP.
        data (dict): Dataset dictionary containing paths and metadata.
        ema (ModelEMA): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (callable): Learning rate scheduling function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (torch.Tensor): Current loss value.
        tloss (dict): Running mean of loss items.
        loss_names (tuple): Names of loss items, derived from the loss dict returned by the criterion on the first
            batch.
        csv (Path): Path to results CSV file.
        metrics (dict): Dictionary of metrics.
        plots (dict): Dictionary of plots.

    Methods:
        train: Execute the training process.
        validate: Run validation on the val set.
        save_model: Save model training checkpoints.
        get_dataset: Get train and validation datasets.
        setup_model: Load, create, or download model.
        build_optimizer: Construct an optimizer for the model.

    Examples:
        Initialize a trainer and start training
        >>> trainer = BaseTrainer(cfg="config.yaml")
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize the BaseTrainer class.

        Args:
            cfg (str | dict | SimpleNamespace, optional): Path to a configuration file or configuration object.
            overrides (dict, optional): Configuration overrides.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        self.args = get_cfg(cfg, overrides)
        if getattr(self.args, "augmentations", None) and not isinstance(self.args.augmentations[0], dict):
            import albumentations as A

            self.args.augmentations = [A.to_dict(t) for t in self.args.augmentations]  # YAML/pickle-safe, DDP-safe
        self.check_resume(overrides)
        self.args.device = parse_device(self.args.device)  # canonical string, resolves '-1' auto-selection once
        self.device = select_device(self.args.device)
        self.accelerator = get_torch_device_backend(self.device) if self.device.type not in {"cpu", "mps"} else None
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            YAML.save(self.save_dir / "args.yaml", vars(self.args))  # save run args
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100  # in case users accidentally pass epochs=None with timed training
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Callbacks - initialize early so on_pretrain_routine_start can capture original args.data
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

        # Device count in the launching process; distinct from utils.WORLD_SIZE set in spawned DDP workers
        if self.device.type in {"cpu", "mps"}:
            world_size = 0
        else:  # i.e. device='0', '0,1,2,3', 'npu:0', or '' auto-selecting a single GPU
            world_size = len(self.args.device.split(",")) if self.args.device else 1

        self.ddp = world_size > 1 and LOCAL_RANK == -1  # spawn DDP workers unless already one
        self.world_size = world_size
        # Run on_pretrain_routine_start before get_dataset() to capture original args.data (e.g., ul:// URIs)
        if RANK in {-1, 0} and not self.ddp:
            callbacks.add_integration_callbacks(self)
            self.run_callbacks("on_pretrain_routine_start")

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolo26n -> yolo26n.pt
        with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times
            self.data = self.get_dataset()

        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ()
        self.csv = self.save_dir / "results.csv"
        if self.csv.exists() and not self.args.resume:
            self.csv.unlink()
        self.plot_idx = [0, 1, 2]
        self.nan_recovery_attempts = 0

    def add_callback(self, event: str, callback):
        """Append the given callback to the event's callback list."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Override the existing callbacks with the given callback for the specified event."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Execute the training process, using DDP subprocess for multi-GPU or direct training for single-GPU."""
        # Run subprocess if DDP training, else train normally
        try:
            if self.ddp:
                # Argument checks
                if self.args.rect:
                    LOGGER.warning("'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                    self.args.rect = False
                if self.args.batch < 1.0:
                    raise ValueError(
                        "AutoBatch with batch<1 not supported for Multi-GPU training, "
                        f"please specify a valid batch size multiple of GPU count {self.world_size}, i.e. batch={self.world_size * 8}."
                    )

                # Command
                cmd, file = None, None
                try:
                    cmd, file = generate_ddp_command(self)
                    LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                finally:
                    if file is not None:
                        ddp_cleanup(self, str(file))

            else:
                self._do_train()
        finally:
            unset_deterministic()  # never leave deterministic state on, including the DDP parent and failed runs
        if not self.ddp:
            self.run_callbacks("teardown")

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _get_warmup_iterations(self, num_batches):
        """Return warmup iterations, leaving at least the final epoch for regular training."""
        warmup_epochs = min(self.args.warmup_epochs, max(self.epochs - 1, 0))
        return round(warmup_epochs * num_batches) if warmup_epochs > 0 else 0

    def _setup_ddp(self):
        """Initialize and set the DistributedDataParallel parameters for training."""
        device_type = self.args.device.split(":", 1)[0]
        device_type = device_type if device_type in {"npu", "xpu"} else "cuda"
        devices = self.args.device.split(":", 1)[-1].split(",")
        index = int(devices[LOCAL_RANK])  # world_size > 1 guarantees a multi-device string
        self.device = torch.device(device_type, index)
        self.accelerator = get_torch_device_backend(self.device)
        self.accelerator.set_device(index)
        if device_type == "cuda":
            os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        elif device_type == "xpu" and not (hasattr(dist, "is_xccl_available") and dist.is_xccl_available()):
            raise RuntimeError("Multi-XPU training requires XCCL, which is not available in this PyTorch build.")
        dist.init_process_group(
            backend={"npu": "hccl", "xpu": "xccl"}.get(device_type, "nccl" if dist.is_nccl_available() else "gloo"),
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=self.world_size,
        )

    def _find_lr(self, num_it: int = 100, lr_min: float = 1e-6, lr_max: float = 1.0):
        """Fit the initial learning rate and warmup to the current model and dataset.

        Sweep the learning rate exponentially over at most one tenth of the run. Fit the rate halfway in log space
        between the fastest loss descent and the highest rate that still improves the loss. Restore model, optimizer,
        scaler, and dataloader state before returning.

        Args:
            num_it (int): Upper bound on optimizer steps in the sweep.
            lr_min (float): Learning rate of the first step.
            lr_max (float): Learning rate of the last step.
        """
        nb = len(self.train_loader) // self.accumulate
        num_it = min(num_it, nb * self.epochs // 10)
        if nb < 10 or num_it < 50:
            LOGGER.info(f"{colorstr('LR finder:')} run too short to sweep, using the 'lr0' equation")
            return
        pg = self.optimizer.param_groups
        base = min(g["lr"] for g in pg)
        ratios = [g["lr"] / base for g in pg]
        model_state = {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}
        optimizer_state, scaler_state = deepcopy(self.optimizer.state_dict()), self.scaler.state_dict()

        self._model_train()
        lrs = np.logspace(math.log10(lr_min), math.log10(lr_max), num_it)
        losses, total, loader = [], torch.zeros(1, device=self.device), iter(self.train_loader)
        desc = f"{colorstr('LR finder:')} sweeping lr {lr_min:g} -> {lr_max:g}"
        for lr in TQDM(lrs, total=num_it, desc=desc) if RANK in {-1, 0} else lrs:
            for g, ratio in zip(pg, ratios):
                g["lr"] = lr * ratio
            total.zero_()
            for _ in range(self.accumulate):
                try:
                    batch = next(loader)
                except StopIteration:
                    loader = iter(self.train_loader)
                    batch = next(loader)
                loss, loss_items = self.forward_batch(batch)
                total += sum(loss_items.values()).detach()
                self.scaler.scale(loss).backward()
            self.optimizer_step()
            if self.world_size > 1:
                dist.all_reduce(total)
            losses.append(total.item() / (self.accumulate * max(self.world_size, 1)))
            if not math.isfinite(losses[-1]) or losses[-1] > 4 * min(losses):
                break

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        self.scaler.load_state_dict(scaler_state)
        pg = self.optimizer.param_groups
        self.train_loader.reset()

        window = 5
        if len(losses) < 3 * window:
            LOGGER.warning(f"{colorstr('LR finder:')} sweep too short to fit, using the 'lr0' equation")
            return
        y = np.convolve(losses, np.ones(window) / window, mode="valid")
        x = np.log10(lrs[window // 2 : window // 2 + len(y)])
        edge = int(y.argmin())
        if not 5 <= edge < len(y) - 1:
            LOGGER.warning(f"{colorstr('LR finder:')} sweep did not bracket an optimum, using the 'lr0' equation")
            return
        velocity = -np.gradient(y[: edge + 1], x[: edge + 1])
        peak, radius = int(velocity.argmax()), max(edge // 8, 3)
        if not 0 < peak < edge:
            LOGGER.warning(f"{colorstr('LR finder:')} sweep found no velocity peak, using the 'lr0' equation")
            return
        region = slice(max(peak - radius, 0), min(peak + radius + 1, edge + 1))
        a, b, _ = np.polyfit(x[region], velocity[region], 2)
        fastest = np.clip(-b / (2 * a) if a < 0 else x[peak], x[0], x[edge])
        lr = float(f"{min(10 ** ((fastest + x[edge]) / 2), 0.01):.3g}")
        lrs = lrs[: len(losses)]

        smooth = np.full(len(losses), np.nan)
        descent = np.full(len(losses), np.nan)
        offset = window // 2
        smooth[offset : offset + len(y)] = y
        descent[offset : offset + len(velocity)] = velocity
        potential = (np.log10(lrs) >= fastest) & (np.log10(lrs) <= x[edge])
        with open(self.save_dir / "lr_finder.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("step", "lr", "loss", "smoothed_loss", "descent_velocity", "potential"))
            for i, (step_lr, loss) in enumerate(zip(lrs, losses)):
                writer.writerow(
                    (
                        i + 1,
                        step_lr,
                        loss,
                        "" if np.isnan(smooth[i]) else smooth[i],
                        "" if np.isnan(descent[i]) else descent[i],
                        bool(potential[i]),
                    )
                )
        YAML.save(
            self.save_dir / "lr_finder.yaml",
            {
                "steps": len(losses),
                "range": [float(lrs[0]), float(lrs[-1])],
                "fastest_descent_lr": float(10**fastest),
                "fastest_descent_sample_lr": float(10 ** x[peak]),
                "stability_edge_lr": float(10 ** x[edge]),
                "potential_lrs": [float(v) for v in lrs[potential]],
                "selected_lr": lr,
            },
        )

        self.args.lr0 = self.args.warmup_bias_lr = lr
        if self.args.warmup_epochs == DEFAULT_CFG.warmup_epochs:
            self.args.warmup_epochs = round(float(np.clip(5.0 - 2.5 * (x[edge] - fastest), 1.0, 5.0)), 1)
        for g, ratio in zip(pg, ratios):
            g["lr"] = g["initial_lr"] = lr * ratio
        self._setup_scheduler()
        LOGGER.info(
            f"{colorstr('LR finder:')} fitted 'lr0={lr:g}', 'warmup_bias_lr={lr:g}' and "
            f"'warmup_epochs={self.args.warmup_epochs:g}'"
        )
        LOGGER.info(
            f"{colorstr('LR finder:')} potential range {10**fastest:.3g} -> {10 ** x[edge]:.3g}; "
            f"saved {self.save_dir / 'lr_finder.csv'} and {self.save_dir / 'lr_finder.yaml'}"
        )

    def _build_train_pipeline(self):
        """Build dataloaders, optimizer, and scheduler for current batch size."""
        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        # A batch_sampler owns its batch sizes, and torch then leaves batch_size None, so there is nothing to check
        batch = self.train_loader.batch_size
        final_batch_size = (len(self.train_loader.sampler) % batch or batch) if batch else 0
        if self.args.imgsz < 2 * self.stride and not self.train_loader.drop_last and final_batch_size == 1:
            raise ValueError(
                f"final batch=1 training at imgsz={self.args.imgsz} gives BatchNorm a single value per channel; "
                f"change batch or use imgsz >= {2 * self.stride}"
            )
        # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
        self.test_loader = self.get_dataloader(
            self.data.get("val") or self.data.get("test"),
            batch_size=batch_size if self.args.task in {"obb", "semantic", "depth"} else batch_size * 2,
            rank=LOCAL_RANK,
            mode="val",
        )
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        self._setup_scheduler()

    def _setup_train(self):
        """Configure model, optimizer, dataloaders, and training utilities before the training loop."""
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        # channels_last (NHWC) is CUDA-only: lossless and Tensor-Core friendly there, but numerically wrong
        # on MPS and no benefit on CPU
        channels_last = self.args.channels_last is True or (self.args.channels_last is None and TORCH_1_11)
        if channels_last and self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        elif self.args.channels_last:
            LOGGER.warning(f"'channels_last=True' is only supported on CUDA, ignoring on '{self.device.type}'.")
        self.set_model_attributes()

        # Compile model (knowledge distillation runs the wrapped model eagerly and relies on
        # find_unused_parameters under DDP for the frozen teacher, so disable compilation when distilling)
        if self.args.distill_model is not None and self.args.compile:
            LOGGER.warning("'compile' is not supported with knowledge distillation and will be disabled.")
            self.args.compile = False
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        if isinstance(unwrap_model(self.model), DistillationModel):
            freeze_layer_names.append("teacher_model.")
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True
        if not any(v.requires_grad for v in self.model.parameters()):
            raise RuntimeError(
                f"'freeze={self.args.freeze}' froze the entire model with no trainable parameters left. "
                f"Reduce 'freeze' or pass a list of specific layer indices."
            )

        # Check AMP
        self.amp = self.args.amp not in {False, "fp32"}
        self.amp = torch.tensor(self.amp).to(self.device)
        if self.amp and self.args.amp != "bf16" and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and self.world_size > 1:  # DDP
            self.amp = self.amp.int()  # gloo errors with boolean
            dist.broadcast(self.amp, src=0)  # broadcast from rank 0 to all other ranks
        self.amp = bool(self.amp)  # as boolean
        if self.device.type == "npu":
            import torch_npu

            self.scaler = torch_npu.npu.amp.GradScaler(enabled=self.amp and self.args.amp != "bf16")
        else:
            self.scaler = (
                torch.amp.GradScaler(
                    self.device.type if self.device.type == "xpu" else "cuda",
                    enabled=self.amp and self.args.amp != "bf16",
                )
                if TORCH_2_4
                else torch.cuda.amp.GradScaler(enabled=self.amp and self.args.amp != "bf16")
            )
        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # resume training would directly load DistillationModel so check here
        if self.args.distill_model is not None and not isinstance(unwrap_model(self.model), DistillationModel):
            self.model = DistillationModel(student_model=self.model, teacher_model=self.args.distill_model)
        if self.world_size > 1:
            # static_graph=True permits params used >1 time per forward (e.g. flow_model in
            # o2m+o2o pose loss branches) under torch.compile.
            ddp_kwargs = {"static_graph": bool(self.args.compile)} if TORCH_1_11 else {}
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device.index],
                broadcast_buffers=False,
                find_unused_parameters=not bool(self.args.compile),
                **ddp_kwargs,
            )

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()
        auto_lr = str(self.args.lr0).lower() == "auto"
        if self.args.lr_find_only and not auto_lr:
            raise ValueError("'lr_find_only=True' requires 'lr0=auto'")
        self._build_train_pipeline()
        self.set_class_weights()  # before the LR finder builds a loss criterion that snapshots the weights
        if auto_lr and not self.resume:
            self._find_lr()
        if self.args.lr_find_only:
            return
        self.validator = self.get_validator()
        self.ema = ModelEMA(self.model)
        if RANK in {-1, 0}:
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            if self.args.plots:
                self.plot_training_labels()

        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self):
        """Perform the full training loop including setup, epoch iteration, validation, and final evaluation."""
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        if self.args.lr_find_only:
            self._teardown()
            return

        nb = len(self.train_loader)  # number of batches
        nw = self._get_warmup_iterations(nb)
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Using {len(self.train_loader.dataset)} train, {len(self.test_loader.dataset)} val images for "
            f"fraction={self.args.fraction} at imgsz={self.args.imgsz}\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        self._oom_retries = 0  # OOM auto-reduce counter for first epoch
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                if self.loss_names:
                    LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni < nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for x in self.optimizer.param_groups:
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = float(
                            np.interp(
                                ni,
                                xi,
                                [
                                    self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                    x["initial_lr"] * self.lf(epoch),
                                ],
                            )
                        )
                        if "momentum" in x:
                            x["momentum"] = float(np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum]))

                # Forward
                try:
                    self.loss, self.loss_items = self.forward_batch(batch)
                    if not self.loss_names:  # derive loss names from the criterion's loss dict on first batch
                        self.loss_names = tuple(self.loss_items)
                        if RANK in {-1, 0}:
                            LOGGER.info(self.progress_string())
                            self.metrics.update(dict.fromkeys(self.label_loss_items(prefix="val"), 0.0))
                    self.tloss = (
                        self.loss_items
                        if self.tloss is None
                        else {k: (self.tloss[k] * i + v) / (i + 1) for k, v in self.loss_items.items()}
                    )

                    # Backward
                    self.scaler.scale(self.loss).backward()
                except RuntimeError as e:
                    is_oom = "out of memory" in str(e).lower()  # torch.cuda.OutOfMemoryError requires torch>=1.13
                    if not is_oom and not any(
                        s in str(e)
                        for s in (
                            "CUBLAS_STATUS_ALLOC_FAILED",
                            "CUDNN_STATUS_INTERNAL_ERROR",
                            "unable to find an engine",
                        )
                    ):
                        raise
                    if epoch > self.start_epoch or self._oom_retries >= 3 or RANK != -1:
                        raise  # only auto-reduce during first epoch on single GPU, max 3 retries
                    self._oom_retries += 1
                    old_batch = self.batch_size
                    self.args.batch = self.batch_size = max(self.batch_size // 2, 1)
                    error = f"{self.device.type.upper()} out of memory" if is_oom else "CUDA backend memory error"
                    LOGGER.warning(
                        f"{error} with batch={old_batch}. "
                        f"Reducing to batch={self.batch_size} and retrying ({self._oom_retries}/3)."
                    )
                    batch = None
                    self.loss = self.loss_items = self.tloss = None
                    self._clear_memory()
                    self._build_train_pipeline()  # rebuild dataloaders, optimizer, scheduler
                    self.scheduler.last_epoch = self.start_epoch - 1
                    nb = len(self.train_loader)
                    nw = self._get_warmup_iterations(nb)
                    last_opt_step = -1
                    self.optimizer.zero_grad()
                    break  # restart epoch loop with reduced batch size
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = len(self.tloss)
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *self.tloss.values(),  # losses
                            (batch.get("cls", batch["img"]) if isinstance(batch, dict) else batch).shape[0],
                            (batch["img"] if isinstance(batch, dict) else batch).shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")
                if self.stop:
                    break  # allow external stop (e.g. platform cancellation) between batches
            else:
                # for/else: this block runs only when the for loop completes without break (no OOM retry)
                self._oom_retries = 0  # reset OOM counter after successful first epoch

            if self._oom_retries and not self.stop:
                continue  # OOM recovery broke the for loop, restart with reduced batch size

            if hasattr(unwrap_model(self.model).criterion, "update"):
                unwrap_model(self.model).criterion.update()

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # Validation
            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(None if self.device.type == "mps" else 0.5)  # prevent VRAM spike
                self.metrics, self.fitness = self.validate()

            # NaN recovery
            if self._handle_nan_recovery(epoch):
                continue

            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if (self.args.save or final_epoch) and self.save_model():
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                nw = self._get_warmup_iterations(nb)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            # clear if memory utilization > 50%; always clear on MPS due to leak https://github.com/ultralytics/ultralytics/issues/22621
            self._clear_memory(None if self.device.type == "mps" else 0.5)

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        # Do final val with best.pt
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._teardown()

    def _teardown(self):
        """Release training resources."""
        self._clear_memory()
        for loader in (self.train_loader, self.test_loader):
            if hasattr(loader, "close"):
                loader.close()  # shut down persistent dataloader workers so none survive to interpreter exit

    def auto_batch(self, max_num_obj=0, dataset_size=0):
        """Calculate optimal batch size based on model and device memory constraints."""
        # Stride-aligned to match the true multi-scale max size; pyramid heads require stride-multiple inputs
        max_imgsz = math.ceil(self.args.imgsz * (1 + self.args.multi_scale) / self.stride) * self.stride
        return check_train_batch_size(
            model=self.model,
            imgsz=max_imgsz,
            amp=torch.bfloat16 if self.args.amp == "bf16" else self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
            dataset_size=dataset_size,
        )  # returns batch size

    def _get_memory(self, fraction=False):
        """Get accelerator memory utilization in GB or as a fraction of total memory."""
        memory, total = 0, 0
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
            if fraction:
                return __import__("psutil").virtual_memory().percent / 100
        elif self.device.type != "cpu":
            memory = self.accelerator.memory_reserved()
            if fraction:
                total = self.accelerator.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)

    def _clear_memory(self, threshold: float | None = None):
        """Clear accelerator memory by calling garbage collector and emptying cache."""
        if threshold:
            assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
            if self._get_memory(fraction=True) <= threshold:
                return
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            self.accelerator.empty_cache()

    def read_results_csv(self):
        """Read results.csv into a dictionary using polars."""
        import polars as pl  # scope for faster 'import ultralytics'

        try:
            return pl.read_csv(self.csv, infer_schema_length=None).to_dict(as_series=False)
        except Exception:
            return {}

    def _model_train(self):
        """Set model in training mode."""
        self.model.train()
        # Freeze BN stat
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        # A transient NaN/Inf permanently poisons the EMA running average (ema = decay*ema + (1-decay)*model), so
        # save_model would otherwise skip every epoch and the run would finish with no checkpoint on valid input.
        # Resync each poisoned EMA tensor from the live model where finite; any tensor that is non-finite in both is
        # left for the nan_to_num_ pass below, so a usable checkpoint is always written.
        ema = unwrap_model(self.ema.ema)
        if not all(torch.isfinite(v).all() for v in ema.state_dict().values() if isinstance(v, torch.Tensor)):
            model_sd = unwrap_model(self.model).state_dict()
            for k, v in ema.state_dict().items():
                if isinstance(v, torch.Tensor) and not torch.isfinite(v).all() and torch.isfinite(model_sd[k]).all():
                    v.copy_(model_sd[k])
        # Serialize NCHW regardless of channels_last training: released versions fuse with .view(), which crashes on
        # NHWC-strided checkpoint weights, and trainer/predictor re-apply channels_last at setup anyway.
        ema = deepcopy(ema).half().to(memory_format=torch.contiguous_format)
        if hasattr(ema, "criterion"):
            ema.criterion = None  # strip training-only state from the serialization snapshot
        # Clamp fp16 serialization overflow without mutating the live EMA.
        for v in ema.state_dict().values():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                torch.nan_to_num_(v)

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": ema,
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, "fitness": self.fitness},
                "train_results": self.read_results_csv(),
                "date": datetime.now().astimezone().isoformat(),
                "version": __version__,
                "git": {
                    "root": str(GIT.root),
                    "branch": GIT.branch,
                    "commit": GIT.commit,
                    "message": GIT.message,
                    "origin": GIT.origin,
                },
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.wdir.mkdir(parents=True, exist_ok=True)  # ensure weights directory exists
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'
        return True

    def get_dataset(self):
        """Get train and validation datasets from data dictionary.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.
        """
        try:
            self.args.data = convert_ndjson_to_yolo_if_needed(self.args.data, self.args.fraction)

            # Task-specific dataset checking
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
                "semantic",
                "depth",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data

    def setup_model(self):
        """Load, create, or download model for any task.

        Returns:
            (dict | None): Checkpoint to resume training from, or None if no checkpoint is loaded.
        """
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = load_checkpoint(self.model)
            cfg = weights.yaml
        if isinstance(self.args.pretrained, (str, Path)) and not self.resume:
            weights, _ = load_checkpoint(self.args.pretrained)
        elif self.args.pretrained is False and not self.resume:
            weights = None

        # rebuild DistillationModel from resuming checkpoint
        if isinstance(weights, DistillationModel):
            if RANK in {-1, 0}:
                LOGGER.info("Resuming training DistillationModel from checkpoint weights")
            student_model = self.get_model(cfg=cfg, weights=weights.student_model, verbose=RANK in {-1, 0})
            student_model.args = self.args
            # teacher is stripped from the checkpoint to save memory/disk; rebuild it from the distill_model path
            teacher_model = weights.teacher_model if weights.teacher_model is not None else self.args.distill_model
            model = DistillationModel(student_model=student_model, teacher_model=teacher_model)
            if getattr(weights, "projector", None) is not None:
                model.projector.load_state_dict(weights.projector.state_dict())  # restore the trained projector
            model.criterion = None
            self.model = model
        else:
            self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK in {-1, 0})  # calls Model(cfg, weights)
        return ckpt

    def forward_batch(self, batch):
        """Run one training forward pass.

        Args:
            batch (dict): Batch to preprocess and run through the model.

        Returns:
            loss (torch.Tensor): Summed loss to backpropagate, scaled by world size under DDP.
            loss_items (dict): Detached per-component losses independent of batch size.
        """
        with autocast(torch.bfloat16 if self.args.amp == "bf16" else self.amp, device=self.device.type):
            batch = self.preprocess_batch(batch)
            if self.args.compile:
                preds = self.model(batch["img"])
                loss, loss_items = unwrap_model(self.model).loss(batch, preds)
            else:
                loss, loss_items = self.model(batch)
            return loss.sum() * (self.world_size if RANK != -1 else 1), loss_items

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        if self.device.type == "npu" and TORCH_2_0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.grad_clip, foreach=False)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Allow custom preprocessing of model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """Run validation on val set using self.validator.

        Returns:
            (tuple): A tuple containing:
                - metrics (dict | None): Dictionary of validation metrics, or None if validation was skipped.
                - fitness (float | None): Fitness score for the validation, or None if validation was skipped.
        """
        if self.ema and self.world_size > 1:
            # Sync EMA buffers from rank 0 to all ranks
            for buffer in self.ema.ema.buffers():
                dist.broadcast(buffer, src=0)
        metrics = self.validator(self)
        if metrics is None:
            return None, None
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if self.best_fitness is None or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Raise NotImplementedError (must be implemented by subclasses)."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Raise NotImplementedError (must return a `torch.utils.data.DataLoader` in subclasses)."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Return a loss dict with labeled training loss items, or a list of loss names if loss_items is None."""
        if loss_items is None:
            return [f"{prefix}/{x}" for x in self.loss_names]
        return {f"{prefix}/{k}": round(float(v), 5) for k, v in loss_items.items()}

    def set_model_attributes(self):
        """Set or update model parameters before training."""
        self.model.names = self.data["names"]

    def set_class_weights(self):
        """Compute and set class weights for handling class imbalance. Override in subclasses."""

    def build_targets(self, preds, targets):
        """Build target tensors for training YOLO model."""

    def progress_string(self):
        """Return a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plot training samples during YOLO training."""

    def plot_training_labels(self):
        """Plot training labels for YOLO model."""

    def save_metrics(self, metrics):
        """Save training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # number of cols
        t = time.time() - self.train_time_start
        self.csv.parent.mkdir(parents=True, exist_ok=True)  # ensure parent directory exists
        s = "" if self.csv.exists() else ("%s," * n % ("epoch", "time", *keys)).rstrip(",") + "\n"
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(s + ("%.6g," * n % (self.epoch + 1, t, *vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plot metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def on_plot(self, name, data=None):
        """Register plots (e.g. to be consumed in callbacks)."""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Perform final evaluation and validation for the YOLO model."""
        model = self.best if self.best.exists() else None
        with torch_distributed_zero_first(LOCAL_RANK):  # strip only on GPU 0; other GPUs should wait
            if RANK in {-1, 0}:
                ckpt = strip_optimizer(self.last) if self.last.exists() else {}
                if model:
                    # update best.pt train_metrics from last.pt
                    strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
        if model:
            LOGGER.info(f"\nValidating {model}...")
            self.validator.args.plots = self.args.plots
            self.validator.args.compile = False  # disable final val compile as too slow
            self.metrics = self.validator(model=model)
            self.metrics.pop("fitness", None)
            self.epoch += 1  # log best metrics at step epochs+1, not overwriting last epoch
            self.run_callbacks("on_fit_epoch_end")
            self.epoch -= 1  # restore epoch

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())
                ckpt_args = load_checkpoint(last)[0].args
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # reinstate model
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                    "augmentations",
                    "save_period",
                    "workers",
                    "cache",
                    "patience",
                    "time",
                    "freeze",
                    "val",
                    "plots",
                    "channels_last",
                    "distill_model",
                    "save_dir",
                ):  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def _load_checkpoint_state(self, ckpt):
        """Load optimizer, scaler, EMA, and best_fitness from checkpoint."""
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema and ckpt.get("ema"):
            self.ema = ModelEMA(self.model)  # validation with EMA creates inference tensors that can't be updated
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            self.ema.updates = ckpt["updates"]
        self.best_fitness = ckpt.get("best_fitness")

    def _handle_nan_recovery(self, epoch):
        """Detect and recover from NaN/Inf loss by loading last checkpoint."""
        loss_nan = self.loss is not None and not self.loss.isfinite()
        fitness_nan = self.fitness is not None and not np.isfinite(self.fitness)
        corrupted = RANK in {-1, 0} and (loss_nan or fitness_nan)
        reason = "Loss NaN/Inf" if loss_nan else "Fitness NaN/Inf"
        if RANK != -1:  # DDP: broadcast to all ranks
            broadcast_list = [corrupted if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            corrupted = broadcast_list[0]
        if not corrupted:
            return False
        if epoch == self.start_epoch:
            LOGGER.warning(f"{reason} detected but can not recover from last.pt...")
            return False  # Cannot recover on first epoch, let training continue
        if not self.last.exists():
            raise RuntimeError(f"{reason} detected but no valid last.pt is available for recovery")
        self.nan_recovery_attempts += 1
        if self.nan_recovery_attempts > 3:
            raise RuntimeError(f"Training failed: NaN persisted for {self.nan_recovery_attempts} epochs")
        LOGGER.warning(f"{reason} detected (attempt {self.nan_recovery_attempts}/3), recovering from last.pt...")
        self._model_train()  # set model to train mode before loading checkpoint to avoid inference tensor errors
        _, ckpt = load_checkpoint(self.last)
        ema = ckpt["ema"].float()
        ema_state = ema.state_dict()
        if not all(torch.isfinite(v).all() for v in ema_state.values() if isinstance(v, torch.Tensor)):
            raise RuntimeError(f"Checkpoint {self.last} is corrupted with NaN/Inf weights")
        model = unwrap_model(self.model)
        if hasattr(model, "student_model"):
            # Distillation: the EMA is stripped of the teacher (rebuilt from the distill_model path), so only the
            # student and projector are restored; loading them separately keeps a strict key match.
            model.student_model.load_state_dict(ema.student_model.state_dict())
            model.projector.load_state_dict(ema.projector.state_dict())
        else:
            model.load_state_dict(ema_state)  # Load EMA weights into model
        self._load_checkpoint_state(ckpt)  # Load optimizer/scaler/EMA/best_fitness
        del ckpt, ema, ema_state
        self.scheduler.last_epoch = epoch - 1
        return True

    def resume_training(self, ckpt):
        """Resume YOLO training from a given checkpoint."""
        if ckpt is None or not self.resume:
            return
        start_epoch = ckpt.get("epoch", -1) + 1
        assert 0 < start_epoch < self.epochs, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        self._load_checkpoint_state(ckpt)
        model = unwrap_model(self.model)
        if getattr(model, "end2end", False):
            # initialize loss and resume o2o and o2m args
            model.criterion = model.init_criterion()
            model.criterion.updates = start_epoch - 1
            model.criterion.update()
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()
            self.train_loader.reset()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Construct an optimizer for the given model.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected based on the
                number of iterations.
            lr (float, optional): The learning rate for the optimizer.
            momentum (float, optional): The momentum factor for the optimizer.
            decay (float, optional): The weight decay for the optimizer.
            iterations (float, optional): The number of iterations, which determines the optimizer if name is 'auto'.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [{}, {}, {}, {}]  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSprop", "SGD", "MuSGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(str(name).lower(), str(name))
        lr_fit = round(0.002 * 5 / (4 + self.data.get("nc", 10)), 6)
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam
        elif str(lr).lower() == "auto":
            self.args.lr0 = lr = lr_fit

        use_muon = name == "MuSGD"
        ratio = self.args.backbone_lr_ratio  # backbone LR = lr * ratio (1.0 = uniform)
        for module_name, module in unwrap_model(model).named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if param.ndim in {2, 4} and use_muon:  # muon only orthogonalizes matrices and conv filters
                    g[3][fullname] = param  # muon params
                elif "bias" in fullname:  # bias (no decay)
                    g[2][fullname] = param
                elif isinstance(module, bn) or "logit_scale" in fullname or fullname.endswith(".freqs"):
                    # weight (no decay). ContrastiveHead and BNContrastiveHead included here with 'logit_scale',
                    # MixedRoPE2D.freqs because decaying rotary frequencies to zero erases positional variation
                    g[1][fullname] = param
                else:  # weight (with decay)
                    g[0][fullname] = param
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optim_args = {"lr": lr, "betas": (momentum, 0.999), "weight_decay": 0.0}
        elif name == "RMSprop":
            optim_args = {"lr": lr, "momentum": momentum}
        elif name == "SGD" or name == "MuSGD":
            optim_args = {"lr": lr, "momentum": momentum, "nesterov": True}
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for additional optimizers at https://github.com/ultralytics/ultralytics."
            )

        num_params = [len(g[0]), len(g[1]), len(g[2])]  # number of param groups
        g[2] = {"params": g[2], **optim_args, "param_group": "bias"}
        g[0] = {"params": g[0], **optim_args, "weight_decay": decay, "param_group": "weight"}
        g[1] = {"params": g[1], **optim_args, "weight_decay": 0.0, "param_group": "bn"}
        groups = [g[0], g[1], g[2]]
        if use_muon:
            num_params[0] = len(g[3])  # update number of params
            g[3] = {"params": g[3], **optim_args, "weight_decay": decay, "use_muon": True, "param_group": "muon"}
            groups.append(g[3])

        # higher lr for certain parameters in MuSGD when finetuning
        # Split each group into cls-head boost (MuSGD only, lr * 3), backbone (lr * ratio to preserve distilled
        # features), and base. Boost is MuSGD-only so an AdamW backbone_lr_ratio arm gets no head boost. Empty
        # subgroups are dropped, so a plain ratio 1.0 run keeps the flat weight/bn/bias grouping unchanged.
        boosted, backbone = set(), set()
        if use_muon or ratio != 1.0:
            # student_model first so a DistillationModel resolves the trainable head, not the frozen teacher's.
            target = unwrap_model(model)
            target = getattr(target, "student_model", target)
            if use_muon:
                # cls_head cv3 of any Detect subclass, by module identity so it holds at any backbone depth.
                head = target.model[-1] if isinstance(target.model[-1], Detect) else None
                if head is not None:
                    boosted = {id(p) for m in head._classification_heads() for p in m.parameters()}
            if ratio != 1.0:  # backbone by yaml layer count, identity-based so depth/wrappers do not matter
                backbone = {id(p) for m in target.model[: len(target.yaml["backbone"])] for p in m.parameters()}
        g_ = []  # split param groups
        for x in groups:
            p = x.pop("params")
            p_boost, p_bb, p_base = [], [], []
            for k, v in p.items():
                # proto.semseg/SemanticSegment is the YOLO26 semantic aux head, boosted like cls_head (MuSGD only).
                boost = use_muon and (id(v) in boosted or "proto.semseg" in k or "SemanticSegment" in k)
                (p_boost if boost else p_bb if id(v) in backbone else p_base).append(v)
            if p_boost:
                g_.append({"params": p_boost, **x, "lr": lr * 3})
            if p_bb:
                g_.append({"params": p_bb, **x, "lr": lr * ratio})
            g_.append({"params": p_base, **x})
        g = g_
        optimizer = (partial(MuSGD, muon=self.args.muon, sgd=self.args.sgd) if use_muon else getattr(optim, name))(
            params=g
        )

        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{num_params[1]} weight(decay=0.0), {num_params[0]} weight(decay={decay}), {num_params[2]} bias(decay=0.0)"
        )
        return optimizer


class MultiTrainer:
    """Fine-tune a single base model across a collection of datasets and aggregate per-dataset results.

    Used automatically by Model.train() when `data` is a list or tuple, allowing one base model to be benchmarked across
    many datasets (such as the RF100 collection) in a single call. The datasets are fine-tuned in series and the same
    base weights seed each run, so every run starts from an identical model. All output is grouped under one sweep
    directory (e.g. runs/detect/multitrain): each dataset gets its own run subdirectory, and the per-dataset and mean
    metrics are written to multitrain_results.json (for post-processing) alongside a multitrain_results.png bar
    chart. The base model object is left unchanged; each dataset's fine-tuned weights live in its own run directory.

    Attributes:
        trainer (type[BaseTrainer] | None): Task trainer class for Python runs, or None for CLI subprocess runs.
        args (dict): Training arguments shared across datasets; its `data` key holds the dataset collection.
        model (torch.nn.Module): Base model whose weights seed each per-dataset fine-tune.
        callbacks (dict | None): Callbacks forwarded to each per-dataset trainer.
        trainers (list[SimpleNamespace]): Completed per-dataset run records.
        metrics (dict): Mapping of each run name (e.g. coco8, coco8-2) to its training-metrics dict from the checkpoint.
        mean_metrics (dict): Mean training metrics across successful datasets.
        save_dir (Path | None): Sweep directory holding the per-dataset runs and the results JSON/plot.

    Examples:
        Fine-tune one base model across several datasets and read back per-run metrics:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
        >>> results = model.train(data=["coco8.yaml", "african-wildlife.yaml"], epochs=10)
        >>> results["coco8"]["fitness"]  # final fitness on the coco8 run
    """

    def __init__(self, trainer, args, model, _callbacks: dict | None = None):
        """Initialize MultiTrainer with a task trainer class, shared training arguments, and the base model.

        Args:
            trainer (type[BaseTrainer] | None): Task trainer class to run once per dataset. None uses CLI subprocesses.
            args (dict): Training arguments; the `data` key holds the list/tuple of datasets to fine-tune on.
            model (torch.nn.Module): Base model whose weights seed each per-dataset fine-tune.
            _callbacks (dict, optional): Callback functions forwarded to each per-dataset trainer.
        """
        self.trainer = trainer
        self.args = args
        self.model = model
        self.callbacks = _callbacks
        self.trainers = []
        self.metrics = {}
        self.mean_metrics = {}
        self.save_dir = None

    def train(self):
        """Fine-tune the base model on each dataset in series and return a {dataset: metrics} mapping."""
        from types import SimpleNamespace

        from ultralytics.utils.patches import torch_load, torch_save

        datasets = self.args["data"]
        # Group every per-dataset run and the summary plot under one sweep directory, e.g. runs/detect/multitrain
        sweep = SimpleNamespace(
            project=self.args.get("project"),
            task=self.args.get("task"),
            mode="train",
            exist_ok=self.args.get("exist_ok", False),
        )
        self.save_dir = get_save_dir(sweep, name="multitrain")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        model_name = Path(str(self.args.get("model") or "multitrain_base")).stem
        base_model = self.save_dir / f"{model_name}.pt" if self.trainer is None else None
        if base_model:
            torch_save(
                {"model": deepcopy(self.model).half(), "train_args": getattr(self.model, "args", {})}, base_model
            )
        try:
            for i, data in enumerate(datasets):
                LOGGER.info(
                    f"\n{colorstr('blue', 'bold', f'MultiTrainer {i + 1}/{len(datasets)}:')} fine-tuning on {data}"
                )
                path = Path(str(data))
                parent = path.parent.name
                name = Path(os.path.abspath(path.parent)).name if path.stem == "data" and parent else path.stem
                run_name = name
                try:
                    overrides = {
                        **self.args,
                        "data": data,
                        "project": str(self.save_dir),  # nest per-dataset runs inside the sweep directory
                        "name": name,
                        "resume": False,
                    }
                    run = SimpleNamespace(
                        project=overrides["project"],
                        name=overrides["name"],
                        task=overrides.get("task"),
                        mode="train",
                        exist_ok=overrides.get("exist_ok", False),
                        save_dir=None,
                    )
                    save_dir = get_save_dir(run)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    run_name = save_dir.name
                    overrides["save_dir"] = str(save_dir)
                    if self.trainer is None:
                        overrides["model"] = str(base_model)
                        overrides["pretrained"] = True
                        subprocess.run(
                            [
                                *_YOLO_CLI_COMMAND,
                                "train",
                                *(f"{k}={v}" for k, v in overrides.items()),
                            ],
                            check=True,
                        )
                    else:
                        trainer = self.trainer(overrides=overrides, _callbacks=self.callbacks)
                        trainer.model = trainer.get_model(weights=self.model, cfg=self.model.yaml)
                        trainer.train()
                    best, last = save_dir / "weights" / "best.pt", save_dir / "weights" / "last.pt"
                    ckpt = best if best.exists() else last
                    metrics = None
                    if self.trainer is not None:
                        metrics = getattr(getattr(trainer, "validator", None), "metrics", None)
                        if metrics is not None:
                            metrics = metrics.results_dict
                    self.metrics[run_name] = metrics or (torch_load(ckpt)["train_metrics"] if ckpt.exists() else None)
                    self.trainers.append(SimpleNamespace(save_dir=save_dir, best=best, last=last))
                except Exception as e:  # one bad dataset should not abort the whole sweep
                    LOGGER.error(f"MultiTrainer: fine-tuning on {data} failed, skipping: {e}")
                    self.metrics[run_name] = None
        finally:
            if base_model:
                base_model.unlink(missing_ok=True)
        if RANK in {-1, 0} and self.trainers:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.save_results()  # JSON of per-dataset + mean metrics for programmatic post-processing
            if self.args.get("plots", True):
                self.plot_results()
        return self.metrics

    def save_results(self):
        """Write per-dataset and mean metrics to multitrain_results.json for programmatic post-processing."""
        import json

        results = {run: ({k: float(v) for k, v in m.items()} if m else None) for run, m in self.metrics.items()}
        valid = [m for m in results.values() if m]
        keys = {k for m in valid for k in m}
        self.mean_metrics = {k: sum(m[k] for m in valid if k in m) / sum(k in m for m in valid) for k in keys}
        file = self.save_dir / "multitrain_results.json"
        with open(file, "w", encoding="utf-8") as f:
            json.dump({"results": results, "mean": self.mean_metrics}, f, indent=2)
        LOGGER.info(f"MultiTrainer results saved to {colorstr('bold', file)}")
        return file

    def plot_results(self):
        """Save a cross-dataset bar chart of the per-dataset metric with the mean across all datasets."""
        from ultralytics.cfg import TASK2METRIC
        from ultralytics.utils.plotting import plot_multitrain_results

        key = TASK2METRIC.get(self.args.get("task"))
        scores = {run: float(m.get(key, m.get("fitness", 0.0))) for run, m in self.metrics.items() if m}
        if not scores:
            return None
        fname = plot_multitrain_results(scores, key=key or "fitness", save_dir=self.save_dir)
        LOGGER.info(f"MultiTrainer results saved to {colorstr('bold', fname)}")
        return fname
