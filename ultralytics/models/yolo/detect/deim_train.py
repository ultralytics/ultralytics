# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""DetectionTrainer with DEIM-style augmentation probability decay and optional flat-cosine LR.

Lifts the augmentation-decay slice of the DEIM training recipe: per-epoch linear decay of
mosaic/mixup/copy-paste toward `aug_decay_min_prob`, then a no-aug tail that zeroes out every v8
augmentation (color, affine, mosaic, mixup, copy_paste, cutmix). Also exposes the DEIM flat-cosine
LR schedule via `lr_scheduler=flatcosine`. Loss, validator, predictor stay standard YOLO -- the
Hungarian matcher and the RT-DETR transform stack do NOT come along.

Activate by setting `deim_aug_scheduler=decay` in the training args; every other DEIM decay knob
(`flat_epoch`, `no_aug_epoch`, `aug_decay_min_prob`) already lives in the default config.
"""

import math
from copy import copy

from torch import optim

from ultralytics.data.dataset import YOLODataset
from ultralytics.models.rtdetr.detr_augment import (
    compute_deim_scheduled_prob,
    compute_policy_epochs,
    resolve_deim_aug_scheduler,
)
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import one_cycle, unwrap_model


class YOLODEIMDataset(YOLODataset):
    """YOLODataset with per-epoch DEIM-style augmentation probability decay.

    Attributes:
        base_hyp (SimpleNamespace): Frozen copy of the initial training hyps.
        deim_aug_scheduler (str): Resolved scheduler mode, 'legacy' or 'decay'.
        policy_epochs (tuple): (start, mid, stop) from compute_policy_epochs; stop is the no-aug boundary.
        mosaic_prob (float): Base mosaic probability from the initial hyp.
        mixup_prob (float): Base mixup probability from the initial hyp.
        copyblend_prob (float): Base copy_paste probability from the initial hyp.
        decay_min_prob (float): Probability floor during the decay phase.
    """

    _NO_AUG_FIELDS = (
        "mosaic",
        "mixup",
        "copy_paste",
        "cutmix",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "hsv_h",
        "hsv_s",
        "hsv_v",
    )

    def __init__(self, *args, data=None, **kwargs):
        """Cache DEIM schedule, then build the dataset."""
        hyp = kwargs["hyp"]
        self.base_hyp = copy(hyp)
        self.deim_aug_scheduler = resolve_deim_aug_scheduler(hyp)
        self.policy_epochs = compute_policy_epochs(hyp)
        self.mosaic_prob = float(hyp.mosaic)
        self.mixup_prob = float(hyp.mixup)
        self.copyblend_prob = float(hyp.copy_paste)
        self.decay_min_prob = float(hyp.aug_decay_min_prob)
        super().__init__(*args, data=data, **kwargs)
        if self.augment and self.deim_aug_scheduler == "decay":
            self.set_epoch(0)

    def _epoch_hyp(self, epoch: int):
        """Clone base hyp and apply DEIM decay for the given epoch."""
        hyp = copy(self.base_hyp)
        stop = self.policy_epochs[2]
        hyp.mosaic = compute_deim_scheduled_prob(self.mosaic_prob, epoch, stop, self.decay_min_prob)
        hyp.mixup = compute_deim_scheduled_prob(self.mixup_prob, epoch, stop, self.decay_min_prob)
        hyp.copy_paste = compute_deim_scheduled_prob(self.copyblend_prob, epoch, stop, self.decay_min_prob)
        if epoch >= stop:
            for f in self._NO_AUG_FIELDS:
                setattr(hyp, f, 0.0)
            hyp.augmentations = []
        return hyp

    def set_epoch(self, epoch: int) -> None:
        """Rebuild transforms with epoch-decayed hyps; no-op when scheduler is 'legacy'."""
        self.epoch = epoch
        if self.augment and self.deim_aug_scheduler == "decay":
            self.transforms = self.build_transforms(hyp=self._epoch_hyp(epoch))


class DetectionDEIMTrainer(DetectionTrainer):
    """DetectionTrainer that swaps YOLODataset for YOLODEIMDataset and propagates the epoch each cycle."""

    _epoch_callback_registered = False

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build a YOLODEIMDataset; mirrors the fields set by build_yolo_dataset."""
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return YOLODEIMDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or (mode == "val"),
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def _setup_scheduler(self):
        """Support 'linear' / 'cosine' / 'flatcosine' via lr_scheduler; fall back to base scheduler otherwise."""
        scheduler_arg = getattr(self.args, "lr_scheduler", None)
        if scheduler_arg is None:
            return super()._setup_scheduler()
        scheduler_name = str(scheduler_arg).strip().lower()
        if not scheduler_name:
            return super()._setup_scheduler()

        if scheduler_name == "linear":
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf
        elif scheduler_name in {"cosine", "cos", "cos_lr"}:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)
        elif scheduler_name in {"flatcosine", "flat_cosine", "flatcos"}:
            if self.args.flat_epoch is None:
                _, flat_epoch, _ = compute_policy_epochs(self.args)
            else:
                flat_epoch = int(self.args.flat_epoch)
            if not (0 <= flat_epoch <= self.epochs):
                raise ValueError(
                    f"flatcosine got invalid flat_epoch={flat_epoch} for epochs={self.epochs}. "
                    "Expected 0 <= flat_epoch <= epochs."
                )
            gamma = float(self.args.lrf)
            if not (0.0 <= gamma <= 1.0):
                raise ValueError(f"flatcosine got invalid lrf={gamma}. Expected 0.0 <= lrf <= 1.0.")
            decay_epochs = max(self.epochs - flat_epoch, 1)

            def _flat_cosine(epoch: int) -> float:
                if epoch < flat_epoch:
                    return 1.0
                progress = min(max((epoch - flat_epoch) / decay_epochs, 0.0), 1.0)
                return gamma + 0.5 * (1.0 - gamma) * (1.0 + math.cos(math.pi * progress))

            self.lf = _flat_cosine
        else:
            LOGGER.warning(f"Unknown lr_scheduler={scheduler_name!r}, falling back to default scheduler.")
            return super()._setup_scheduler()

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _on_train_epoch_start(self, trainer=None) -> None:
        """Propagate the current epoch to the dataset so transforms rebuild under decay mode."""
        trainer = trainer or self
        dataset = trainer.train_loader.dataset
        dataset.set_epoch(trainer.epoch)
        trainer.train_loader.reset()

    def train(self, *args, **kwargs):
        """Register the epoch callback once and disable base close_mosaic when decay is active."""
        if self.args.close_mosaic and resolve_deim_aug_scheduler(self.args) == "decay":
            self.args.close_mosaic = 0
        if not self._epoch_callback_registered:
            self.add_callback("on_train_epoch_start", self._on_train_epoch_start)
            self._epoch_callback_registered = True
        return super().train(*args, **kwargs)
