# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy
from typing import Optional

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import LOGGER, RANK, colorstr

from .val import RTDETRDataset, RTDETRValidator


class RTDETRTrainer(DetectionTrainer):
    """
    Trainer class for the RT-DETR model developed by Baidu for real-time object detection.

    This class extends the DetectionTrainer class for YOLO to adapt to the specific features and architecture of RT-DETR.
    The model leverages Vision Transformers and has capabilities like IoU-aware query selection and adaptable inference
    speed.

    Attributes:
        loss_names (tuple): Names of the loss components used for training.
        data (dict): Dataset configuration containing class count and other parameters.
        args (dict): Training arguments and hyperparameters.
        save_dir (Path): Directory to save training results.
        test_loader (DataLoader): DataLoader for validation/testing data.

    Methods:
        get_model: Initialize and return an RT-DETR model for object detection tasks.
        build_dataset: Build and return an RT-DETR dataset for training or validation.
        get_validator: Return a DetectionValidator suitable for RT-DETR model validation.

    Notes:
        - F.grid_sample used in RT-DETR does not support the `deterministic=True` argument.
        - AMP training can lead to NaN outputs and may produce errors during bipartite graph matching.

    Examples:
        >>> from ultralytics.models.rtdetr.train import RTDETRTrainer
        >>> args = dict(model="rtdetr-l.yaml", data="coco8.yaml", imgsz=640, epochs=3)
        >>> trainer = RTDETRTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the RTDETRTrainer, registering a callback to disable strong augmentations in later epochs."""
        super().__init__(cfg, overrides, _callbacks)
        self.strong_aug_disabled = False
        self.add_callback("on_train_epoch_start", self._disable_strong_aug_if_needed)

    def _setup_train(self, world_size):
        """Setup training with dynamic augmentation and scale-adaptive hyperparameters for RT-DETR v2."""
        super()._setup_train(world_size)
        self._apply_scale_adaptive_hyperparameters()
        # Set default value for disable_strong_aug_epochs if not provided
        if not hasattr(self.args, "disable_strong_aug_epochs"):
            self.args.disable_strong_aug_epochs = self.cfg.disable_strong_aug_epochs

    # TODO 后续可能迁移至utils/callbacks/rtdetr.py
    def _disable_strong_aug_if_needed(self, trainer):
        """Disables strong augmentations in the final epochs of training for RT-DETR v2."""
        if self.args.disable_strong_aug_epochs > 0 and not self.strong_aug_disabled:
            remaining_epochs = self.epochs - self.epoch
            if remaining_epochs <= self.args.disable_strong_aug_epochs:
                LOGGER.info(
                    f"{colorstr('bright_yellow', 'RT-DETR v2')}: Disabling strong augmentation for the final "
                    f"{self.args.disable_strong_aug_epochs} epochs..."
                )
                if hasattr(self.train_loader.dataset, "disable_strong_aug"):
                    self.train_loader.dataset.disable_strong_aug()
                    if hasattr(self.train_loader, "reset"):
                        self.train_loader.reset()
                self.strong_aug_disabled = True

    def get_model(self, cfg: Optional[dict] = None, weights: Optional[str] = None, verbose: bool = True):
        """
        Initialize and return an RT-DETR model for object detection tasks.

        Args:
            cfg (dict, optional): Model configuration.
            weights (str, optional): Path to pre-trained model weights.
            verbose (bool): Verbose logging if True.

        Returns:
            (RTDETRDetectionModel): Initialized model.
        """
        model = RTDETRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path: str, mode: str = "val", batch: Optional[int] = None):
        """
        Build and return an RT-DETR dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): Dataset mode, either 'train' or 'val'.
            batch (int, optional): Batch size for rectangle training.

        Returns:
            (RTDETRDataset): Dataset object for the specific mode.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_validator(self):
        """Return a DetectionValidator suitable for RT-DETR model validation."""
        self.loss_names = "giou_loss", "cls_loss", "l1_loss"
        args = copy(self.args)
        if hasattr(args, "_lr_scaled"):
            del args._lr_scaled
        if hasattr(args, "_wd_scaled"):
            del args._wd_scaled
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=args)

    def _apply_scale_adaptive_hyperparameters(self):
        """
        Apply scale-adaptive hyperparameters for RT-DETR v2 based on model size.

        Different model sizes require different learning rates for optimal performance:
        - Larger models (more parameters) need smaller learning rates
        - Smaller models can handle larger learning rates
        """
        # Detect model scale from config filename or parameters
        model_scale = self._detect_model_scale()

        # Scale-adaptive learning rate mapping (based on RT-DETR v2 paper)
        scale_lr_mapping = {
            "n": 1.0,  # nano - baseline LR
            "s": 0.8,  # small - slightly reduced
            "m": 0.6,  # medium - more reduced
            "l": 0.4,  # large - significantly reduced
            "x": 0.3,  # extra large - most reduced
        }

        if model_scale in scale_lr_mapping:
            lr_scale = scale_lr_mapping[model_scale]
            original_lr = getattr(self.args, "lr0", 0.01)

            # Apply learning rate scaling
            if not hasattr(self.args, "_lr_scaled"):
                self.args.lr0 = original_lr * lr_scale
                self.args._lr_scaled = True  # Prevent double scaling

                LOGGER.info(
                    f"{colorstr('bright_yellow', 'RT-DETR v2')}: Applied scale-adaptive LR for model '{model_scale}': "
                    f"{original_lr:.4f} -> {self.args.lr0:.4f} (scale: {lr_scale})"
                )

        # Scale-adaptive weight decay (optional)
        scale_wd_mapping = {
            "n": 1.0,
            "s": 1.2,
            "m": 1.5,
            "l": 2.0,
            "x": 2.5,
        }

        if model_scale in scale_wd_mapping and not hasattr(self.args, "_wd_scaled"):
            wd_scale = scale_wd_mapping[model_scale]
            original_wd = getattr(self.args, "weight_decay", 0.0001)
            self.args.weight_decay = original_wd * wd_scale
            self.args._wd_scaled = True

    def _detect_model_scale(self):
        """
        Detect the model scale (n/s/m/l/x) from configuration or model parameters.

        Returns:
            str: Model scale identifier ('n', 's', 'm', 'l', 'x') or 'unknown'
        """
        # Method 1: Check config filename
        if hasattr(self.args, "model") and self.args.model:
            model_file = str(self.args.model).lower()
            for scale in ["n", "s", "m", "l", "x"]:
                if f"rtdetrv2-{scale}" in model_file or f"rtdetr-{scale}-v2" in model_file:
                    return scale
                if f"rtdetr-{scale}" in model_file:  # fallback for v1 configs
                    return scale

        # Method 2: Estimate from model parameters (fallback)
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            if total_params < 10e6:  # < 10M parameters
                return "n"
            elif total_params < 25e6:  # < 25M parameters
                return "s"
            elif total_params < 60e6:  # < 60M parameters
                return "m"
            elif total_params < 120e6:  # < 120M parameters
                return "l"
            else:  # > 120M parameters
                return "x"
        except Exception:
            pass

        return "unknown"
