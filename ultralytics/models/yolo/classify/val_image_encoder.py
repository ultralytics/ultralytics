# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch

from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER


class _DistillMetrics:
    """Minimal metrics for encoder distillation (loss-only, no accuracy).

    Attributes speed, save_dir, confusion_matrix exist for BaseValidator compatibility.
    """

    keys = ["distill_loss"]

    def __init__(self):
        """Initialize with default loss."""
        self._loss = float("inf")
        self.speed = {}
        self.save_dir = None
        self.confusion_matrix = None

    @property
    def fitness(self):
        """Return negative loss as fitness (lower loss = higher fitness for early stopping)."""
        return -self._loss

    @property
    def results_dict(self):
        """Return metrics dict for logging."""
        return {"metrics/distill_loss": self._loss, "fitness": self.fitness}


class ImageEncoderValidator(BaseValidator):
    """Validator for single or multi-teacher encoder distillation -- loss only, no accuracy.

    Inherits from BaseValidator (not ClassificationValidator) because distillation has no class labels, predictions, or
    accuracy metrics. BaseValidator's default no-ops for update_metrics, gather_stats, plot_val_samples,
    plot_predictions, and postprocess are all correct as-is.

    Attributes:
        teachers (dict): Frozen teacher models keyed by safe name (set by trainer).
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize ImageEncoderValidator.

        Args:
            dataloader: Validation DataLoader.
            save_dir (str | Path, optional): Directory to save results.
            args (dict, optional): Validation configuration.
            _callbacks (dict, optional): Callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.teacher_models = {}
        self.metrics = _DistillMetrics()

    def init_metrics(self, model):
        """Force fp32 validation.

        YOLO backbone produces nan features in raw fp16 on ~12% of val batches (fp16
        overflow in intermediate activations). Force fp32 following UNIC convention
        (unic/main_unic.py:432-465: no autocast during eval, model stays fp32).

        Args:
            model (nn.Module): The model being validated (may be DDP-wrapped).
        """
        self.args.half = False
        model.float()

    def preprocess(self, batch):
        """Preprocess batch: resize for student and teacher, run all teachers.

        Validation runs the student at a fixed ``args.imgsz`` (no multi-scale rotation); when the loader
        serves a larger scale under multi-scale distillation, the teacher is downsampled to its native res.

        Args:
            batch (torch.Tensor): Images at the load resolution (B, 3, H, W).

        Returns:
            (dict): Batch with 'img', 'cls', per-teacher entries, and '_teacher_keys'.
        """
        imgs = batch.to(self.device, non_blocking=True)
        student_imgs = (
            torch.nn.functional.interpolate(imgs, size=self.args.imgsz, mode="bilinear", antialias=True)
            if imgs.shape[-1] != self.args.imgsz
            else imgs
        )
        if self.args.half:
            student_imgs = student_imgs.half()

        teacher_imgsz = getattr(self, "_teacher_imgsz", imgs.shape[-1])
        teacher_imgs = (
            torch.nn.functional.interpolate(imgs, size=teacher_imgsz, mode="bilinear", antialias=True)
            if imgs.shape[-1] != teacher_imgsz
            else imgs
        )

        teacher_keys = list(self.teacher_models.keys())
        result = {
            "img": student_imgs,
            "cls": torch.zeros(imgs.shape[0], dtype=torch.long, device=self.device),
            "_teacher_keys": teacher_keys,
        }

        for sk in teacher_keys:
            out = self.teacher_models[sk].encode(teacher_imgs)
            result[sk] = {"cls": out.cls, "patches": out.patches}

        return result

    def finalize_metrics(self):
        """Set speed and save_dir on metrics (loss is updated in get_stats)."""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir

    def get_stats(self):
        """Return metrics dict, updating distill_loss from accumulated val loss.

        Must update here because base validator calls get_stats() before finalize_metrics().
        """
        if getattr(self, "loss", None) is not None:
            self.metrics._loss = float(self.loss.sum())
        return self.metrics.results_dict

    def get_desc(self):
        """Return header for validation progress bar."""
        return f"{'':>22}{'distill_loss':>11}"

    def print_results(self):
        """Print val distillation loss."""
        LOGGER.info(f"{'all':>22}{self.metrics._loss:>11.4f}")
