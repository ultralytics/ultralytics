from ultralytics.utils.torch_utils import copy_attr
from .tasks import load_checkpoint
import torch.nn.functional as F
from torch import nn
import torch


class DistillationModel(nn.Module):
    """Distillation model wrapper.
    Currently only supports feature-based distillation with a single feature index on YOLO models.
    """

    def __init__(self, teacher_model: str | nn.Module, student_model: nn.Module, feats_idx: int):
        """Initialize DistillationModel."""
        super().__init__()
        assert feats_idx == 23
        assert isinstance(feats_idx, int), "Currently only single feature index is supported."
        if isinstance(teacher_model, str):
            teacher_model = load_checkpoint(teacher_model)[0]
        device = next(student_model.parameters()).device
        self.teacher_model = teacher_model.to(device).eval()
        for v in self.teacher_model.parameters():
            v.requires_grad = False
        # get the feature dimensions
        with torch.inference_mode():
            teacher_dim = self.unwrap_output(teacher_model(torch.zeros(1, 3, 256, 256).to(device), embed=[feats_idx], direct_return=True)).shape[1]
            student_dim = self.unwrap_output(student_model(torch.zeros(1, 3, 256, 256).to(device), embed=[feats_idx], direct_return=True)).shape[1]
        self.student_model = student_model
        self.feats_idx = feats_idx
        self.projector = nn.Linear(student_dim, teacher_dim) if student_dim != teacher_dim else nn.Identity()
        copy_attr(self, student_model)
        self.temperature = 2

    def distillation_loss(self, student_logits, teacher_logits):
        # Soft targets from teacher
        soft_targets = F.log_softmax(teacher_logits / self.temperature, dim=1)
        student_soft_logits = F.log_softmax(student_logits / self.temperature, dim=1)

        # Distillation loss (Kullback-Leibler divergence)
        distillation_loss = F.kl_div(student_soft_logits, soft_targets, reduction="batchmean") * (self.temperature**2)
        return distillation_loss

    def forward(self, x, *args, **kwargs):
        """Forward pass through the student model."""
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.student_model.predict(x, *args, **kwargs)

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        with torch.inference_mode():
            teacher_feats = self.unwrap_output(self.teacher_model(batch["img"]))
        preds = self.student_model(batch["img"])
        student_feats = self.unwrap_output(preds)

        student_feats = F.normalize(student_feats.flatten(2).permute(0, 2, 1), p=2, dim=-1)
        teacher_feats = F.normalize(teacher_feats.flatten(2).permute(0, 2, 1), p=2, dim=-1)
        cos_sim = F.cosine_similarity(student_feats, teacher_feats, dim=-1)
        loss_distill = (1 - cos_sim).mean()[None] * self.student_model.args.dis
        # loss_distill = self.distillation_loss(student_feats, teacher_feats)[None] * self.student_model.args.dis

        regular_loss, regular_loss_detach = self.student_model.loss(batch, preds)
        return torch.cat([regular_loss, loss_distill]), torch.cat([regular_loss_detach, loss_distill.detach()])

    def unwrap_output(self, outputs):
        """Unwrap output from various possible structures."""
        if isinstance(outputs, tuple):
            return self.unwrap_output(outputs[1])
        elif isinstance(outputs, dict):
            if "one2one" in outputs:
                return self.unwrap_output(outputs["one2one"])
            elif "scores" in outputs:
                return outputs["scores"]
        return outputs
