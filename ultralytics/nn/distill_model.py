from ultralytics.utils.torch_utils import smart_inference_mode
import torch.nn.functional as F


class DistillationModel:
    def __init__(self, teacher_model, student_model, feats_idx, temperature=1.0, alpha=0.5):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        assert isinstance(feats_idx, int), "Currently only single feature index is supported."
        self.feats_idx = feats_idx

    # def distillation_loss(self, student_logits, teacher_logits, true_labels):
    #     # Soft targets from teacher
    #     soft_targets = F.log_softmax(teacher_logits / self.temperature, dim=1)
    #     student_soft_logits = F.log_softmax(student_logits / self.temperature, dim=1)
    #
    #     # Distillation loss (Kullback-Leibler divergence)
    #     distillation_loss = F.kl_div(student_soft_logits, soft_targets, reduction="batchmean") * (self.temperature**2)
    #
    #     # Hard target loss (Cross-entropy)
    #     hard_target_loss = F.cross_entropy(student_logits, true_labels)
    #
    #     # Combined loss
    #     total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_target_loss
    #     return total_loss

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        with smart_inference_mode():
            teacher_feats = self.teacher_model(batch["img"], return_feats=True)[1][self.feats_idx]
        preds, feats = self.student_model(batch["img"], return_feats=True)
        student_feats = feats[self.feats_idx]

        regular_loss = self.student_model.loss(batch, preds)
        # TODO: placeholder for distillation loss
        distill_loss = F.mse_loss(student_feats, teacher_feats.detach())
        return regular_loss + distill_loss
