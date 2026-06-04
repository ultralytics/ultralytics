# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Teacher-student feature distillation for Ultralytics transformer-decoder detectors.

Adapted from the `exp-main-distill-clean` branch (YOLO Detect head) for RT-DETR /
DFINE / DEIM transformer decoders. Forward hooks capture FPN/PAN features at the
indices feeding the decoder; a per-level 1x1-conv projector aligns student channels
to teacher channels, and a score-weighted L2 loss is computed between them. The
per-pixel score is the teacher's own channel-L2 feature magnitude, normalized per
image to [0, 1] — a class-agnostic spatial saliency. This replaces YOLO's
classification-score weighting since the transformer decoder produces only
per-query logits, not a dense per-pixel score map.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.utils.torch_utils import copy_attr

from .tasks import load_checkpoint


class FeatureHook:
    """Picklable forward hook that stores layer output into a shared dict."""

    def __init__(self, feat_dict, idx):
        """Store a reference to the shared dict and the layer index this hook writes to."""
        self.feat_dict = feat_dict
        self.idx = idx

    def __call__(self, module, input, output):
        """Write the layer output into the shared dict keyed by layer index."""
        self.feat_dict[self.idx] = output


class DistillationModel(nn.Module):
    """Teacher-student knowledge distillation for RT-DETR / DFINE / DEIM detectors.

    Wraps a frozen teacher and a trainable student. Forward hooks capture features at the FPN/PAN layers feeding the
    transformer decoder (`RTDETRDecoder.f`). A per-level projector aligns student channels to teacher channels, and the
    loss is a score-weighted L2 between projected student and teacher features, summed over levels and scaled by `dis`.

    Attributes:
        teacher_model (nn.Module): Frozen teacher model providing features.
        student_model (nn.Module): Trainable student model being distilled.
        feats_idx (list[int]): FPN/PAN layer indices feeding the transformer decoder.
        projector (nn.ModuleList): 1x1-conv-ReLU-1x1-conv projector per feature level.
        dis (float): Distillation loss weight factor.
    """

    def __init__(self, teacher_model: str | nn.Module, student_model: nn.Module):
        """Build the distillation wrapper around a teacher checkpoint and an initialized student.

        Args:
            teacher_model (str | nn.Module): Teacher checkpoint path or already-loaded module.
            student_model (nn.Module): Student module to be trained.
        """
        super().__init__()
        if isinstance(teacher_model, str):
            teacher_model = load_checkpoint(teacher_model)[0]
        device = next(student_model.parameters()).device
        self.teacher_model = teacher_model.to(device)
        self._freeze_teacher()
        self.student_model = student_model
        self.feats_idx = self.get_distill_layers(student_model)

        self._teacher_feats = {}
        self._student_feats = {}
        for idx in self.feats_idx:
            self.teacher_model.model[idx].register_forward_hook(FeatureHook(self._teacher_feats, idx))
            self.student_model.model[idx].register_forward_hook(FeatureHook(self._student_feats, idx))

        # Dummy forward to capture feature shapes through the hooks.
        imgsz = student_model.args.imgsz
        with torch.no_grad():
            teacher_model(torch.zeros(2, 3, imgsz, imgsz).to(device))
            student_model(torch.zeros(2, 3, imgsz, imgsz).to(device))
        teacher_output = [self._teacher_feats[idx] for idx in self.feats_idx]
        student_output = [self._student_feats[idx] for idx in self.feats_idx]
        assert all(t.ndim == 4 and s.ndim == 4 for t, s in zip(teacher_output, student_output)), (
            "Expected 4D FPN feature maps at every hook index for transformer-decoder distillation."
        )

        copy_attr(self, student_model)
        self.dis = self.student_model.args.dis

        projectors = []
        for s, t in zip(student_output, teacher_output):
            projectors.append(
                nn.Sequential(
                    nn.Conv2d(s.shape[1], t.shape[1], kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(t.shape[1], t.shape[1], kernel_size=1, stride=1, padding=0),
                )
            )
        self.projector = nn.ModuleList(projectors).to(device)

    def __getstate__(self):
        """Return a clean copy of state for pickling without hooks and extracted features."""
        state = self.__dict__.copy()
        state["_teacher_feats"] = {}
        state["_student_feats"] = {}
        return state

    def __setstate__(self, state):
        """Clear stale features and hooks, then re-register forward hooks after unpickling."""
        self.__dict__.update(state)
        self._teacher_feats = {}
        self._student_feats = {}
        for idx in self.feats_idx:
            self.teacher_model.model[idx]._forward_hooks.clear()
            self.student_model.model[idx]._forward_hooks.clear()
            self.teacher_model.model[idx].register_forward_hook(FeatureHook(self._teacher_feats, idx))
            self.student_model.model[idx].register_forward_hook(FeatureHook(self._student_feats, idx))

    @staticmethod
    def get_distill_layers(model):
        """Return the FPN/PAN layer indices feeding the transformer decoder.

        Raises ValueError if the model does not have a transformer-decoder detection head.
        """
        for m in model.model:
            if isinstance(m, RTDETRDecoder):  # DFineDecoder / DeimDecoder are subclasses
                return list(m.f)
        raise ValueError("No transformer-decoder head (RTDETRDecoder family) found in model")

    def _freeze_teacher(self):
        """Disable gradients and set the teacher to eval mode for distillation."""
        self.teacher_model.eval()
        for v in self.teacher_model.parameters():
            if v.requires_grad:
                v.requires_grad = False

    def train(self, mode: bool = True):
        """Set train mode while keeping the teacher frozen in eval mode."""
        super().train(mode)
        self._freeze_teacher()
        return self

    def forward(self, x, *args, **kwargs):
        """Dispatch to loss when given a batch dict, otherwise run student prediction."""
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.student_model.predict(x, *args, **kwargs)

    @staticmethod
    def _spatial_saliency(feat: torch.Tensor) -> torch.Tensor:
        """Per-pixel teacher feature magnitude as a class-agnostic spatial attention mask.

        Computes the channel-wise L2 norm of `feat` and normalizes to [0, 1] per image.

        Args:
            feat (torch.Tensor): Teacher feature of shape (N, C, H, W).

        Returns:
            (torch.Tensor): Score tensor of shape (N, 1, H*W).
        """
        n, _, h, w = feat.shape
        mag = feat.pow(2).mean(dim=1, keepdim=True).sqrt()
        mag = mag / (mag.amax(dim=(2, 3), keepdim=True) + 1e-9)
        return mag.view(n, 1, h * w)

    def loss(self, batch, preds=None):
        """Compute combined detection loss and feature distillation loss.

        Args:
            batch (dict): Batch dict with the input image and detection targets.
            preds (torch.Tensor | list[torch.Tensor], optional): Cached student predictions.

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Concatenated (regular_loss, distill_loss) tensors, both raw and
                detached for logging.
        """
        loss_distill = torch.zeros(1, device=batch["img"].device)
        if not self.training:  # validation while training: skip distillation, keep loss shape
            preds = self.student_model(batch["img"])
            regular_loss, regular_loss_detach = self.student_model.loss(batch, preds)
            distill_loss_detach = torch.zeros(1, device=batch["img"].device)
            return self._concat_losses(regular_loss, loss_distill, regular_loss_detach, distill_loss_detach)

        self._teacher_feats.clear()
        self._student_feats.clear()

        with torch.no_grad():
            self.teacher_model(batch["img"])
        preds = self.student_model(batch["img"])

        regular_loss, regular_loss_detach = self.student_model.loss(batch, preds)

        for i, feat_idx in enumerate(self.feats_idx):
            teacher_feat = self._teacher_feats[feat_idx]
            student_feat = self.projector[i](self._student_feats[feat_idx])
            teacher_score = self._spatial_saliency(teacher_feat)
            loss_distill += self.loss_sl2(student_feat, teacher_feat, teacher_score) * self.dis

        distill_loss_detach = loss_distill.detach()
        return self._concat_losses(regular_loss, loss_distill, regular_loss_detach, distill_loss_detach)

    @staticmethod
    def _concat_losses(regular_loss, loss_distill, regular_loss_detach, distill_loss_detach):
        """Concatenate regular and distillation losses, promoting 0-D scalars to 1-D first.

        DFINE/DEIM `student_model.loss` returns a 0-D scalar total loss for backward and a
        1-D component breakdown for logging; YOLO returns 1-D for both. Promoting any 0-D
        tensor to 1-D normalizes shapes so `torch.cat` accepts either input convention.
        """
        regular_loss = regular_loss.unsqueeze(0) if regular_loss.ndim == 0 else regular_loss
        regular_loss_detach = regular_loss_detach.unsqueeze(0) if regular_loss_detach.ndim == 0 else regular_loss_detach
        return torch.cat([regular_loss, loss_distill]), torch.cat([regular_loss_detach, distill_loss_detach])

    @staticmethod
    def loss_sl2(student_feat: torch.Tensor, teacher_feat: torch.Tensor, teacher_score: torch.Tensor) -> torch.Tensor:
        """Score-weighted L2 distillation loss for a single feature level.

        Args:
            student_feat (torch.Tensor): Projected student feature of shape (N, C, H, W).
            teacher_feat (torch.Tensor): Teacher feature of shape (N, C, H, W).
            teacher_score (torch.Tensor): Per-pixel score of shape (N, 1, H*W).

        Returns:
            (torch.Tensor): Scalar score-weighted L2 loss.
        """
        n, c = student_feat.shape[:2]
        student_feat = student_feat.view(n, c, -1)
        teacher_feat = teacher_feat.view(n, c, -1)
        mse = F.mse_loss(student_feat, teacher_feat, reduction="none")
        return (mse * teacher_score).sum() / (teacher_score.sum() * c + 1e-9)

    @property
    def model(self):
        """Forward .model access to the student's underlying nn.Sequential of yaml-parsed layers.

        Many validators and exporters in the codebase access `model.model[-1]` to read head
        attributes (e.g. one_to_many_groups, reg_max). Property avoids duplicate submodule
        registration while making those code paths transparent to the distillation wrapper.
        """
        return self.student_model.model

    @property
    def criterion(self):
        """Expose the student model's loss criterion."""
        return self.student_model.criterion

    @criterion.setter
    def criterion(self, value) -> None:
        """Forward criterion updates to the student model."""
        self.student_model.criterion = value

    def init_criterion(self):
        """Initialize the loss criterion via the student model."""
        return self.student_model.init_criterion()

    def fuse(self, verbose: bool = True):
        """Fuse student model layers for inference speedup."""
        self.student_model.fuse(verbose)
        return self
