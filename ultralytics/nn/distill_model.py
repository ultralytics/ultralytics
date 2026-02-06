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
        if isinstance(feats_idx, int):
            feats_idx = [feats_idx]
        if isinstance(teacher_model, str):
            teacher_model = load_checkpoint(teacher_model)[0]
        device = next(student_model.parameters()).device
        # self.teacher_model = teacher_model.to(device).eval()
        self.teacher_model = teacher_model.to(device).train()
        for v in self.teacher_model.parameters():
            v.requires_grad = False
        self.student_model = student_model
        self.feats_idx = feats_idx
        # get the feature dimensions
        with torch.inference_mode():
            teacher_output = teacher_model(torch.zeros(1, 3, 256, 256).to(device), embed=feats_idx, direct_return=True)
            student_output = student_model(torch.zeros(1, 3, 256, 256).to(device), embed=feats_idx, direct_return=True)
            assert len(teacher_output) == len(student_output), "Feature dimensions must match in length."
        copy_attr(self, student_model)
        self.distill_box_loss = self.student_model.args.distill_box_loss
        self.distill_cls_loss = self.student_model.args.distill_cls_loss
        self.distill_feature_loss = self.student_model.args.distill_feature_loss
        self.distill_box = self.student_model.args.distill_box
        self.distill_cls = self.student_model.args.distill_cls
        self.distill_feature = self.student_model.args.distill_feature
        if self.distill_feature_loss:
            projectors = []
            for student_out, teacher_out in zip(student_output, teacher_output):
                student_dim = self.decouple_outputs(student_out, shape_check=True).shape[1]
                teacher_dim = self.decouple_outputs(teacher_out, shape_check=True).shape[1]
                projectors.append(nn.Conv2d(student_dim, teacher_dim, kernel_size=1, stride=1, padding=0) if student_dim != teacher_dim else nn.Identity())
            self.projector = nn.ModuleList(projectors)
        if self.distill_feature_loss == "mgd":
            generations = []
            for teacher_out in teacher_output:
                if not isinstance(teacher_out, dict):
                    teacher_dim = teacher_out.shape[1]
                    generations.append(
                        nn.Sequential(
                            nn.Conv2d(teacher_dim, teacher_dim, kernel_size=3, padding=1),
                            nn.SiLU(),
                            nn.Conv2d(teacher_dim, teacher_dim, kernel_size=3, padding=1),
                        )
                    )
            self.generation = nn.ModuleList(generations)
        if self.distill_feature_loss == "cwd":
            norms = []
            for teacher_out in teacher_output:
                if not isinstance(teacher_out, dict):
                    teacher_dim = teacher_out.shape[1]
                    norms.append(nn.BatchNorm2d(teacher_dim, affine=False))
            self.norm = nn.ModuleList(norms)
        self.distill_area = self.student_model.args.distill_area
        self.distill_branch = self.student_model.args.distill_branch
        self.distill_branch = self.student_model.args.distill_branch.split(",")
        for branch in self.distill_branch:
            assert branch in {"one2one", "one2many"}

    def loss_kl(self, student_logits, teacher_logits, temperature: float = 5.0):
        """The KL divergence loss for knowledge distillation."""
        soft_targets = F.softmax(teacher_logits / temperature, dim=1)  # train does not have softmax
        student_soft_logits = F.log_softmax(student_logits / temperature, dim=1)

        # Distillation loss (Kullback-Leibler divergence)
        distillation_loss = F.kl_div(student_soft_logits, soft_targets, reduction="mean") * (temperature**2)
        # distillation_loss = distillation_loss / teacher_logits.shape[-1]  # divide the number of anchors
        return distillation_loss

    def forward(self, x, *args, **kwargs):
        """Forward pass through the student model."""
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.student_model.predict(x, *args, **kwargs)
        # y = []  # outputs
        # sx = x.clone()
        # for i, m in enumerate(self.student_model.model):
        #     sx = m(sx)  # run
        #     if i == 10:
        #         sx = self.projector[0](sx.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        #         break
        # for i, m in enumerate(self.teacher_model.model):
        #     if m.f != -1:  # if not from previous layer
        #         x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        #     x = m(x)  # run
        #     if i == 10:
        #         y.append(sx)
        #     else:
        #         y.append(x if m.i in self.save else None)  # save output
        # return x

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        with torch.inference_mode():
            teacher_feats = self.teacher_model(batch["img"], embed=self.feats_idx, direct_return=True)
        preds, feats = self.student_model(batch["img"], return_feats=True)

        regular_loss, regular_loss_detach, targets = self.student_model.loss(batch, preds, return_targets=True)

        loss_distill_cls = torch.zeros(1, device=batch["img"].device)
        loss_distill_box = torch.zeros(1, device=batch["img"].device)
        loss_distill_feature = torch.zeros(1, device=batch["img"].device)
        for i, feat_idx in enumerate(self.feats_idx):
            # handle head ouput
            teacher_feat = self.decouple_outputs(teacher_feats[i])
            student_feat = self.decouple_outputs(feats[feat_idx])
            assert isinstance(teacher_feat, type(student_feat)), (
                f"Expect same type for teacher feature and student feature, but got teacher: {type(teacher_feat)} and student: {type(student_feat)}"
            )
            # means distill head, and the output shape should be exactly the same
            if isinstance(teacher_feat, dict):
                for branch in self.distill_branch:
                    teacher_feat = self.decouple_outputs(teacher_feats[i], branch=branch)
                    student_feat = self.decouple_outputs(feats[feat_idx], branch=branch)
                    assert "boxes" in teacher_feat and "scores" in teacher_feat
                    if self.distill_cls_loss:
                        teacher_logits = teacher_feat["scores"]
                        student_logits = student_feat["scores"]
                        teacher_logits = teacher_logits.permute(0, 2, 1).contiguous()  # (bs, c, anchors) -> (bs, anchors, c)
                        student_logits = student_logits.permute(0, 2, 1).contiguous()
                        if self.distill_area == "main":
                            fg_mask = targets[branch][0]
                            teacher_logits = teacher_logits[fg_mask]  # (n, c)
                            student_logits = student_logits[fg_mask]
                        else:
                            teacher_logits = teacher_logits.view(-1, teacher_logits.shape[-1])  # (bs, anchors, c) -> (bs*anchors, c)
                            student_logits = student_logits.view(-1, student_logits.shape[-1])
                        loss_distill_cls += self.cls_kd_loss(student_logits, teacher_logits) * self.distill_cls
                    if self.distill_box_loss:
                        teacher_boxes = teacher_feat["boxes"]
                        student_boxes = student_feat["boxes"]
                        if self.distill_area == "main":
                            teacher_boxes = teacher_boxes.permute(0, 2, 1).contiguous()  # (bs, c, anchors) -> (bs, anchors, c)
                            student_boxes = student_boxes.permute(0, 2, 1).contiguous()
                            fg_mask = targets[branch][0]
                            teacher_boxes = teacher_boxes[fg_mask]  # (n, c)
                            student_boxes = student_boxes[fg_mask]
                        loss_distill_box += self.box_kd_loss(student_boxes, teacher_boxes) * self.distill_box
            else:
                if self.distill_feature_loss:
                    student_feat = (
                        self.projector[i](student_feat)
                        if student_feat.ndim == 4
                        else student_feat
                    )
                    loss_distill_feature += self.feature_kd_loss(student_feat, teacher_feat, feat_idx=i) * self.distill_feature

        loss_distill_detach = (loss_distill_cls + loss_distill_box + loss_distill_feature).detach()
        batch_size = batch["img"].shape[0]
        loss_distill = loss_distill_cls + loss_distill_box + loss_distill_feature
        return torch.cat([regular_loss, loss_distill]), torch.cat([regular_loss_detach, loss_distill_detach])

    def loss_cosine(self, student_feat, teacher_feat):
        """Compute cosine similarity loss between teacher and student features."""
        if student_feat.ndim == 4:
            student_feat = student_feat.flatten(2).permute(0, 2, 1)
        if teacher_feat.ndim == 4:
            teacher_feat = teacher_feat.flatten(2).permute(0, 2, 1)
        student_feat = F.normalize(student_feat, p=2, dim=-1)
        teacher_feat = F.normalize(teacher_feat, p=2, dim=-1)
        cos_sim = F.cosine_similarity(student_feat, teacher_feat, dim=-1)
        loss = (1 - cos_sim).mean()
        return loss

    def loss_mgd(self, student_feat, teacher_feat, lambda_mgd=0.65, feat_idx=0):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = teacher_feat.shape

        device = student_feat.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(student_feat, mat)
        new_fea = self.generation[feat_idx](masked_fea)

        dis_loss = loss_mse(new_fea, teacher_feat) / N
        return dis_loss

    def loss_cwd(self, student_feat, teacher_feat, feat_idx=0, temperature: float = 1.0):
        student_feat = self.norm[feat_idx](student_feat)
        teacher_feat = self.norm[feat_idx](teacher_feat)

        N, C, H, W = teacher_feat.shape
        softmax_pred_T = F.softmax(teacher_feat.view(-1, W * H) / temperature, dim=1)  # [N*C, H*W]
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        cost = torch.sum(
            softmax_pred_T * logsoftmax(teacher_feat.view(-1, W * H) / temperature) -
            softmax_pred_T * logsoftmax(student_feat.view(-1, W * H) / temperature)) * (temperature**2)

        dis_loss = cost / (N * C)
        return dis_loss

    def box_kd_loss(self, student_boxes, teacher_boxes):
        if self.distill_box_loss == "cos":
            return self.loss_cosine(student_boxes, teacher_boxes)
        elif self.distill_box_loss == "l1":
            return F.l1_loss(student_boxes, teacher_boxes)
        elif self.distill_box_loss == "l2":
            return F.mse_loss(student_boxes, teacher_boxes)
        else:
            raise ValueError(f"Unknown box distillation loss: {self.distill_box_loss}")

    def cls_kd_loss(self, student_logits, teacher_logits, temperature=5.0):
        from ultralytics.utils.torch_utils import autocast
        with autocast(enabled=False):
            student_logits = student_logits.float()
            teacher_logits = teacher_logits.float()
            if self.distill_cls_loss == "softmax":
                return self.loss_kl(student_logits, teacher_logits, temperature)
            elif self.distill_cls_loss == "sigmoid":
                distillation_loss = F.binary_cross_entropy_with_logits(
                        student_logits / temperature,
                        torch.sigmoid(teacher_logits / temperature),
                        reduction='mean'
                    ) * (temperature ** 2)
                return distillation_loss
            else:
                raise ValueError(f"Unknown cls distillation loss: {self.distill_cls_loss}")

    def feature_kd_loss(self, student_feat, teacher_feat, feat_idx=0):
        if self.distill_feature_loss == "cos":
            return self.loss_cosine(student_feat, teacher_feat)
        elif self.distill_feature_loss == "l1":
            return F.l1_loss(student_feat, teacher_feat)
        elif self.distill_feature_loss == "l2":
            return F.mse_loss(student_feat, teacher_feat)
        elif self.distill_feature_loss == "mgd":
            return self.loss_mgd(student_feat, teacher_feat, feat_idx=feat_idx)
        elif self.distill_feature_loss == "cwd":
            return self.loss_cwd(student_feat, teacher_feat, feat_idx=feat_idx)
        else:
            raise ValueError(f"Unknown box distillation loss: {self.distill_feature_loss}")

    @property
    def criterion(self):
        """Get the criterion from the student model."""
        return self.student_model.criterion

    @criterion.setter
    def criterion(self, value) -> None:
        """Set value for student criterion."""
        self.student_model.criterion = value

    def fuse(self, verbose: bool = True):
        """Fuse model layers for inference speedup."""
        self.student_model.fuse(verbose)
        return self

    def decouple_outputs(self, preds, shape_check=False, branch="one2one"):
        """Decouple outputs for teacher/student models."""
        if isinstance(preds, tuple):  # decouple for val mode
            preds = preds[1]
        if isinstance(preds, dict):
            if branch in preds:
                preds = preds[branch]
            if shape_check:
                preds = preds["boxes"]
        return preds
