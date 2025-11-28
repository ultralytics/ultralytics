"""
YOLOv11 Stereo 3D Detection - Complete Implementation
A stereo 3D detection network based on Stereo CenterNet
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR


# ============================================================================
# Part 1: Feature Fusion Layer
# ============================================================================

class StereoFeatureFusion(nn.Module):
    """Stereo feature fusion module."""

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, left_feat: torch.Tensor, right_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            left_feat: [B, C, H, W]
            right_feat: [B, C, H, W]
        Returns:
            fused: [B, C_out, H, W]
        """
        concat_feat = torch.cat([left_feat, right_feat], dim=1)
        fused = self.fusion_conv(concat_feat)
        return fused


# ============================================================================
# Part 2: Neck (FPN-style Feature Pyramid)
# ============================================================================

class StereoPAN(nn.Module):
    """Stereo Path Aggregation Network (PAN)."""

    def __init__(self, in_channels: int = 256, out_channels: int = 256):
        super().__init__()

        # Top-Down pathway
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.td_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels + out_channels, out_channels, 3, 1, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        # Bottom-Up pathway
        self.bu_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 2, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(out_channels + in_channels, out_channels, 3, 1, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: [P3, P4, P5] multi-scale features from backbone
        Returns:
            [P3_out, P4_out, P5_out]
        """
        # Simplified: only handle P3 (primary detection layer).
        # In practice, handle multiple levels.
        return features  # placeholder


# ============================================================================
# Part 3: Detection Head (10 branches)
# ============================================================================

class StereoCenterNetHead(nn.Module):
    """Stereo CenterNet detection head with 10 parallel branches."""

    def __init__(self, in_channels: int = 256, num_classes: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Shared feature extractor (optional)
        self.shared_head = self._build_shared_head(in_channels)

        # Ten branch definitions
        self.branches = nn.ModuleDict(
            {
                # Task A: Stereo 2D detection (5 branches)
                "heatmap": self._build_branch(in_channels, num_classes),
                "offset": self._build_branch(in_channels, 2),
                "bbox_size": self._build_branch(in_channels, 2),
                "lr_distance": self._build_branch(in_channels, 1),
                "right_width": self._build_branch(in_channels, 1),
                # Task B: 3D components (5 branches)
                "dimensions": self._build_branch(in_channels, 3),
                "orientation": self._build_branch(in_channels, 8),
                "vertices": self._build_branch(in_channels, 8),
                "vertex_offset": self._build_branch(in_channels, 8),
                "vertex_dist": self._build_branch(in_channels, 4),
            }
        )

    def _build_shared_head(self, in_channels: int) -> nn.Sequential:
        """Shared feature extraction head."""
        return nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def _build_branch(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Build a single branch: Conv(3×3) → BN → ReLU → Conv(1×1)."""
        return nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 1, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] fused features from the neck
        Returns:
            Dict of 10 branch outputs
        """
        # Shared feature extraction
        shared_feat = self.shared_head(x)  # [B, 256, H, W]

        # Run the 10 branches in parallel
        outputs = {}
        for branch_name, branch_module in self.branches.items():
            outputs[branch_name] = branch_module(shared_feat)

        return outputs


# ============================================================================
# Part 4: Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    """Focal loss (for heatmaps) to address class imbalance."""

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Focal loss:
        L = -1/N * Σ [
            (1-ŷ)^α * log(ŷ)         if y=1
            (1-y)^β * ŷ^α * log(1-ŷ) if y=0
        ]
        """
        pred = torch.clamp(pred, 1e-4, 1 - 1e-4)

        pos_mask = target.eq(1)
        neg_mask = target.lt(1)

        pos_loss = -torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_mask
        neg_loss = (
            -torch.log(1 - pred) * torch.pow(pred, self.alpha) * torch.pow(1 - target, self.beta) * neg_mask
        )

        num_pos = pos_mask.sum().float()
        num_pos = torch.clamp(num_pos, min=1.0)

        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss


class StereoCenterNetLoss(nn.Module):
    """Total loss for Stereo CenterNet."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.focal_loss_fn = FocalLoss(alpha=2.0, beta=4.0)
        self.num_classes = num_classes

        # Per-branch weights
        self.loss_weights = {
            "heatmap": 1.0,
            "offset": 1.0,
            "bbox_size": 0.1,
            "lr_distance": 1.0,
            "right_width": 0.1,
            "dimensions": 0.1,
            "orientation": 1.0,
            "vertices": 1.0,
            "vertex_offset": 1.0,
            "vertex_dist": 1.0,
        }

    def forward(self, predictions: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the loss for each branch.

        Args:
            predictions: Dict of network outputs
            targets: Dict of ground truth targets
        Returns:
            (total_loss, loss_dict)
        """
        losses = {}

        # ===== Task A: Stereo 2D Detection =====

        # 1. Heatmap (Focal Loss)
        losses["heatmap"] = self.focal_loss_fn(predictions["heatmap"], targets["heatmap"])

        # 2. Center offset (L1, only at heatmap points)
        losses["offset"] = self.masked_l1_loss(
            predictions["offset"], targets["offset"], mask=targets.get("heatmap", None)
        )

        # 3. 2D box size (Smooth L1)
        losses["bbox_size"] = self.masked_l1_loss(
            predictions["bbox_size"], targets["bbox_size"], mask=targets.get("heatmap", None)
        )

        # 4. Left-right center distance (L1)
        losses["lr_distance"] = self.masked_l1_loss(
            predictions["lr_distance"], targets["lr_distance"], mask=targets.get("heatmap", None)
        )

        # 5. Right box width (special handling)
        losses["right_width"] = self.right_width_loss(
            predictions["right_width"], targets["right_width"], mask=targets.get("heatmap", None)
        )

        # ===== Task B: 3D Components =====

        # 6. 3D dimensions (L1)
        losses["dimensions"] = self.masked_l1_loss(
            predictions["dimensions"], targets["dimensions"], mask=targets.get("heatmap", None)
        )

        # 7. Orientation angle (Multi-Bin)
        losses["orientation"] = self.orientation_loss(
            predictions["orientation"], targets["orientation"], mask=targets.get("heatmap", None)
        )

        # 8. Vertex coordinates (L1)
        losses["vertices"] = self.masked_l1_loss(
            predictions["vertices"], targets["vertices"], mask=targets.get("heatmap", None)
        )

        # 9. Vertex sub-pixel offset (L1)
        losses["vertex_offset"] = self.masked_l1_loss(
            predictions["vertex_offset"], targets["vertex_offset"], mask=targets.get("heatmap", None)
        )

        # 10. Vertex distance (L1)
        losses["vertex_dist"] = self.masked_l1_loss(
            predictions["vertex_dist"], targets["vertex_dist"], mask=targets.get("heatmap", None)
        )

        # Weighted sum
        total_loss = sum(self.loss_weights[k] * v for k, v in losses.items())

        return total_loss, losses

    def masked_l1_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Smooth L1 loss with optional mask."""
        loss = F.smooth_l1_loss(pred, target, reduction="none")

        if mask is not None:
            # Expand mask to pred dims
            if mask.dim() == 3:  # [B, H, W]
                mask = mask.unsqueeze(1).expand_as(pred)  # [B, C, H, W]

            mask = mask.float()
            loss = loss * mask

            num_pos = mask.sum().float()
            num_pos = torch.clamp(num_pos, min=1.0)
            loss = loss.sum() / num_pos
        else:
            loss = loss.mean()

        return loss

    def right_width_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Special loss for right-box width (with sigmoid transform)."""
        # Apply sigmoid transform: wr = 1/σ(ŵr) - 1
        pred_transformed = 1.0 / (torch.sigmoid(pred) + 1e-4) - 1.0

        loss = F.l1_loss(pred_transformed, target, reduction="none")

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1).expand_as(pred)

            mask = mask.float()
            loss = loss * mask
            num_pos = mask.sum().float()
            num_pos = torch.clamp(num_pos, min=1.0)
            loss = loss.sum() / num_pos
        else:
            loss = loss.mean()

        return loss

    def orientation_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Orientation angle loss (Multi-Bin encoding).

        pred: [B, 8, H, W]
            layout: [bin_logit_1, bin_logit_2, sin_1, cos_1, sin_2, cos_2, pad, pad]
        target: [B, 8, H, W]
            layout: [bin_id, bin_id, sin, cos, ...]
        """
        # Split bin classification and angle regression
        bin_pred = pred[:, :2, :, :]  # [B, 2, H, W]
        angle_pred = pred[:, 2:6, :, :]  # [B, 4, H, W]

        bin_target = target[:, 0:1, :, :].long().squeeze(1)  # [B, H, W]
        angle_target = target[:, 2:6, :, :]  # [B, 4, H, W]

        # Bin classification loss
        bin_loss = F.cross_entropy(bin_pred, bin_target, reduction="none")  # [B, H, W]

        # Angle regression loss (sin/cos only, 4 dims)
        angle_loss = F.l1_loss(angle_pred, angle_target, reduction="none")
        angle_loss = angle_loss.mean(dim=1)  # [B, H, W]

        # Total loss
        total_loss = bin_loss + 0.5 * angle_loss

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.float()
            total_loss = total_loss * mask
            num_pos = mask.sum().float()
            num_pos = torch.clamp(num_pos, min=1.0)
            total_loss = total_loss.sum() / num_pos
        else:
            total_loss = total_loss.mean()

        return total_loss


class UncertaintyWeightedLoss(nn.Module):
    """Uncertainty-weighted multi-task loss (Kendall et al., 2018)."""

    def __init__(self, num_tasks: int = 10):
        super().__init__()
        self.log_vars = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_tasks)])

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        L_total = Σ_i (exp(-log_var_i) * L_i + log_var_i)

        Args:
            losses: Dict of individual task losses
        Returns:
            total weighted loss
        """
        total_loss = 0

        for i, (task_name, loss_val) in enumerate(losses.items()):
            log_var = self.log_vars[i]
            # L_i_weighted = exp(-σ_i) * L_i + σ_i
            weighted_loss = torch.exp(-log_var) * loss_val + log_var
            total_loss = total_loss + weighted_loss

        return total_loss


# ============================================================================
# Part 5: Full Model
# ============================================================================

class StereoYOLOv11(nn.Module):
    """YOLOv11 Stereo 3D Detection - Full model."""

    def __init__(self, backbone_type: str = "resnet18", num_classes: int = 3, in_channels: int = 256):
        super().__init__()

        self.backbone_type = backbone_type
        self.num_classes = num_classes

        # 1. Backbone (shared)
        self.backbone = self._build_backbone(backbone_type)
        backbone_out_channels = self._get_backbone_channels(backbone_type)

        # 2. Feature fusion
        self.fusion = StereoFeatureFusion(in_channels=backbone_out_channels, out_channels=in_channels)

        # 3. Neck (feature pyramid)
        self.neck = StereoPAN(in_channels=in_channels, out_channels=in_channels)

        # 4. Detection heads
        self.heads = StereoCenterNetHead(in_channels=in_channels, num_classes=num_classes)

        # 5. Loss functions
        self.criterion = StereoCenterNetLoss(num_classes=num_classes)
        self.uncertainty_loss = UncertaintyWeightedLoss(num_tasks=10)

    def _build_backbone(self, backbone_type: str) -> nn.Module:
        """Build the backbone network."""
        if backbone_type == "resnet18":
            from torchvision import models

            backbone = models.resnet18(pretrained=True)
            # Remove final FC layer
            return nn.Sequential(*list(backbone.children())[:-2])

        elif backbone_type == "resnet50":
            from torchvision import models

            backbone = models.resnet50(pretrained=True)
            return nn.Sequential(*list(backbone.children())[:-2])

        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

    def _get_backbone_channels(self, backbone_type: str) -> int:
        """Get the number of output channels of the backbone."""
        if "resnet18" in backbone_type:
            return 512
        elif "resnet50" in backbone_type:
            return 2048
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

    def forward(
        self,
        left_img: torch.Tensor,
        right_img: torch.Tensor,
        targets: Optional[Dict] = None,
    ) -> Tuple[Dict, Optional[torch.Tensor]]:
        """
        Full forward pass.

        Args:
            left_img: [B, 3, H, W]
            right_img: [B, 3, H, W]
            targets: Dict of ground truth (training)

        Returns:
            (predictions, loss) or predictions
        """
        # 1. Backbone feature extraction
        left_feat = self.backbone(left_img)  # [B, C_backbone, H/32, W/32]
        right_feat = self.backbone(right_img)  # [B, C_backbone, H/32, W/32]

        # 2. Feature fusion
        fused_feat = self.fusion(left_feat, right_feat)  # [B, 256, H/32, W/32]

        # 3. Neck processing (simplified: single level here)
        # In practice, handle multiple scales
        neck_out = [fused_feat, fused_feat, fused_feat]

        # 4. Detection heads (use P3 output, 4× downsample)
        predictions = self.heads(neck_out[0])

        # 5. Compute loss (training)
        if targets is not None:
            # Per-branch losses
            total_loss, _loss_dict = self.criterion(predictions, targets)
            # (Optional) apply uncertainty weighting
            # total_loss = self.uncertainty_loss(_loss_dict)
            return predictions, total_loss
        else:
            return predictions


class StereoYOLOv11Wrapper(nn.Module):
    """Wrapper that accepts a single 6-channel image tensor and splits into left/right.

    Exposes a YOLO-like interface: forward(img, targets=None) -> (loss, items) during training.
    """

    def __init__(self, backbone_type: str = "resnet18", num_classes: int = 3, in_channels: int = 256):
        super().__init__()
        self.core = StereoYOLOv11(backbone_type=backbone_type, num_classes=num_classes, in_channels=in_channels)
        self.names = {i: str(i) for i in range(num_classes)}

    def forward(self, x, targets: Optional[Dict] = None, augment: bool = False):
        """Accepts either a tensor [B,6,H,W] or a dict with keys 'img' and optional 'targets'."""
        # Unpack input
        if isinstance(x, dict):
            img6 = x.get("img", None)
            # Allow targets to be supplied via input dict if not provided explicitly
            if targets is None:
                targets = x.get("targets", None)
        else:
            img6 = x

        if not isinstance(img6, torch.Tensor):
            raise TypeError("StereoYOLOv11Wrapper expected a Tensor or dict with key 'img'.")

        assert img6.shape[1] == 6, "StereoYOLOv11Wrapper expects a 6-channel input (left+right)."
        left = img6[:, 0:3, :, :]
        right = img6[:, 3:6, :, :]

        if targets is not None:
            preds, total_loss = self.core(left, right, targets)
            # Return a dict with 'loss' key for Ultralytics Trainer compatibility
            return {"loss": total_loss, "preds": preds}
        else:
            preds = self.core(left, right, None)
            return preds


# ============================================================================
# Part 6: Training Utilities
# ============================================================================

class Trainer:
    """Trainer."""

    def __init__(self, model: StereoYOLOv11, device: str = "cuda", learning_rate: float = 1.5e-4):
        self.model = model.to(device)
        self.device = device

        # Optimizer (AdamW)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Learning rate schedule
        self.scheduler = MultiStepLR(self.optimizer, milestones=[40], gamma=0.1)  # decay 10× at epoch 40

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        loss_stats = {}

        for batch_idx, (left_img, right_img, targets) in enumerate(train_loader):
            left_img = left_img.to(self.device)
            right_img = right_img.to(self.device)

            # Forward pass
            predictions, loss, loss_dict = self.model(left_img, right_img, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=35.0)

            self.optimizer.step()

            # Logging stats
            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k not in loss_stats:
                    loss_stats[k] = 0
                loss_stats[k] += v.item()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

        # Average losses
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        for k in loss_stats:
            loss_stats[k] /= num_batches

        return {"total": avg_loss, **loss_stats}

    def step_scheduler(self):
        """Step the learning rate scheduler."""
        self.scheduler.step()


if __name__ == "__main__":
    # Create model
    model = StereoYOLOv11(backbone_type="resnet18", num_classes=3, in_channels=256)

    # Create dummy inputs
    left_img = torch.randn(2, 3, 384, 1280)
    right_img = torch.randn(2, 3, 384, 1280)

    # Create dummy targets (None for inference-only)
    targets = None

    # Forward pass
    if targets is None:
        predictions = model(left_img, right_img)

        # Inspect outputs
        for branch_name, output in predictions.items():
            print(f"{branch_name}: {output.shape}")
    else:
        predictions, loss, loss_dict = model(left_img, right_img, targets)
        print(f"Total Loss: {loss.item():.4f}")
        for k, v in loss_dict.items():
            print(f"  {k}: {v.item():.4f}")
