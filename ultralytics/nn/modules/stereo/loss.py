# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Multi-task loss functions for Stereo CenterNet with uncertainty weighting."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for heatmap and vertices branches.

    Follows constitution formula:
    L = -1/N Σ {
        (1-ŷ)^α log(ŷ)           if y=1 (正样本)
        (1-y)^β ŷ^α log(1-ŷ)     otherwise (负样本)
    }

    Args:
        alpha: Focal loss alpha parameter (default: 2).
        beta: Focal loss beta parameter (default: 4).
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        """Initialize Focal Loss.

        Args:
            alpha: Alpha parameter for focal loss.
            beta: Beta parameter for focal loss.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            pred: Predicted heatmap [B, C, H, W] (before sigmoid).
            target: Target Gaussian heatmap [B, C, H, W] (values in [0, 1]).

        Returns:
            Scalar loss value.
        """
        # Apply sigmoid to predictions
        pred_sigmoid = torch.sigmoid(pred)
        pred_sigmoid = torch.clamp(pred_sigmoid, min=1e-4, max=1 - 1e-4)

        # Positive and negative masks
        pos_mask = target.eq(1.0)
        neg_mask = target.lt(1.0)

        # Positive loss: (1-ŷ)^α log(ŷ)
        pos_loss = -torch.log(pred_sigmoid) * torch.pow(1 - pred_sigmoid, self.alpha) * pos_mask

        # Negative loss: (1-y)^β ŷ^α log(1-ŷ)
        neg_loss = (
            -torch.log(1 - pred_sigmoid)
            * torch.pow(pred_sigmoid, self.alpha)
            * torch.pow(1 - target, self.beta)
            * neg_mask
        )

        # Normalize by number of positive samples
        num_pos = pos_mask.sum().float()
        num_pos = torch.clamp(num_pos, min=1.0)

        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss


class L1Loss(nn.Module):
    """L1 Loss for regression branches.

    Computes L1 loss only at positive locations (object centers or vertices).
    """

    def __init__(self):
        """Initialize L1 Loss."""
        super().__init__()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute L1 loss.

        Args:
            pred: Predictions [B, C, H, W].
            target: Targets [B, C, H, W].
            mask: Optional mask [B, C, H, W] or [B, 1, H, W] indicating positive locations.

        Returns:
            Scalar loss value.
        """
        loss = torch.abs(pred - target)

        # Apply mask if provided (only compute at positive locations)
        if mask is not None:
            # Expand mask to match pred shape if needed
            if mask.shape[1] == 1 and pred.shape[1] > 1:
                mask = mask.expand_as(pred)
            loss = loss * mask

            # Normalize by number of positive samples
            num_pos = mask.sum().float()
            num_pos = torch.clamp(num_pos, min=1.0)
            loss = loss.sum() / num_pos
        else:
            # Average over all locations
            loss = loss.mean()

        return loss


class OrientationLoss(nn.Module):
    """Multi-bin orientation loss (CE + L1).

    Combines classification loss for bin selection and L1 loss for sin/cos regression.
    """

    def __init__(self):
        """Initialize Orientation Loss."""
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.l1_loss = nn.L1Loss(reduction="none")

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute orientation loss.

        Args:
            pred: Predictions [B, 8, H, W] (2 bins × 4 values: bin_logits[2], sin[2], cos[2]).
            target: Targets [B, 8, H, W] (same format).
            mask: Optional mask indicating positive locations.

        Returns:
            Scalar loss value.
        """
        # Split into bin classification and sin/cos regression
        bin_logits = pred[:, :2, :, :]  # [B, 2, H, W] - bin classification logits
        sin_cos_pred = pred[:, 2:, :, :]  # [B, 6, H, W] - sin/cos predictions

        bin_target = target[:, :2, :, :]  # [B, 2, H, W] - bin targets (one-hot)
        sin_cos_target = target[:, 2:, :, :]  # [B, 6, H, W] - sin/cos targets

        # Classification loss (CE)
        bin_target_idx = bin_target.argmax(dim=1)  # [B, H, W]
        ce_loss = self.ce_loss(bin_logits, bin_target_idx)  # [B, H, W]

        # Regression loss (L1 on sin/cos)
        l1_loss = self.l1_loss(sin_cos_pred, sin_cos_target).mean(dim=1)  # [B, H, W]

        # Combine losses
        loss = ce_loss + l1_loss  # [B, H, W]

        # Apply mask if provided
        if mask is not None:
            if mask.shape[1] == 1:
                mask = mask.squeeze(1)  # [B, H, W]
            loss = loss * mask
            num_pos = mask.sum().float()
            num_pos = torch.clamp(num_pos, min=1.0)
            loss = loss.sum() / num_pos
        else:
            loss = loss.mean()

        return loss


class RightWidthLoss(nn.Module):
    """Right width loss with Sigmoid transform.

    Uses formula: wr = 1/σ(ŵr) - 1
    """

    def __init__(self):
        """Initialize Right Width Loss."""
        super().__init__()
        self.l1_loss = L1Loss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute right width loss.

        Args:
            pred: Predictions [B, 1, H, W] (before sigmoid transform).
            target: Targets [B, 1, H, W] (actual right width values).

        Returns:
            Scalar loss value.
        """
        # Apply sigmoid transform: wr = 1/σ(ŵr) - 1
        pred_transformed = 1.0 / (torch.sigmoid(pred) + 1e-8) - 1.0

        # Compute L1 loss
        loss = self.l1_loss(pred_transformed, target, mask)
        return loss


class StereoLoss(nn.Module):
    """Uncertainty-weighted multi-task loss for Stereo CenterNet.

    Uses formula: L_total = Σᵢ (1/(2σᵢ²)) × Lᵢ + log(σᵢ)

    Where σᵢ are learnable uncertainty parameters.
    """

    def __init__(self, num_classes: int = 3):
        """Initialize Stereo Loss.

        Args:
            num_classes: Number of object classes.
        """
        super().__init__()
        self.num_classes = num_classes

        # Loss functions for each branch
        self.focal_loss = FocalLoss(alpha=2.0, beta=4.0)
        self.l1_loss = L1Loss()
        self.orientation_loss = OrientationLoss()
        self.right_width_loss = RightWidthLoss()

        # Learnable uncertainty parameters (σᵢ) - initialized to 1.0
        self.uncertainty = nn.ParameterDict(
            {
                "heatmap": nn.Parameter(torch.ones(1)),
                "offset": nn.Parameter(torch.ones(1)),
                "bbox_size": nn.Parameter(torch.ones(1)),
                "lr_distance": nn.Parameter(torch.ones(1)),
                "right_width": nn.Parameter(torch.ones(1)),
                "dimensions": nn.Parameter(torch.ones(1)),
                "orientation": nn.Parameter(torch.ones(1)),
                "vertices": nn.Parameter(torch.ones(1)),
                "vertex_offset": nn.Parameter(torch.ones(1)),
                "vertex_dist": nn.Parameter(torch.ones(1)),
            }
        )

    def forward(
        self, predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute uncertainty-weighted multi-task loss.

        Args:
            predictions: Dictionary of 10 branch predictions.
            targets: Dictionary of ground truth targets.

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains per-branch losses.
        """
        losses = {}

        # Get heatmap mask for positive locations
        heatmap_mask = targets.get("heatmap", None)
        if heatmap_mask is not None:
            # Create binary mask (1.0 at positive locations)
            heatmap_mask = (heatmap_mask > 0.5).float()

        # Branch 1: Heatmap (Focal Loss)
        losses["heatmap"] = self.focal_loss(predictions["heatmap"], targets["heatmap"])

        # Branch 2: Offset (L1 Loss)
        losses["offset"] = self.l1_loss(predictions["offset"], targets["offset"], heatmap_mask)

        # Branch 3: Bbox size (L1 Loss)
        losses["bbox_size"] = self.l1_loss(predictions["bbox_size"], targets["bbox_size"], heatmap_mask)

        # Branch 4: LR distance (L1 Loss)
        losses["lr_distance"] = self.l1_loss(
            predictions["lr_distance"], targets["lr_distance"], heatmap_mask
        )

        # Branch 5: Right width (L1 Loss + Sigmoid transform)
        losses["right_width"] = self.right_width_loss(
            predictions["right_width"], targets["right_width"], heatmap_mask
        )

        # Branch 6: Dimensions (L1 Loss)
        losses["dimensions"] = self.l1_loss(predictions["dimensions"], targets["dimensions"], heatmap_mask)

        # Branch 7: Orientation (CE + L1)
        losses["orientation"] = self.orientation_loss(
            predictions["orientation"], targets["orientation"], heatmap_mask
        )

        # Branch 8: Vertices (Focal Loss)
        losses["vertices"] = self.focal_loss(predictions["vertices"], targets["vertices"])

        # Branch 9: Vertex offset (L1 Loss)
        losses["vertex_offset"] = self.l1_loss(
            predictions["vertex_offset"], targets["vertex_offset"], heatmap_mask
        )

        # Branch 10: Vertex distance (L1 Loss)
        losses["vertex_dist"] = self.l1_loss(
            predictions["vertex_dist"], targets["vertex_dist"], heatmap_mask
        )

        # Uncertainty-weighted total loss: L_total = Σᵢ (1/(2σᵢ²)) × Lᵢ + log(σᵢ)
        total_loss = torch.tensor(0.0, device=list(predictions.values())[0].device)
        for branch_name, loss_value in losses.items():
            sigma = self.uncertainty[branch_name]
            weighted_loss = (1.0 / (2.0 * sigma**2)) * loss_value + torch.log(sigma + 1e-8)
            total_loss = total_loss + weighted_loss

        return total_loss, losses

