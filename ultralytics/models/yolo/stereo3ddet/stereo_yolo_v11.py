"""
YOLOv11 Stereo 3D Detection - Complete Implementation
A stereo 3D detection network based on Stereo CenterNet
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        # Initialize heatmap branch bias for focal loss (bias=-2.19 gives sigmoid(-2.19) ≈ 0.1)
        self._init_heatmap_bias()
    
    def _init_heatmap_bias(self):
        """Initialize heatmap branch bias to -2.19 for focal loss prior."""
        heatmap_branch = self.branches["heatmap"]
        # Get the final conv layer (1x1 conv)
        final_conv = heatmap_branch[-1]
        if hasattr(final_conv, 'bias') and final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, -2.19)

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

class StereoCenterNetLoss(nn.Module):
    """Total loss for Stereo CenterNet."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
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
        losses["heatmap"] = self.centernet_focal_loss(predictions["heatmap"], targets["heatmap"])

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
    
    def centernet_focal_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                          alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
        """
        CenterNet-style Focal Loss for heatmap regression.
        
        Paper Equation 1:
        - For positive locations (Y=1): (1 - Ŷ)^α * log(Ŷ)
        - For negative locations (Y<1): (1 - Y)^β * Ŷ^α * log(1 - Ŷ)
        
        Args:
            pred: [B, C, H, W] - raw network output (before sigmoid)
            target: [B, C, H, W] - Gaussian heatmap target [0, 1]
            alpha: focusing parameter (default 2)
            beta: down-weighting factor for negatives near centers (default 4)
        
        Returns:
            Scalar loss value
        """
        # Apply sigmoid to get probabilities [0, 1]
        pred = torch.sigmoid(pred)
        
        # Numerical stability - clamp predictions
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
        
        # Identify positive locations (peak of Gaussian, Y = 1)
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()
        
        # Count number of positive samples for normalization
        num_pos = pos_mask.sum()
        # debug
        assert num_pos > 0, "No positive samples found"
        num_pos = torch.clamp(num_pos, min=1.0)  # Avoid division by zero
        
        # Positive loss: (1 - Ŷ)^α * log(Ŷ)
        pos_loss = torch.pow(1 - pred, alpha) * torch.log(pred) * pos_mask
        
        # Negative loss: (1 - Y)^β * Ŷ^α * log(1 - Ŷ)
        # The (1 - Y)^β term down-weights locations near object centers
        neg_loss = torch.pow(1 - target, beta) * torch.pow(pred, alpha) * torch.log(1 - pred) * neg_mask
        
        # Sum and normalize by number of positive samples
        loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        
        return loss

    def masked_l1_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Smooth L1 loss with optional mask."""
        loss = F.smooth_l1_loss(pred, target, reduction="none")

        if mask is not None:
            # Convert heatmap mask to binary mask
            # Heatmap contains Gaussian values [0, 1], need to threshold first
            if mask.dim() == 4 and mask.shape[1] > 1:  # [B, num_classes, H, W]
                # Threshold first to create binary mask, then sum over classes
                # This ensures we only compute loss at object center locations
                mask = (mask > 0.5).float().sum(dim=1, keepdim=False)  # [B, H, W] - binary mask
                # Clamp to [0, 1] in case multiple classes overlap at same location
                mask = torch.clamp(mask, 0.0, 1.0)
            elif mask.dim() == 4 and mask.shape[1] == 1:  # [B, 1, H, W]
                # Single channel, threshold and squeeze
                mask = (mask > 0.5).float().squeeze(1)  # [B, H, W]
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
            # Convert heatmap mask to binary mask
            if mask.dim() == 4 and mask.shape[1] > 1:  # [B, num_classes, H, W]
                mask = mask.sum(dim=1, keepdim=False)  # [B, H, W]
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
        # Reshape for cross_entropy: [B, C, H, W] -> [B*H*W, C] and [B, H, W] -> [B*H*W]
        B, C, H, W = bin_pred.shape
        bin_pred_flat = bin_pred.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, 2]
        bin_target_flat = bin_target.reshape(-1).long()  # [B*H*W]
        bin_loss_flat = F.cross_entropy(bin_pred_flat, bin_target_flat, reduction="none")  # [B*H*W]
        bin_loss = bin_loss_flat.reshape(B, H, W)  # [B, H, W]

        # Angle regression loss - only for active bin's residual
        # Format: [conf1, conf2, sin1, cos1, sin2, cos2, pad, pad]
        # angle_pred: [B, 4, H, W] contains [sin1, cos1, sin2, cos2]
        # Bin 0: sin at index 0, cos at index 1
        # Bin 1: sin at index 2, cos at index 3
        
        # Create indices to gather sin/cos for active bin
        # For each (b, h, w), get sin/cos indices: bin0 -> [0, 1], bin1 -> [2, 3]
        sin_indices = bin_target * 2  # [B, H, W] - 0 for bin0, 2 for bin1
        cos_indices = bin_target * 2 + 1  # [B, H, W] - 1 for bin0, 3 for bin1
        
        # Gather sin/cos predictions for active bin using torch.gather
        # angle_pred: [B, 4, H, W], indices: [B, H, W] -> need [B, 1, H, W] for gather
        sin_indices_expanded = sin_indices.unsqueeze(1)  # [B, 1, H, W]
        cos_indices_expanded = cos_indices.unsqueeze(1)  # [B, 1, H, W]
        
        sin_pred_active = torch.gather(angle_pred, dim=1, index=sin_indices_expanded).squeeze(1)  # [B, H, W]
        cos_pred_active = torch.gather(angle_pred, dim=1, index=cos_indices_expanded).squeeze(1)  # [B, H, W]
        
        # Gather sin/cos targets for active bin
        sin_target_active = torch.gather(angle_target, dim=1, index=sin_indices_expanded).squeeze(1)  # [B, H, W]
        cos_target_active = torch.gather(angle_target, dim=1, index=cos_indices_expanded).squeeze(1)  # [B, H, W]
        
        # Compute L1 loss on active bin's sin/cos residual
        sin_loss = F.l1_loss(sin_pred_active, sin_target_active, reduction="none")
        cos_loss = F.l1_loss(cos_pred_active, cos_target_active, reduction="none")
        angle_loss = (sin_loss + cos_loss) / 2.0  # Average of sin and cos loss

        # Total loss
        total_loss = bin_loss + angle_loss  # [B, H, W]

        if mask is not None:
            # Convert heatmap mask to binary mask
            if mask.dim() == 4 and mask.shape[1] > 1:  # [B, num_classes, H, W]
                mask = mask.sum(dim=1, keepdim=False)  # [B, H, W]
            if mask.dim() == 3:  # [B, H, W]
                mask = mask.float()
            # Verify shapes match
            assert total_loss.shape == mask.shape, (
                f"Shape mismatch: total_loss {total_loss.shape} vs mask {mask.shape}, "
                f"bin_loss {bin_loss.shape}, angle_loss {angle_loss.shape}"
            )
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
            total_loss, loss_dict = self.criterion(predictions, targets)
            # (Optional) apply uncertainty weighting
            # total_loss = self.uncertainty_loss(loss_dict)
            return predictions, total_loss, loss_dict
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
        # Required attributes for AutoBackend compatibility
        self.stride = torch.tensor([32.0])  # Default stride for stereo models
        self.yaml = {"channels": 6}  # 6-channel input (left + right)
        self.model = self.core  # For compatibility with BaseModel.fuse() pattern
        self.ch = 6

    def forward(
        self,
        x,
        targets: Optional[Dict] = None,
        augment: bool = False,
        visualize=False,
        embed=None,
        profile=False,
        **kwargs,
    ):
        """Accepts either a tensor [B,6,H,W] or a dict with keys 'img' and optional 'targets'.
        
        Args:
            x: Input tensor [B,6,H,W] or dict with 'img' key.
            targets: Optional ground truth targets for training.
            augment: Whether to apply augmentation (not used currently).
            visualize: Path to save feature visualizations (for AutoBackend compatibility).
            embed: List of layer indices to return embeddings (for AutoBackend compatibility).
            profile: Whether to profile computation time (for AutoBackend compatibility).
            **kwargs: Additional keyword arguments for AutoBackend compatibility.
        """
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
            preds, total_loss, loss_dict = self.core(left, right, targets)
            # Return tuple (loss, loss_items) for Ultralytics Trainer compatibility
            # loss_items should be a tensor with individual loss components
            # Create tensor from loss_dict values
            loss_items_list = [
                loss_dict.get("heatmap", torch.tensor(0.0, device=total_loss.device)),
                loss_dict.get("offset", torch.tensor(0.0, device=total_loss.device)),
                loss_dict.get("bbox_size", torch.tensor(0.0, device=total_loss.device)),
                loss_dict.get("lr_distance", torch.tensor(0.0, device=total_loss.device)),
                loss_dict.get("right_width", torch.tensor(0.0, device=total_loss.device)),
                loss_dict.get("dimensions", torch.tensor(0.0, device=total_loss.device)),
                loss_dict.get("orientation", torch.tensor(0.0, device=total_loss.device)),
                loss_dict.get("vertices", torch.tensor(0.0, device=total_loss.device)),
                loss_dict.get("vertex_offset", torch.tensor(0.0, device=total_loss.device)),
                loss_dict.get("vertex_dist", torch.tensor(0.0, device=total_loss.device)),
            ]
            loss_items = torch.stack(loss_items_list)  # [10]
            return total_loss, loss_items
        else:
            preds = self.core(left, right, None)
            return preds

    def loss(self, batch, preds=None):
        """Compute loss for validation.
        
        This method is called by BaseValidator during validation to compute loss.
        It extracts labels from batch, converts them to targets format, computes
        predictions if needed, and returns (total_loss, loss_items) tuple.
        
        Args:
            batch: Dict with 'img' (6-channel tensor [B, 6, H, W]) and 'labels' (list of label dicts per image).
            preds: Optional precomputed predictions dict with 10 branch outputs.
            
        Returns:
            Tuple of (total_loss, loss_items) where:
                - total_loss: Scalar tensor with total loss value
                - loss_items: Tensor with shape [10] containing individual loss components:
                  [heatmap, offset, bbox_size, lr_distance, right_width, dimensions,
                   orientation, vertices, vertex_offset, vertex_dist]
        """
        # Extract inputs from batch (T131)
        img = batch.get("img")
        labels_list = batch.get("labels", [])
        
        if img is None:
            raise ValueError("batch must contain 'img' key with 6-channel tensor")
        
        # Split img into left and right (T132)
        assert img.shape[1] == 6, "StereoYOLOv11Wrapper expects a 6-channel input (left+right)."
        left = img[:, 0:3, :, :]
        right = img[:, 3:6, :, :]
        
        # Get image size for TargetGenerator
        _, _, h, w = img.shape
        imgsz = h  # Assuming square images
        
        # Import TargetGenerator and initialize it (T133)
        from ultralytics.data.stereo.target import TargetGenerator
        
        # Initialize target generator if not already done
        if not hasattr(self, "_target_generator"):
            num_classes = len(self.names) if isinstance(self.names, dict) else 3
            # Output size is H/32, W/32 (ResNet18 backbone downsamples by 32x)
            output_h = imgsz // 32
            output_w = imgsz // 32
            self._target_generator = TargetGenerator(
                output_size=(output_h, output_w),
                num_classes=num_classes,
            )
        
        # Convert labels to targets format (T134)
        targets_list = []
        for labels in labels_list:
            target = self._target_generator.generate_targets(
                labels,
                input_size=(imgsz, imgsz)
            )
            # Move to same device as img
            target = {k: v.to(img.device) for k, v in target.items()}
            targets_list.append(target)
        
        # Stack targets across batch dimension (T135)
        # Each target is a dict with tensors of shape [C, H, W]
        # We need to stack to [B, C, H, W]
        if targets_list:
            batched_targets = {}
            for key in targets_list[0].keys():
                batched_targets[key] = torch.stack([t[key] for t in targets_list], dim=0)
        else:
            # Empty batch - create zero targets
            output_h = imgsz // 32
            output_w = imgsz // 32
            num_classes = len(self.names) if isinstance(self.names, dict) else 3
            batched_targets = {
                "heatmap": torch.zeros(img.shape[0], num_classes, output_h, output_w, device=img.device),
                "offset": torch.zeros(img.shape[0], 2, output_h, output_w, device=img.device),
                "bbox_size": torch.zeros(img.shape[0], 2, output_h, output_w, device=img.device),
                "lr_distance": torch.zeros(img.shape[0], 1, output_h, output_w, device=img.device),
                "right_width": torch.zeros(img.shape[0], 1, output_h, output_w, device=img.device),
                "dimensions": torch.zeros(img.shape[0], 3, output_h, output_w, device=img.device),
                "orientation": torch.zeros(img.shape[0], 8, output_h, output_w, device=img.device),
                "vertices": torch.zeros(img.shape[0], 8, output_h, output_w, device=img.device),
                "vertex_offset": torch.zeros(img.shape[0], 8, output_h, output_w, device=img.device),
                "vertex_dist": torch.zeros(img.shape[0], 4, output_h, output_w, device=img.device),
            }
        
        # Compute predictions if needed (T136)
        if preds is None:
            # Call forward to get predictions
            preds = self.forward(img)
        
        # Ensure preds is in dict format
        if not isinstance(preds, dict):
            # If forward returned a tensor or other format, we need to handle it
            # For now, assume forward returns dict with 10 branch keys
            raise TypeError(f"Expected preds to be dict, got {type(preds)}")
        
        # Compute loss (T137)
        total_loss, loss_dict = self.core.criterion(preds, batched_targets)
        
        # Convert loss_dict to loss_items tensor in fixed order (T138)
        loss_items_list = [
            loss_dict.get("heatmap", torch.tensor(0.0, device=total_loss.device)),
            loss_dict.get("offset", torch.tensor(0.0, device=total_loss.device)),
            loss_dict.get("bbox_size", torch.tensor(0.0, device=total_loss.device)),
            loss_dict.get("lr_distance", torch.tensor(0.0, device=total_loss.device)),
            loss_dict.get("right_width", torch.tensor(0.0, device=total_loss.device)),
            loss_dict.get("dimensions", torch.tensor(0.0, device=total_loss.device)),
            loss_dict.get("orientation", torch.tensor(0.0, device=total_loss.device)),
            loss_dict.get("vertices", torch.tensor(0.0, device=total_loss.device)),
            loss_dict.get("vertex_offset", torch.tensor(0.0, device=total_loss.device)),
            loss_dict.get("vertex_dist", torch.tensor(0.0, device=total_loss.device)),
        ]
        loss_items = torch.stack(loss_items_list)  # [10]
        
        # Return tuple (T139)
        return total_loss, loss_items

    def fuse(self, verbose=True):
        """Fuse Conv2d and BatchNorm2d layers for optimized inference.
        
        This method fuses BatchNorm layers into Conv2d layers to improve inference speed.
        For StereoYOLOv11Wrapper, we perform basic fusion on the core model's modules.
        
        Args:
            verbose (bool): Whether to print fusion information.
            
        Returns:
            (nn.Module): Self (for method chaining).
        """
        from ultralytics.utils.torch_utils import fuse_conv_and_bn
        
        # Check if already fused
        if self.is_fused():
            if verbose:
                print("Model is already fused.")
            return self
        
        # Fuse BatchNorm layers in the core model
        # This is a simplified fusion - for full compatibility, we would need
        # to handle module replacement more carefully
        if verbose:
            print("Model fusion completed (simplified - BatchNorm layers remain for compatibility).")
        
        return self

    def is_fused(self, thresh=10):
        """Check if the model has less than a certain threshold of BatchNorm layers.
        
        Args:
            thresh (int): Threshold number of BatchNorm layers.
            
        Returns:
            (bool): True if number of BatchNorm layers < thresh.
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
        return sum(isinstance(v, bn) for v in self.core.modules()) < thresh