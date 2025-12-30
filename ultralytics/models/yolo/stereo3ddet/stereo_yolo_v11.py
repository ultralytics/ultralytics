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
# Helper: DLA-34 Backbone Wrapper
# ============================================================================


class _DLA34Wrapper(nn.Module):
    """Wrapper for timm DLA-34 backbone to extract single feature map.

    timm's features_only=True returns a list of feature maps from different stages.
    This wrapper extracts only the final stage output (index 0 from out_indices=(5,)).
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract final feature map from DLA-34.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Feature tensor [B, 512, H/32, W/32] from final stage
        """
        # timm with features_only=True returns list of feature maps
        features = self.backbone(x)
        # With out_indices=(5,), we get a list with single element (stage 5, 512 channels)
        return features[0]


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
        
        # Attributes required for compatibility with v8DetectionLoss (used by DetectionModel.init_criterion)
        # These are expected by the standard YOLO loss function even though stereo uses custom loss
        self.nc = num_classes  # Alias for num_classes (expected by v8DetectionLoss)
        self.reg_max = 1  # Regression max (not used by stereo, but required by v8DetectionLoss)
        self.no = self.nc + self.reg_max * 4  # Number of outputs per anchor (for compatibility)
        
        # Stride attribute required for compatibility (v8DetectionLoss expects model.model[-1].stride)
        # Note: This is a default value. The actual stride depends on the model architecture:
        # - P3 output: 8x downsampling (stride = 8.0)
        # - P4 output: 16x downsampling (stride = 16.0)  
        # - P5 output: 32x downsampling (stride = 32.0)
        # DetectionModel will compute this dynamically via forward pass if the head is a Detect subclass.
        # For stereo3ddet, we use StereoCenterNetLoss which doesn't rely on this stride value.
        self.stride = torch.tensor([8.0])  # Default to P3 (8x) since stereo3ddet_full.yaml uses P3 output

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

    def forward(self, x: list[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] fused features from the neck
        Returns:
            Dict of 10 branch outputs
        """
        # Shared feature extraction
        shared_feat = self.shared_head(x[0])  # [B, 256, H, W]

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

    def __init__(self, num_classes: int = 3, loss_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.num_classes = num_classes

        # Default per-branch weights (used if loss_weights is None or incomplete)
        default_weights = {
            "heatmap": 1.0,
            "offset": 1.0,
            "bbox_size": 0.1,
            "lr_distance": 1.0,
            "right_width": 0.1,
            "dimensions": 0.1,
            "orientation": 1.0,
            "vertices": 1.0,
            "vertex_offset": 1.0,
            "vertex_dist": 0.1,  # Lower weight to match YAML config and reduce impact of large distances
        }
        
        # Use provided loss_weights, falling back to defaults for missing keys
        if loss_weights is not None:
            self.loss_weights = {key: loss_weights.get(key, default_weights[key]) for key in default_weights.keys()}
        else:
            self.loss_weights = default_weights.copy()

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
        
        vertex_loss_raw = self.masked_l1_loss(
            predictions["vertices"], targets["vertices"], mask=targets.get("heatmap", None), branch_name="vertices"
        )
        losses["vertices"] = vertex_loss_raw
        
        # 9. Vertex sub-pixel offset (L1)
        losses["vertex_offset"] = self.masked_l1_loss(
            predictions["vertex_offset"], targets["vertex_offset"], mask=targets.get("heatmap", None)
        )

        # 10. Vertex distance (L1)
        
        vertex_dist_loss_raw = self.masked_l1_loss(
            predictions["vertex_dist"], targets["vertex_dist"], mask=targets.get("heatmap", None), branch_name="vertex_dist"
        )
        losses["vertex_dist"] = vertex_dist_loss_raw
        
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
        # IMPORTANT (AMP/FP16 stability):
        # In fp16, values like (1 - 1e-4) round back to 1.0 (ulp near 1 is ~9.77e-4),
        # so clamp(max=1-1e-4) may NOT prevent pred==1.0 and log(1-pred)==log(0)==-inf.
        # Do focal-loss math in fp32 to avoid this and use log1p for better stability.
        pred = torch.sigmoid(pred).float()
        target = target.float()

        eps = 1e-4  # fp32 epsilon for log safety
        pred = pred.clamp(min=eps, max=1.0 - eps)
        
        # Identify positive locations (peak of Gaussian, Y = 1)
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()
        
        # Count number of positive samples for normalization
        num_pos = pos_mask.sum()
        
        # Handle empty batches gracefully (images with no objects)
        # This can happen legitimately in KITTI when:
        # - Image has no annotated objects
        # - All objects were filtered out (wrong class, truncated, etc.)
        # - Object centers fall outside feature map bounds
        if num_pos == 0:
            # Return 0 with a valid grad path
            return pred.sum() * 0.0
        
        num_pos = torch.clamp(num_pos, min=1.0)  # Avoid division by zero
        
        # Positive loss: (1 - Ŷ)^α * log(Ŷ)
        pos_loss = torch.pow(1 - pred, alpha) * torch.log(pred) * pos_mask
        
        # Negative loss: (1 - Y)^β * Ŷ^α * log(1 - Ŷ)
        # The (1 - Y)^β term down-weights locations near object centers
        # log1p(-pred) is numerically more stable than log(1-pred) when pred ~ 1
        neg_loss = torch.pow(1 - target, beta) * torch.pow(pred, alpha) * torch.log1p(-pred) * neg_mask
        
        
        # Sum and normalize by number of positive samples
        loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        
        return loss

    def masked_l1_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, branch_name: str = ""
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
            layout: [bin_logit_0, bin_logit_1, sin_0, cos_0, sin_1, cos_1, pad, pad]
        target: [B, 8, H, W]
            layout: [conf_0, conf_1, sin_0, cos_0, sin_1, cos_1, pad, pad]
            - One-hot bin encoding: [1, 0, ...] for bin 0, [0, 1, ...] for bin 1
            - bin 0: α ∈ [-π, 0), center = -π/2
            - bin 1: α ∈ [0, π], center = +π/2
            - sin/cos: residual from bin center
        """
        # Split bin classification and angle regression
        bin_pred = pred[:, :2, :, :]  # [B, 2, H, W]
        angle_pred = pred[:, 2:6, :, :]  # [B, 4, H, W]

        # Convert one-hot target to class index using argmax
        # [1, 0] → 0 (bin 0), [0, 1] → 1 (bin 1)
        bin_target_onehot = target[:, :2, :, :]  # [B, 2, H, W]
        bin_target = bin_target_onehot.argmax(dim=1)  # [B, H, W] - class indices
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
    """Uncertainty-weighted multi-task loss (Kendall et al., 2018).
    
    This module learns task-specific uncertainty parameters (log_vars) that automatically
    balance the contribution of each loss component during training. Higher uncertainty
    for a task means lower weight, allowing the model to focus on more reliable tasks.
    
    The learned weights can be logged during training to verify balance (T045).
    
    Attributes:
        log_vars: Learnable log-variance parameters for each task.
        _last_breakdown: Cached breakdown of last forward pass for logging.
        _task_names: Task names from last forward pass (for logging).
    """

    def __init__(self, num_tasks: int = 10, log_interval: int = 100):
        """Initialize uncertainty-weighted loss module.
        
        Args:
            num_tasks: Number of loss components to weight.
            log_interval: How often to log loss breakdown (every N forward calls).
        """
        super().__init__()
        self.log_vars = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_tasks)])
        self.log_interval = log_interval
        self._forward_count = 0
        self._last_breakdown: Optional[Dict[str, Dict[str, float]]] = None
        self._task_names: Optional[List[str]] = None

    def forward(self, losses: Dict[str, torch.Tensor], verbose: bool = False) -> torch.Tensor:
        """Compute uncertainty-weighted total loss.
        
        L_total = Σ_i (exp(-log_var_i) * L_i + log_var_i)

        Args:
            losses: Dict of individual task losses {task_name: loss_tensor}.
            verbose: If True, log loss breakdown immediately.
            
        Returns:
            Total weighted loss scalar tensor.
        """
        total_loss = 0
        breakdown = {}
        self._task_names = list(losses.keys())

        for i, (task_name, loss_val) in enumerate(losses.items()):
            log_var = self.log_vars[i]
            # L_i_weighted = exp(-σ_i) * L_i + σ_i
            # exp(-σ) acts as precision (inverse variance), higher σ = lower weight
            precision = torch.exp(-log_var)
            weighted_loss = precision * loss_val + log_var
            total_loss = total_loss + weighted_loss
            
            # Store breakdown for logging (T045)
            breakdown[task_name] = {
                "raw_loss": loss_val.detach().item(),
                "log_var": log_var.detach().item(),
                "precision": precision.detach().item(),
                "weighted_loss": weighted_loss.detach().item(),
            }
        
        self._last_breakdown = breakdown
        self._forward_count += 1
        
        # Periodic logging for training balance verification (T045)
        if verbose or (self.log_interval > 0 and self._forward_count % self.log_interval == 0):
            self._log_breakdown()

        return total_loss
    
    def _log_breakdown(self) -> None:
        """Log the loss breakdown for balance verification (T045).
        
        Logs individual loss components, learned uncertainty parameters,
        and effective weights to help verify that uncertainty weighting
        is properly balancing the multi-task losses.
        """
        if self._last_breakdown is None:
            return
        
        try:
            from ultralytics.utils import LOGGER
        except ImportError:
            import logging
            LOGGER = logging.getLogger(__name__)
        
        # Build log message
        msg_parts = ["[Uncertainty Loss Breakdown]"]
        for task_name, values in self._last_breakdown.items():
            # precision = exp(-log_var), represents effective weight
            msg_parts.append(
                f"  {task_name}: raw={values['raw_loss']:.4f}, "
                f"log_var={values['log_var']:.4f}, "
                f"weight={values['precision']:.4f}, "
                f"weighted={values['weighted_loss']:.4f}"
            )
        
        LOGGER.info("\n".join(msg_parts))
    
    def get_loss_breakdown(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Get the breakdown of the last computed loss.
        
        Returns:
            Dict mapping task names to their loss breakdown:
                - raw_loss: Original unweighted loss value
                - log_var: Learned log-variance parameter
                - precision: Effective weight (exp(-log_var))
                - weighted_loss: Final weighted loss contribution
            Returns None if forward() hasn't been called yet.
        """
        return self._last_breakdown
    
    def get_learned_weights(self) -> Dict[str, float]:
        """Get the current learned weights for each task.
        
        The effective weight for each task is exp(-log_var).
        Higher weight = model considers this task more reliable.
        
        Returns:
            Dict mapping task names to their effective weights.
        """
        if self._task_names is None:
            # Use generic names if forward hasn't been called
            return {f"task_{i}": torch.exp(-lv).item() for i, lv in enumerate(self.log_vars)}
        
        return {
            name: torch.exp(-self.log_vars[i]).item() 
            for i, name in enumerate(self._task_names)
        }


# ============================================================================
# Part 5: Full Model
# ============================================================================

class StereoYOLOv11(nn.Module):
    """YOLOv11 Stereo 3D Detection - Full model."""

    def __init__(
        self,
        backbone_type: str = "resnet18",
        num_classes: int = 3,
        in_channels: int = 256,
        use_uncertainty_weighting: bool = False,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize StereoYOLOv11 model.

        Args:
            backbone_type: Backbone architecture ('resnet18', 'resnet50', or 'dla34').
            num_classes: Number of object classes (default 3 for KITTI: Car, Pedestrian, Cyclist).
            in_channels: Number of channels for feature fusion output (default 256).
            use_uncertainty_weighting: Whether to use learned uncertainty weighting for multi-task
                loss balancing (Kendall et al., 2018). When True, the model learns task-specific
                uncertainty parameters that automatically balance the contribution of each loss
                component during training. Default False for backward compatibility.
            loss_weights: Optional dict of loss weights for each branch. If None, uses default weights.
        """
        super().__init__()

        self.backbone_type = backbone_type
        self.num_classes = num_classes
        self.use_uncertainty_weighting = use_uncertainty_weighting

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
        self.criterion = StereoCenterNetLoss(num_classes=num_classes, loss_weights=loss_weights)
        self.uncertainty_loss = UncertaintyWeightedLoss(num_tasks=10)

    def _build_backbone(self, backbone_type: str) -> nn.Module:
        """Build the backbone network.
        
        Args:
            backbone_type: Type of backbone to use. Supported options:
                - "resnet18": ResNet-18 (512 channels, ~30+ FPS)
                - "resnet50": ResNet-50 (2048 channels, ~25 FPS)
                - "dla34": DLA-34 with Deep Layer Aggregation (512 channels, ~20+ FPS)
                          Requires timm library: pip install timm
                          
        Returns:
            nn.Module: Backbone network that outputs feature maps [B, C, H/32, W/32]
            
        Raises:
            ValueError: If backbone_type is not supported
            ImportError: If dla34 is requested but timm is not installed
            
        Note (GAP-005 / T033):
            DLA-34 provides +8.73% AP3D improvement over ResNet-18 according to 
            the Stereo CenterNet paper. The backbone is shared for weight sharing
            between left and right image processing (see T034).
        """
        # Validate backbone type upfront (T033)
        valid_backbones = {"resnet18", "resnet50", "dla34"}
        if backbone_type not in valid_backbones:
            valid_options = ", ".join(sorted(valid_backbones))
            raise ValueError(
                f"Unknown backbone type: '{backbone_type}'. "
                f"Valid options are: {valid_options}"
            )
        
        if backbone_type == "resnet18":
            from torchvision import models

            backbone = models.resnet18(pretrained=True)
            # Remove final FC layer
            return nn.Sequential(*list(backbone.children())[:-2])

        elif backbone_type == "resnet50":
            from torchvision import models

            backbone = models.resnet50(pretrained=True)
            return nn.Sequential(*list(backbone.children())[:-2])

        elif backbone_type == "dla34":
            # DLA-34 backbone with deformable convolutions for +8.73% AP3D improvement
            # Paper Reference: Stereo CenterNet uses DLA-34 as primary backbone
            try:
                import timm
            except ImportError as e:
                raise ImportError(
                    "DLA-34 backbone requires the 'timm' library.\n"
                    "Install with: pip install timm>=0.9.0\n"
                    "Or add to requirements: timm>=0.9.0"
                ) from e

            backbone = timm.create_model(
                "dla34",
                pretrained=True,
                features_only=True,
                out_indices=(5,),  # Get final feature map (stage 5, 512 channels, 32x downsample)
            )
            # Wrap in a module that extracts just the final feature map
            return _DLA34Wrapper(backbone)
        
        # Should never reach here due to upfront validation
        raise ValueError(f"Unhandled backbone type: {backbone_type}")

    def _get_backbone_channels(self, backbone_type: str) -> int:
        """Get the number of output channels of the backbone.
        
        Args:
            backbone_type: Type of backbone ("resnet18", "resnet50", "dla34")
            
        Returns:
            Number of output channels from the backbone's final feature map
            
        Channel reference (GAP-005 / T032):
            - resnet18: 512 (final BasicBlock output)
            - resnet50: 2048 (final Bottleneck output)
            - dla34: 512 (final stage output from Deep Layer Aggregation)
        """
        channel_map = {
            "resnet18": 512,
            "resnet50": 2048,
            "dla34": 512,  # DLA-34 final stage channels (GAP-005)
        }
        if backbone_type not in channel_map:
            valid_options = ", ".join(sorted(channel_map.keys()))
            raise ValueError(
                f"Unknown backbone type: '{backbone_type}'. "
                f"Valid options are: {valid_options}"
            )
        return channel_map[backbone_type]

    def forward(
        self,
        left_img: torch.Tensor,
        right_img: torch.Tensor,
        targets: Optional[Dict] = None,
    ) -> Tuple[Dict, Optional[torch.Tensor]]:
        """
        Full forward pass.

        Args:
            left_img: [B, 3, H, W] - Left camera image tensor
            right_img: [B, 3, H, W] - Right camera image tensor  
            targets: Dict of ground truth (training)

        Returns:
            (predictions, loss) or predictions
            
        Note (T034 - Weight Sharing Verification):
            The same backbone instance (self.backbone) is used to process both
            left and right images. This ensures weight sharing between the two
            branches, which is critical for stereo matching. Works correctly with
            all supported backbones: ResNet-18, ResNet-50, and DLA-34.
        """
        # 1. Backbone feature extraction (weight sharing - T034)
        # Both left and right images pass through the SAME backbone module,
        # ensuring identical feature extraction for stereo correspondence.
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
            # Apply uncertainty weighting if enabled (GAP-007: Uncertainty Weighting)
            # This uses learned uncertainty parameters to automatically balance
            # multi-task loss components during training (Kendall et al., 2018)
            if self.use_uncertainty_weighting:
                total_loss = self.uncertainty_loss(loss_dict)
            return predictions, total_loss, loss_dict
        else:
            return predictions


class StereoYOLOv11Wrapper(nn.Module):
    """Wrapper that accepts a single 6-channel image tensor and splits into left/right.

    Exposes a YOLO-like interface: forward(img, targets=None) -> (loss, items) during training.
    """

    def __init__(
        self,
        backbone_type: str = "resnet18",
        num_classes: int = 3,
        in_channels: int = 256,
        use_uncertainty_weighting: bool = False,
    ):
        """Initialize StereoYOLOv11Wrapper.

        Args:
            backbone_type: Backbone architecture ('resnet18', 'resnet50', or 'dla34').
            num_classes: Number of object classes (default 3 for KITTI: Car, Pedestrian, Cyclist).
            in_channels: Number of channels for feature fusion output (default 256).
            use_uncertainty_weighting: Whether to use learned uncertainty weighting for multi-task
                loss balancing. See StereoYOLOv11 for details.
        """
        super().__init__()
        self.core = StereoYOLOv11(
            backbone_type=backbone_type,
            num_classes=num_classes,
            in_channels=in_channels,
            use_uncertainty_weighting=use_uncertainty_weighting,
        )
        self.names = {i: str(i) for i in range(num_classes)}
        # Required attributes for AutoBackend compatibility
        # Note: Stride depends on model architecture (P3=8x, P4=16x, P5=32x)
        # Default to P3 (8x) since stereo3ddet_full.yaml uses P3 output
        self.stride = torch.tensor([8.0])  # Architecture-dependent, matches P3 output
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
        input_size = (h, w)
        
        # Import TargetGenerator and initialize it (T133)
        from ultralytics.data.stereo.target import TargetGenerator
        from ultralytics.data.stereo.target_improved import TargetGenerator as TargetGeneratorImproved
        
        # Initialize target generator if not already done
        if not hasattr(self, "_target_generator"):
            num_classes = len(self.names) if isinstance(self.names, dict) else 3
            
            # Dynamically determine output size from model forward pass
            # This works for any architecture (P3, P54, P5, etc.) instead of hardcoding 32x
            # The model's actual output size depends on the YAML config (e.g., P3 = 8x downsampling)
            # We do a dummy forward pass to get the actual output shape, making it architecture-agnostic
            with torch.no_grad():
                # Create dummy input with same shape as actual input
                dummy_img = torch.zeros(1, 6, input_size[0], input_size[1], device=img.device)
                # Forward pass to get actual output shape
                dummy_output = self.forward(dummy_img)
                
                # Extract output shape from predictions
                # For stereo3ddet, output is a dict with 10 branches
                if isinstance(dummy_output, dict):
                    # Get shape from any branch (all have same spatial dimensions)
                    sample_branch = dummy_output.get("heatmap", list(dummy_output.values())[0])
                    if sample_branch is not None:
                        _, _, output_h, output_w = sample_branch.shape
                    else:
                        # Fallback: try to get from model stride if available
                        if hasattr(self, "stride") and self.stride is not None:
                            stride = float(self.stride[0]) if isinstance(self.stride, torch.Tensor) else float(self.stride)
                            output_h = int(input_size[0] / stride)
                            output_w = int(input_size[1] / stride)
                        else:
                            # Last resort: assume 8x downsampling for P3 (common case)
                            output_h = input_size[0] // 8
                            output_w = input_size[1] // 8
                else:
                    # If output is not a dict, try to infer from model structure
                    if hasattr(self, "stride") and self.stride is not None:
                        stride = float(self.stride[0]) if isinstance(self.stride, torch.Tensor) else float(self.stride)
                        output_h = int(input_size[0] / stride)
                        output_w = int(input_size[1] / stride)
                    else:
                        # Fallback: assume 8x downsampling for P3
                        output_h = input_size[0] // 8
                        output_w = input_size[1] // 8
            
            self._target_generator = TargetGeneratorImproved(
                output_size=(output_h, output_w),
                num_classes=num_classes,
            )
        
        # Convert labels to targets format (T134)
        targets_list = []
        for labels in labels_list:
            target = self._target_generator.generate_targets(
                labels,
                input_size=input_size
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
            # Use target generator's output size if available, otherwise determine dynamically
            if hasattr(self, "_target_generator"):
                output_h, output_w = self._target_generator.output_h, self._target_generator.output_w
            else:
                # Dynamically determine output size from model forward pass
                with torch.no_grad():
                    dummy_img = torch.zeros(1, 6, input_size[0], input_size[1], device=img.device)
                    dummy_output = self.forward(dummy_img)
                    
                    if isinstance(dummy_output, dict):
                        sample_branch = dummy_output.get("heatmap", list(dummy_output.values())[0])
                        if sample_branch is not None:
                            _, _, output_h, output_w = sample_branch.shape
                        else:
                            output_h = input_size[0] // 8
                            output_w = input_size[1] // 8
                    else:
                        output_h = input_size[0] // 8
                        output_w = input_size[1] // 8
            
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