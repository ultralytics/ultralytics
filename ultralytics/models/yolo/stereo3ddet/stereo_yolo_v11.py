"""
YOLOv11 Stereo 3D Detection - Complete Implementation
基于Stereo CenterNet的立体3D检测网络
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR


# ============================================================================
# 第一部分: 特征融合层
# ============================================================================

class StereoFeatureFusion(nn.Module):
    """立体特征融合模块"""

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
# 第二部分: Neck (FPN-style特征金字塔)
# ============================================================================

class StereoPAN(nn.Module):
    """立体路径聚合网络 (Path Aggregation Network)"""

    def __init__(self, in_channels: int = 256, out_channels: int = 256):
        super().__init__()

        # 自顶向下 (Top-Down)
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

        # 自底向上 (Bottom-Up)
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
            features: [P3, P4, P5] 来自Backbone的多尺度特征
        Returns:
            [P3_out, P4_out, P5_out]
        """
        # 简化版: 只处理P3层 (主检测层)
        # 实际应用中应处理多层
        return features  # 占位符


# ============================================================================
# 第三部分: 检测Head (10个分支)
# ============================================================================

class StereoCenterNetHead(nn.Module):
    """立体CenterNet检测头 - 10个并行分支"""

    def __init__(self, in_channels: int = 256, num_classes: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 共享特征提取 (可选)
        self.shared_head = self._build_shared_head(in_channels)

        # 10个分支定义
        self.branches = nn.ModuleDict(
            {
                # Task A: 立体2D检测 (5个分支)
                "heatmap": self._build_branch(in_channels, num_classes),
                "offset": self._build_branch(in_channels, 2),
                "bbox_size": self._build_branch(in_channels, 2),
                "lr_distance": self._build_branch(in_channels, 1),
                "right_width": self._build_branch(in_channels, 1),
                # Task B: 3D组件 (5个分支)
                "dimensions": self._build_branch(in_channels, 3),
                "orientation": self._build_branch(in_channels, 8),
                "vertices": self._build_branch(in_channels, 8),
                "vertex_offset": self._build_branch(in_channels, 8),
                "vertex_dist": self._build_branch(in_channels, 4),
            }
        )

    def _build_shared_head(self, in_channels: int) -> nn.Sequential:
        """共享的特征提取头"""
        return nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def _build_branch(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """构建单个分支: Conv(3×3) → BN → ReLU → Conv(1×1)"""
        return nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 1, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] 来自Neck的融合特征
        Returns:
            Dict of 10 branch outputs
        """
        # 共享特征提取
        shared_feat = self.shared_head(x)  # [B, 256, H, W]

        # 并行运行10个分支
        outputs = {}
        for branch_name, branch_module in self.branches.items():
            outputs[branch_name] = branch_module(shared_feat)

        return outputs


# ============================================================================
# 第四部分: Loss函数
# ============================================================================

class FocalLoss(nn.Module):
    """焦点损失 (用于热图) - 解决类不平衡"""

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        焦点损失
        L = -1/N * Σ [
            (1-ŷ)^α * log(ŷ)       if y=1
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
    """立体CenterNet的总Loss"""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.focal_loss_fn = FocalLoss(alpha=2.0, beta=4.0)
        self.num_classes = num_classes

        # 各分支权重
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
        计算各分支的Loss

        Args:
            predictions: Dict of network outputs
            targets: Dict of ground truth targets
        Returns:
            (total_loss, loss_dict)
        """
        losses = {}

        # ===== Task A: Stereo 2D Detection =====

        # 1. 热图 (Focal Loss)
        losses["heatmap"] = self.focal_loss_fn(predictions["heatmap"], targets["heatmap"])

        # 2. 中心点偏移 (L1, 仅在热图点处)
        losses["offset"] = self.masked_l1_loss(
            predictions["offset"], targets["offset"], mask=targets.get("heatmap", None)
        )

        # 3. 2D框尺寸 (Smooth L1)
        losses["bbox_size"] = self.masked_l1_loss(
            predictions["bbox_size"], targets["bbox_size"], mask=targets.get("heatmap", None)
        )

        # 4. 左右中心距离 (L1)
        losses["lr_distance"] = self.masked_l1_loss(
            predictions["lr_distance"], targets["lr_distance"], mask=targets.get("heatmap", None)
        )

        # 5. 右框宽度 (特殊处理)
        losses["right_width"] = self.right_width_loss(
            predictions["right_width"], targets["right_width"], mask=targets.get("heatmap", None)
        )

        # ===== Task B: 3D Components =====

        # 6. 3D尺寸 (L1)
        losses["dimensions"] = self.masked_l1_loss(
            predictions["dimensions"], targets["dimensions"], mask=targets.get("heatmap", None)
        )

        # 7. 方向角 (Multi-Bin)
        losses["orientation"] = self.orientation_loss(
            predictions["orientation"], targets["orientation"], mask=targets.get("heatmap", None)
        )

        # 8. 顶点坐标 (L1)
        losses["vertices"] = self.masked_l1_loss(
            predictions["vertices"], targets["vertices"], mask=targets.get("heatmap", None)
        )

        # 9. 顶点亚像素偏移 (L1)
        losses["vertex_offset"] = self.masked_l1_loss(
            predictions["vertex_offset"], targets["vertex_offset"], mask=targets.get("heatmap", None)
        )

        # 10. 顶点距离 (L1)
        losses["vertex_dist"] = self.masked_l1_loss(
            predictions["vertex_dist"], targets["vertex_dist"], mask=targets.get("heatmap", None)
        )

        # 加权求和
        total_loss = sum(self.loss_weights[k] * v for k, v in losses.items())

        return total_loss, losses

    def masked_l1_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """带mask的Smooth L1 Loss"""
        loss = F.smooth_l1_loss(pred, target, reduction="none")

        if mask is not None:
            # 扩展mask到pred的维度
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
        """右框宽度的特殊Loss (with sigmoid变换)"""
        # 应用sigmoid变换: wr = 1/σ(ŵr) - 1
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
        方向角Loss (Multi-Bin编码)

        pred: [B, 8, H, W]
          结构: [bin_logit_1, bin_logit_2, sin_1, cos_1, sin_2, cos_2, pad, pad]
        target: [B, 8, H, W]
          结构: [bin_id, bin_id, sin, cos, ...]
        """
        # 分离bin分类和角度回归
        bin_pred = pred[:, :2, :, :]  # [B, 2, H, W]
        angle_pred = pred[:, 2:6, :, :]  # [B, 4, H, W]

        bin_target = target[:, 0:1, :, :].long().squeeze(1)  # [B, H, W]
        angle_target = target[:, 2:6, :, :]  # [B, 4, H, W]

        # Bin分类 Loss
        bin_loss = F.cross_entropy(bin_pred, bin_target, reduction="none")  # [B, H, W]

        # 角度回归 Loss (仅sin/cos, 共4维)
        angle_loss = F.l1_loss(angle_pred, angle_target, reduction="none")
        angle_loss = angle_loss.mean(dim=1)  # [B, H, W]

        # 总Loss
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
    """不确定性加权的多任务Loss (Kendall et al. 2018)"""

    def __init__(self, num_tasks: int = 10):
        super().__init__()
        self.log_vars = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_tasks)])

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        L_total = Σ_i (1/(2*σ_i²) * L_i + log(σ_i))

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
# 第五部分: 完整模型
# ============================================================================

class StereoYOLOv11(nn.Module):
    """YOLOv11 Stereo 3D Detection - 完整模型"""

    def __init__(self, backbone_type: str = "resnet18", num_classes: int = 3, in_channels: int = 256):
        super().__init__()

        self.backbone_type = backbone_type
        self.num_classes = num_classes

        # 1. Backbone (共享)
        self.backbone = self._build_backbone(backbone_type)
        backbone_out_channels = self._get_backbone_channels(backbone_type)

        # 2. 特征融合
        self.fusion = StereoFeatureFusion(in_channels=backbone_out_channels, out_channels=in_channels)

        # 3. Neck (特征金字塔)
        self.neck = StereoPAN(in_channels=in_channels, out_channels=in_channels)

        # 4. Detection Heads
        self.heads = StereoCenterNetHead(in_channels=in_channels, num_classes=num_classes)

        # 5. Loss函数
        self.criterion = StereoCenterNetLoss(num_classes=num_classes)
        self.uncertainty_loss = UncertaintyWeightedLoss(num_tasks=10)

    def _build_backbone(self, backbone_type: str) -> nn.Module:
        """构建骨干网络"""
        if backbone_type == "resnet18":
            from torchvision import models

            backbone = models.resnet18(pretrained=True)
            # 去掉最后的FC层
            return nn.Sequential(*list(backbone.children())[:-2])

        elif backbone_type == "resnet50":
            from torchvision import models

            backbone = models.resnet50(pretrained=True)
            return nn.Sequential(*list(backbone.children())[:-2])

        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

    def _get_backbone_channels(self, backbone_type: str) -> int:
        """获取Backbone输出通道数"""
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
        完整前向传播

        Args:
            left_img: [B, 3, H, W]
            right_img: [B, 3, H, W]
            targets: Dict of ground truth (训练时)

        Returns:
            (predictions, loss) 或 predictions
        """
        # 1. Backbone特征提取
        left_feat = self.backbone(left_img)  # [B, C_backbone, H/32, W/32]
        right_feat = self.backbone(right_img)  # [B, C_backbone, H/32, W/32]

        # 2. 特征融合
        fused_feat = self.fusion(left_feat, right_feat)  # [B, 256, H/32, W/32]

        # 3. Neck处理 (简化版: 这里只处理一层)
        # 实际应该处理多尺度
        neck_out = [fused_feat, fused_feat, fused_feat]

        # 4. Detection Heads (使用P3输出, 对应4倍下采样)
        predictions = self.heads(neck_out[0])

        # 5. 计算Loss (训练时)
        if targets is not None:
            # 各分支单独Loss
            total_loss, loss_dict = self.criterion(predictions, targets)

            # (可选) 使用不确定性加权
            # total_loss = self.uncertainty_loss(loss_dict)

            return predictions, total_loss, loss_dict
        else:
            return predictions


# ============================================================================
# 第六部分: 训练工具
# ============================================================================

class Trainer:
    """训练器"""

    def __init__(self, model: StereoYOLOv11, device: str = "cuda", learning_rate: float = 1.5e-4):
        self.model = model.to(device)
        self.device = device

        # 优化器 (使用AdamW)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # 学习率调度
        self.scheduler = MultiStepLR(self.optimizer, milestones=[40], gamma=0.1)  # 第40 epoch降10倍

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """训练一个epoch"""
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

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=35.0)

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k not in loss_stats:
                    loss_stats[k] = 0
                loss_stats[k] += v.item()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

        # 平均Loss
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        for k in loss_stats:
            loss_stats[k] /= num_batches

        return {"total": avg_loss, **loss_stats}

    def step_scheduler(self):
        """更新学习率"""
        self.scheduler.step()


if __name__ == "__main__":
    # 创建模型
    model = StereoYOLOv11(backbone_type="resnet18", num_classes=3, in_channels=256)

    # 创建虚拟输入
    left_img = torch.randn(2, 3, 384, 1280)
    right_img = torch.randn(2, 3, 384, 1280)

    # 创建虚拟targets (无target时仅推理)
    targets = None

    # 前向传播
    if targets is None:
        predictions = model(left_img, right_img)

        # 查看输出
        for branch_name, output in predictions.items():
            print(f"{branch_name}: {output.shape}")
    else:
        predictions, loss, loss_dict = model(left_img, right_img, targets)
        print(f"Total Loss: {loss.item():.4f}")
        for k, v in loss_dict.items():
            print(f"  {k}: {v.item():.4f}")
