# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLOA v2 modules: bbox-mask renderer, heatmap bias fusion, optional SegBranch.

Soft-hint fusion: a 1-channel mask is turned into a bounded per-pixel bias added
(broadcast over channels) to PAN features before the Detect head. PAN feature
addition keeps the Detect head unmodified, lets reg and cls both see the bias
(empirical question — see spec §2), and is bounded vs the previous multiplicative
amplifier that forced detections.

See docs_yoloa_v2/specs/2026-06-02-softhint-fusion-design.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("BboxMaskRenderer", "HeatmapBiasFusion", "SegBranch", "binary_seg_loss")


class BboxMaskRenderer(nn.Module):
    """Render normalized YOLO-format bboxes into a 1xHxW mask.

    Two modes:
      - "rect":  hard rectangle (inside bbox = 1, outside = 0)
      - "gauss": per-bbox 2D Gaussian centered at bbox center,
                 sigma_x = w * sigma_factor, sigma_y = h * sigma_factor;
                 multiple bboxes in the same image are combined with max.

    Output spatial size is fixed at construction (default 80 to match P3).
    """

    def __init__(self, mask_size: int = 80, mode: str = "rect", sigma_factor: float = 0.25):
        super().__init__()
        assert mode in ("rect", "gauss"), f"mode must be 'rect' or 'gauss', got {mode!r}"
        self.mask_size = int(mask_size)
        self.mode = mode
        self.sigma_factor = float(sigma_factor)
        ys, xs = torch.meshgrid(
            torch.arange(self.mask_size, dtype=torch.float32),
            torch.arange(self.mask_size, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("grid_x", xs + 0.5, persistent=False)
        self.register_buffer("grid_y", ys + 0.5, persistent=False)

    def forward(self, bboxes: torch.Tensor, batch_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
        H = self.mask_size
        device = self.grid_x.device
        dtype = self.grid_x.dtype
        mask = torch.zeros(batch_size, 1, H, H, device=device, dtype=dtype)
        if bboxes.numel() == 0:
            return mask

        bboxes = bboxes.to(device=device, dtype=dtype)
        batch_idx = batch_idx.to(device=device, dtype=torch.long)

        cx = bboxes[:, 0] * H
        cy = bboxes[:, 1] * H
        w = bboxes[:, 2] * H
        h = bboxes[:, 3] * H

        if self.mode == "rect":
            x1 = (cx - w / 2)[:, None, None]
            x2 = (cx + w / 2)[:, None, None]
            y1 = (cy - h / 2)[:, None, None]
            y2 = (cy + h / 2)[:, None, None]
            inside = (
                (self.grid_x[None] >= x1)
                & (self.grid_x[None] < x2)
                & (self.grid_y[None] >= y1)
                & (self.grid_y[None] < y2)
            ).to(dtype)
        else:  # gauss
            sigma_x = (w * self.sigma_factor).clamp(min=0.5)
            sigma_y = (h * self.sigma_factor).clamp(min=0.5)
            dx = self.grid_x[None] - cx[:, None, None]
            dy = self.grid_y[None] - cy[:, None, None]
            inside = torch.exp(
                -(dx**2 / (2 * sigma_x[:, None, None] ** 2) + dy**2 / (2 * sigma_y[:, None, None] ** 2))
            )

        for b in range(batch_size):
            sel = batch_idx == b
            if sel.any():
                mask[b, 0] = inside[sel].max(dim=0).values
        return mask

    def extra_repr(self) -> str:
        return f"mask_size={self.mask_size}, mode={self.mode!r}, sigma_factor={self.sigma_factor}"


class HeatmapBiasFusion(nn.Module):
    """Soft-hint fusion: 1-ch mask -> bounded per-pixel bias broadcast onto PAN features.

    Output shape ``(B, 1, H, W)`` — the caller broadcasts (adds) it to a PAN feature
    of shape ``(B, C, H, W)``. The conv stack is SHARED across PAN scales; the caller
    is responsible for resizing the mask to each scale before calling forward.

    Per-scale magnitude is controlled by ``beta[i]``, initialized to zero so training
    starts as pure passthrough (vanilla YOLO). Without a hard cap, beta can in
    principle grow large; that is intentional — the detection loss decides how much
    to lean on the heatmap.

    Output per pixel is in ``[-beta_i, +beta_i]`` via tanh.
    """

    def __init__(self, num_scales: int = 3, c_mid: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, c_mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_mid, 1, 3, padding=1),
        )
        self.beta = nn.Parameter(torch.zeros(num_scales))

    def forward(self, mask: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Return bias (B, 1, H, W) for the given PAN scale.

        Args:
            mask: (B, 1, H, W) already resized to the target PAN scale.
            scale_idx: index into ``self.beta``.

        Returns:
            Bias tensor (B, 1, H, W) in ``[-beta_i, +beta_i]``.
        """
        return self.beta[scale_idx] * torch.tanh(self.conv(mask))


class SegBranch(nn.Module):
    """Lightweight semantic-segmentation head that predicts a 1-channel anomaly heatmap.

    Consumes the P3 and P4 PAN features and emits per-pixel logits at P3 resolution
    (e.g. 80x80 for 640 input). A P4 auxiliary head provides deep supervision during
    training.
    """

    def __init__(self, ch: tuple, nc: int = 1, c_mid: int | None = None):
        super().__init__()
        self.nc = nc
        c_mid = ch[0] if c_mid is None else c_mid
        self.classifier = nn.Sequential(Conv(ch[0], c_mid, 3), nn.Conv2d(c_mid, nc, 1))
        self.aux_head = nn.Sequential(Conv(ch[1], c_mid, 3), nn.Conv2d(c_mid, nc, 1)) if len(ch) > 1 else None

    def forward(self, x: list[torch.Tensor]):
        logits = self.classifier(x[0])
        if self.training and self.aux_head is not None:
            return logits, self.aux_head(x[1])
        return logits


def binary_seg_loss(
    logits: torch.Tensor, target: torch.Tensor, aux_logits: torch.Tensor | None = None, aux_weight: float = 0.4
) -> torch.Tensor:
    """BCE + soft-Dice loss for a single-channel anomaly heatmap."""
    if target.shape[2:] != logits.shape[2:]:
        target = F.interpolate(target, size=logits.shape[2:], mode="nearest")
    target = target.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = logits.sigmoid()
    inter = (prob * target).sum(dim=(1, 2, 3))
    card = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (1.0 - (2.0 * inter + 1.0) / (card + 1.0)).mean()
    loss = bce + dice
    if aux_logits is not None:
        aux_t = target
        if aux_t.shape[2:] != aux_logits.shape[2:]:
            aux_t = F.interpolate(target, size=aux_logits.shape[2:], mode="nearest")
        loss = loss + aux_weight * F.binary_cross_entropy_with_logits(aux_logits, aux_t)
    return loss


if __name__ == "__main__":
    # Smoke test: render, fusion init=0 passthrough, fusion with learned beta bounded,
    # broadcast onto a C-channel PAN feature.
    B, H = 4, 80
    renderer = BboxMaskRenderer(mask_size=H, mode="rect")
    bboxes = torch.tensor(
        [
            [0.25, 0.25, 0.20, 0.20],
            [0.75, 0.75, 0.30, 0.10],
            [0.50, 0.50, 0.10, 0.10],
        ]
    )
    batch_idx = torch.tensor([0, 0, 2], dtype=torch.long)
    mask = renderer(bboxes, batch_idx, B)
    assert mask.shape == (B, 1, H, H)
    assert mask[1].sum().item() == 0.0  # image with no bboxes

    fusion = HeatmapBiasFusion(num_scales=3)
    # beta init=0 -> bias is exactly zero for every scale
    for s in range(3):
        bias = fusion(mask, s)
        assert bias.shape == (B, 1, H, H)
        assert bias.abs().max().item() == 0.0, f"scale {s} bias not zero at init"
    print("HeatmapBiasFusion init=0 passthrough OK.")

    # With beta set non-zero, output is bounded in [-beta, +beta]
    with torch.no_grad():
        fusion.beta.fill_(1.5)
    bias = fusion(mask, 0)
    assert bias.abs().max().item() <= 1.5 + 1e-6, "bias exceeded beta after tanh"
    print(f"HeatmapBiasFusion beta=1.5 bounded OK (max abs = {bias.abs().max().item():.4f}).")

    # Broadcast onto a C-channel PAN feature: (B, 1, H, W) + (B, C, H, W) -> (B, C, H, W).
    p = torch.randn(B, 256, H, H)
    p_fused = p + fusion(mask, 0)
    assert p_fused.shape == p.shape
    print(f"Broadcast OK: P (B,256,H,W) + bias (B,1,H,W) -> {tuple(p_fused.shape)}.")

    # Resize-per-scale smoke
    mask_p4 = F.interpolate(mask, size=(40, 40), mode="bilinear", align_corners=False)
    mask_p5 = F.interpolate(mask, size=(20, 20), mode="bilinear", align_corners=False)
    assert fusion(mask_p4, 1).shape == (B, 1, 40, 40)
    assert fusion(mask_p5, 2).shape == (B, 1, 20, 20)
    print("HeatmapBiasFusion multi-scale OK.")

    print("\nAll smoke tests passed.")
