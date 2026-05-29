# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLOA v2 modules: bbox-mask renderer, heatmap encoder, heatmap-guided fusion.

See docs_yoloa_v2/design.md for the Phase 0 design.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("BboxMaskRenderer", "HeatmapEncoder", "HeatmapGuidedFusion", "SegBranch", "binary_seg_loss")


class BboxMaskRenderer(nn.Module):
    """Render normalized YOLO-format bboxes into a 1xHxW mask.

    Two modes:
      - "rect":  hard rectangle (inside bbox = 1, outside = 0)
      - "gauss": per-bbox 2D Gaussian centered at bbox center,
                 sigma_x = w * sigma_factor, sigma_y = h * sigma_factor;
                 multiple bboxes in the same image are combined with max.

    Output spatial size is fixed at construction (default 80 to match P3).
    The mask is non-learnable; this module exists to live in the model
    hierarchy so coordinate grids follow .to(device).
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
        # +0.5 -> pixel-center coordinates
        self.register_buffer("grid_x", xs + 0.5, persistent=False)
        self.register_buffer("grid_y", ys + 0.5, persistent=False)

    def forward(self, bboxes: torch.Tensor, batch_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Render mask.

        Args:
            bboxes: (N, 4) normalized [cx, cy, w, h] in [0, 1].
            batch_idx: (N,) long tensor, image index in [0, batch_size).
            batch_size: B.

        Returns:
            (B, 1, H, W) float mask in [0, 1].
        """
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
            # (N, H, W) indicator
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

        # Combine per-bbox maps into per-image mask via max-reduce by batch_idx.
        # Vectorized scatter-max via one-hot would allocate (N, B); the loop
        # over B is cheap (B ~ 96) and avoids that overhead.
        for b in range(batch_size):
            sel = batch_idx == b
            if sel.any():
                mask[b, 0] = inside[sel].max(dim=0).values
        return mask

    def extra_repr(self) -> str:
        return f"mask_size={self.mask_size}, mode={self.mode!r}, sigma_factor={self.sigma_factor}"


class HeatmapEncoder(nn.Module):
    """1-channel mask -> C-channel feature, single scale.

    Two Conv-GELU layers. No BN (input is a 1-channel mask, BN over 1 channel
    is unstable and the scale of the mask is well-defined in [0, 1]).
    """

    def __init__(self, c_out: int, c_mid: int | None = None, k: int = 3):
        super().__init__()
        c_mid = c_out if c_mid is None else c_mid
        p = k // 2
        self.conv1 = nn.Conv2d(1, c_mid, k, padding=p, bias=True)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(c_mid, c_out, k, padding=p, bias=True)
        self.act2 = nn.GELU()

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.act2(self.conv2(self.act1(self.conv1(mask))))


class HeatmapGuidedFusion(nn.Module):
    """Multiplicative attention fusion: P_out = P * 2 * sigmoid(AnomalyFeat).

    Properties:
      - sigmoid(0) = 0.5 -> multiplier = 1.0 -> exact passthrough when AF == 0.
        This is required by the mask-dropout training (see design §3.4):
        when the mask is dropped, callers pass AF == 0 and the fusion becomes
        identity, so the model degrades to vanilla YOLO.
      - sigmoid(+inf) = 1 -> multiplier = 2.0 (emphasize).
      - sigmoid(-inf) = 0 -> multiplier = 0.0 (suppress).
    """

    def forward(self, p: torch.Tensor, af: torch.Tensor) -> torch.Tensor:
        return p * (2.0 * torch.sigmoid(af))


class SegBranch(nn.Module):
    """Lightweight semantic-segmentation head that predicts a 1-channel anomaly heatmap.

    Adapted from upstream ``SemanticSegment`` (head.py). Consumes the P3 and P4 PAN
    features and emits per-pixel logits at P3 resolution (e.g. 80x80 for 640 input).
    A P4 auxiliary head provides deep supervision during training. The predicted
    heatmap (``sigmoid(logits)``) feeds ``HeatmapGuidedFusion`` at inference, so the
    model no longer needs an external/GT prior to localize anomalies.
    """

    def __init__(self, ch: tuple, nc: int = 1, c_mid: int | None = None):
        """Initialize the seg head.

        Args:
            ch (tuple): Channel sizes of the consumed feature maps (P3, P4).
            nc (int): Number of output channels (1 = binary anomaly heatmap).
            c_mid (int, optional): Intermediate width. Defaults to the P3 channel count.
        """
        super().__init__()
        self.nc = nc
        c_mid = ch[0] if c_mid is None else c_mid
        self.classifier = nn.Sequential(Conv(ch[0], c_mid, 3), nn.Conv2d(c_mid, nc, 1))
        self.aux_head = nn.Sequential(Conv(ch[1], c_mid, 3), nn.Conv2d(c_mid, nc, 1)) if len(ch) > 1 else None

    def forward(self, x: list[torch.Tensor]):
        """Predict per-pixel logits from [P3, P4].

        Returns:
            (torch.Tensor | tuple): main logits ``[B, nc, H/8, W/8]``; during training
                a ``(main, aux)`` tuple when the auxiliary head is present.
        """
        logits = self.classifier(x[0])
        if self.training and self.aux_head is not None:
            return logits, self.aux_head(x[1])
        return logits


def binary_seg_loss(
    logits: torch.Tensor, target: torch.Tensor, aux_logits: torch.Tensor | None = None, aux_weight: float = 0.4
) -> torch.Tensor:
    """BCE + soft-Dice loss for a single-channel anomaly heatmap.

    Args:
        logits: (B, 1, h, w) raw seg logits.
        target: (B, 1, H, W) binary mask in {0, 1}; resized to ``logits`` if needed.
        aux_logits: optional (B, 1, h', w') auxiliary logits for deep supervision.
        aux_weight: weight on the auxiliary BCE term.

    Returns:
        Scalar loss (per-sample mean; caller scales by batch size / gain).
    """
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
    # Smoke test
    B, H = 4, 80
    renderer_rect = BboxMaskRenderer(mask_size=H, mode="rect")
    renderer_gauss = BboxMaskRenderer(mask_size=H, mode="gauss")

    bboxes = torch.tensor(
        [
            [0.25, 0.25, 0.20, 0.20],
            [0.75, 0.75, 0.30, 0.10],
            [0.50, 0.50, 0.10, 0.10],
        ]
    )
    batch_idx = torch.tensor([0, 0, 2], dtype=torch.long)

    m_rect = renderer_rect(bboxes, batch_idx, B)
    m_gauss = renderer_gauss(bboxes, batch_idx, B)
    print(f"rect mask:  shape={tuple(m_rect.shape)}, "
          f"range=[{m_rect.min().item():.3f}, {m_rect.max().item():.3f}], "
          f"sum={m_rect.sum().item():.1f}")
    print(f"gauss mask: shape={tuple(m_gauss.shape)}, "
          f"range=[{m_gauss.min().item():.3f}, {m_gauss.max().item():.3f}], "
          f"sum={m_gauss.sum().item():.1f}")
    assert m_rect.shape == (B, 1, H, H)
    assert m_gauss.shape == (B, 1, H, H)
    # Image 1 has no bboxes -> all zeros
    assert m_rect[1].sum().item() == 0.0
    assert m_gauss[1].sum().item() == 0.0
    # Image 3 has no bboxes
    assert m_rect[3].sum().item() == 0.0

    # Empty input case
    m_empty = renderer_rect(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), B)
    assert m_empty.shape == (B, 1, H, H) and m_empty.sum().item() == 0.0

    # HeatmapEncoder for each PAN scale
    enc_p3 = HeatmapEncoder(c_out=256)
    enc_p4 = HeatmapEncoder(c_out=512)
    enc_p5 = HeatmapEncoder(c_out=512)
    mask_p3 = m_rect  # 80
    mask_p4 = torch.nn.functional.interpolate(m_rect, scale_factor=0.5, mode="bilinear", align_corners=False)
    mask_p5 = torch.nn.functional.interpolate(m_rect, scale_factor=0.25, mode="bilinear", align_corners=False)
    af_p3 = enc_p3(mask_p3)
    af_p4 = enc_p4(mask_p4)
    af_p5 = enc_p5(mask_p5)
    print(f"AF_P3: {tuple(af_p3.shape)}, AF_P4: {tuple(af_p4.shape)}, AF_P5: {tuple(af_p5.shape)}")
    assert af_p3.shape == (B, 256, 80, 80)
    assert af_p4.shape == (B, 512, 40, 40)
    assert af_p5.shape == (B, 512, 20, 20)

    # HeatmapGuidedFusion: passthrough at AF=0
    fusion = HeatmapGuidedFusion()
    p3 = torch.randn(B, 256, 80, 80)
    af_zero = torch.zeros_like(p3)
    p3_out = fusion(p3, af_zero)
    assert torch.allclose(p3_out, p3, atol=1e-6), "Fusion must be exact passthrough when AF=0"
    print(f"Fusion passthrough at AF=0: max abs diff = {(p3_out - p3).abs().max().item():.2e}")

    # Sanity: with AF=+10 features get scaled near 2x; with AF=-10 near 0
    p3_high = fusion(p3, torch.full_like(p3, 10.0))
    p3_low = fusion(p3, torch.full_like(p3, -10.0))
    ratio_high = (p3_high / p3.clamp(min=1e-6)).mean().item()
    print(f"AF=+10 -> mean multiplier ~2.0, got {2.0 * torch.sigmoid(torch.tensor(10.0)).item():.4f}")
    print(f"AF=-10 -> mean multiplier ~0.0, got {2.0 * torch.sigmoid(torch.tensor(-10.0)).item():.4f}")

    print("\nAll smoke tests passed.")
