"""Tests + demo for the boundary-weighted segmentation loss feature.

Unit tests verify:
  1. boundary_band produces the expected morphological-gradient shape (1 on a k-wide
     band around each object's contour, 0 elsewhere).
  2. single_mask_loss returns bnd_loss == 0 when boundary_weight == 0.
  3. single_mask_loss returns bnd_loss > 0 when boundary_weight > 0 and the GT has
     a non-trivial contour.
  4. bnd_loss scales linearly with boundary_weight (it's a multiplicative uplift).
  5. seg_loss is identical regardless of boundary_weight (only the bnd column changes).

Demonstration (also a test — asserts the inequality at the end):
  6. Train a single predicted mask via gradient descent against a synthetic GT shape
     with three settings: weight=0, weight=2, weight=5. Track Boundary-IoU. Assert
     that weight>0 reaches a strictly higher final Boundary-IoU than weight=0.

Run all:
    pytest tests/test_boundary_loss.py -v -s

Run just the demo, with prints:
    pytest tests/test_boundary_loss.py::test_demo_higher_weight_improves_boundary_iou -v -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics.utils.loss import boundary_band, v8SegmentationLoss


def _make_gt_circle(size: int = 80, cy: int = 40, cx: int = 40, r: int = 20, device: str = "cpu") -> torch.Tensor:
    """Return a (1, H, W) binary mask with a single filled circle."""
    yy, xx = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing="ij")
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).float()
    return mask.unsqueeze(0)


def _hard_iou(pred_logits: torch.Tensor, gt: torch.Tensor) -> float:
    """Hard IoU on binarized predictions."""
    pred_bin = (pred_logits.sigmoid() > 0.5).float()
    inter = (pred_bin * gt).sum()
    union = ((pred_bin + gt) > 0).float().sum()
    return float(inter / (union + 1e-9))


def _soft_boundary_iou(pred_logits: torch.Tensor, gt: torch.Tensor, k: int = 3) -> float:
    """Soft Boundary-IoU on the GT boundary band: probability-weighted overlap.

    Unlike hard IoU, this metric never saturates at 1.0 with finite logits — it keeps reflecting how *confidently* the
    model classifies boundary pixels. That's exactly what boundary-weighted BCE is supposed to improve: not just getting
    boundary pixels on the right side of 0.5, but pushing their logits further from 0.
    """
    p = pred_logits.sigmoid()
    band = boundary_band(gt, k)
    inter = (p * gt * band).sum() + ((1 - p) * (1 - gt) * band).sum()
    denom = band.sum()
    return float(inter / (denom + 1e-9))


# ─────────────────────────────────────── boundary_band ────────────────────────────────────────


def test_boundary_band_all_zeros_input():
    """No object → no boundary."""
    gt = torch.zeros(1, 32, 32)
    band = boundary_band(gt, kernel=3)
    assert band.shape == gt.shape
    assert band.sum().item() == 0.0


def test_boundary_band_all_ones_input():
    """Fully filled mask → no boundary (dilation == erosion == 1)."""
    gt = torch.ones(1, 32, 32)
    band = boundary_band(gt, kernel=3)
    # Interior is empty; only the image-border pixels may have a band due to padding effects.
    interior = band[:, 1:-1, 1:-1]
    assert interior.sum().item() == 0.0


def test_boundary_band_circle_has_expected_thickness():
    """Band on a filled circle is roughly an annulus of thickness ~k pixels."""
    gt = _make_gt_circle(size=80, r=20)
    k = 3
    band = boundary_band(gt, kernel=k)
    # Band must be a subset of (dilate ∪ original) and cover the GT edge
    n_band = int(band.sum().item())
    # Circle perimeter ≈ 2πr ≈ 126. With k=3 dilation it widens to ~3× perimeter ≈ 380.
    # Allow a generous range.
    assert 200 < n_band < 600, f"unexpected band size {n_band} for r=20, k=3"


def test_boundary_band_kernel_5_is_thicker_than_kernel_3():
    """Bigger kernel → wider band."""
    gt = _make_gt_circle(size=80, r=20)
    band_3 = boundary_band(gt, kernel=3)
    band_5 = boundary_band(gt, kernel=5)
    assert band_5.sum().item() > band_3.sum().item()


# ─────────────────────────────────────── single_mask_loss ─────────────────────────────────────


def _make_mask_loss_inputs(device: str = "cpu"):
    """Build minimal inputs for v8SegmentationLoss.single_mask_loss.

    We use a single instance and exploit the einsum: pred (1, K) @ proto (K, H, W) → (1, H, W). Choosing K=1,
    pred=[[1.0]], and using `proto` directly as the logits map gives us full control.
    """
    H = W = 64
    gt = _make_gt_circle(size=H, r=15, device=device)  # (1, H, W)
    proto = (torch.randn(1, H, W, device=device) * 0.1).requires_grad_(True)  # acts as logits
    pred = torch.ones(1, 1, device=device, requires_grad=True)
    xyxy = torch.tensor([[5.0, 5.0, H - 5.0, W - 5.0]], device=device)  # bbox in proto pixels
    area = torch.tensor([(H - 10) * (W - 10) / float(H * W)], device=device)  # normalized area
    return gt, pred, proto, xyxy, area


def test_single_mask_loss_bnd_is_zero_when_weight_is_zero():
    gt, pred, proto, xyxy, area = _make_mask_loss_inputs()
    seg, bnd = v8SegmentationLoss.single_mask_loss(gt, pred, proto, xyxy, area, boundary_weight=0.0, boundary_kernel=3)
    assert seg.item() > 0.0
    assert bnd.item() == 0.0


def test_single_mask_loss_bnd_is_positive_when_weight_is_positive():
    gt, pred, proto, xyxy, area = _make_mask_loss_inputs()
    _, bnd = v8SegmentationLoss.single_mask_loss(gt, pred, proto, xyxy, area, boundary_weight=1.0, boundary_kernel=3)
    assert bnd.item() > 0.0


def test_single_mask_loss_seg_independent_of_boundary_weight():
    """Seg column is plain BCE — must be identical regardless of boundary_weight."""
    gt, pred, proto, xyxy, area = _make_mask_loss_inputs()
    seg_0, _ = v8SegmentationLoss.single_mask_loss(gt, pred, proto, xyxy, area, 0.0, 3)
    seg_5, _ = v8SegmentationLoss.single_mask_loss(gt, pred, proto, xyxy, area, 5.0, 3)
    assert torch.allclose(seg_0, seg_5, atol=1e-7)


def test_single_mask_loss_bnd_scales_linearly_with_weight():
    """Bnd = w * Σ(bce * band) → doubling w doubles bnd."""
    gt, pred, proto, xyxy, area = _make_mask_loss_inputs()
    _, bnd_1 = v8SegmentationLoss.single_mask_loss(gt, pred, proto, xyxy, area, 1.0, 3)
    _, bnd_2 = v8SegmentationLoss.single_mask_loss(gt, pred, proto, xyxy, area, 2.0, 3)
    _, bnd_5 = v8SegmentationLoss.single_mask_loss(gt, pred, proto, xyxy, area, 5.0, 3)
    assert torch.allclose(bnd_2, 2 * bnd_1, rtol=1e-5)
    assert torch.allclose(bnd_5, 5 * bnd_1, rtol=1e-5)


def test_single_mask_loss_gradient_amplified_on_boundary():
    """The boundary uplift must produce larger gradients on band pixels than on interior pixels."""
    gt, pred, proto, xyxy, area = _make_mask_loss_inputs()
    seg, bnd = v8SegmentationLoss.single_mask_loss(gt, pred, proto, xyxy, area, 5.0, 3)
    (seg + bnd).backward()
    band = boundary_band(gt, 3).bool()
    grad = proto.grad.abs()
    # mean |grad| inside the band should exceed mean |grad| in the interior
    interior = (gt > 0) & (~band)
    mean_grad_band = grad[band.expand_as(grad)].mean().item()
    mean_grad_interior = grad[interior.expand_as(grad)].mean().item()
    assert mean_grad_band > mean_grad_interior * 3, (
        f"Boundary gradient {mean_grad_band:.4g} not significantly larger than interior "
        f"{mean_grad_interior:.4g} with weight=5"
    )


# ───────────────────────────────────── End-to-end demo ────────────────────────────────────────


def _train_synthetic(
    weight: float,
    steps: int = 200,
    lr: float = 0.05,
    seed: int = 0,
    kernel: int = 3,
):
    """Optimize a per-pixel logit map to fit a synthetic circle with SGD.

    SGD is used (not Adam) because Adam's adaptive per-parameter scaling normalizes away the gradient-magnitude
    difference that's the *whole point* of boundary weighting. With SGD, a band pixel under weight=5 receives a 6×
    larger update step than the same pixel under weight=0, so the optimization paths genuinely differ.

    Loss uses `.sum()` reduction (not `.mean()`) so per-pixel gradients are not diluted by the 1/(H·W) factor — this
    keeps the effective per-pixel learning rate at `lr`, not `lr/6400`.

    Returns history and final logits.
    """
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt = _make_gt_circle(size=80, r=22, device=device)
    logits = torch.zeros_like(gt).requires_grad_(True)
    opt = torch.optim.SGD([logits], lr=lr)

    history = []
    for step in range(steps):
        opt.zero_grad()
        bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
        if weight > 0:
            band = boundary_band(gt, kernel)
            loss = (bce * (1.0 + weight * band)).sum()
        else:
            loss = bce.sum()
        loss.backward()
        opt.step()

        if step % 20 == 0 or step == steps - 1:
            with torch.no_grad():
                history.append(
                    {
                        "step": step,
                        "loss": loss.item(),
                        "iou": _hard_iou(logits, gt),
                        "sbiou": _soft_boundary_iou(logits, gt, kernel),
                    }
                )
    return history, logits.detach()


@pytest.mark.parametrize("steps", [200])
def test_demo_higher_weight_improves_boundary_iou(steps: int) -> None:
    """End-to-end: higher boundary_weight produces higher soft Boundary-IoU at the same step budget. Soft B-IoU
    (probability-weighted, on the GT boundary band) keeps increasing as the model becomes more confident on edge
    pixels, so it stays sensitive even after hard IoU saturates at 1.0.
    """
    settings = [0.0, 2.0, 5.0]
    finals = {}
    print(f"\n{'weight':>8} | {'step':>5} | {'loss':>8} | {'IoU':>6} | {'sB-IoU':>7}")
    print("-" * 54)
    for w in settings:
        history, _ = _train_synthetic(weight=w, steps=steps)
        for h in history:
            print(f"{w:>8.1f} | {h['step']:>5d} | {h['loss']:>8.4f} | {h['iou']:>6.3f} | {h['sbiou']:>7.4f}")
        print()
        finals[w] = history[-1]

    print("=" * 54)
    print(f"{'weight':>8} | {'final IoU':>9} | {'final sB-IoU':>12}")
    for w, h in finals.items():
        print(f"{w:>8.1f} | {h['iou']:>9.3f} | {h['sbiou']:>12.4f}")

    # Hard IoU should be ~equal across weights (this is the easy part — every weight converges).
    # The boundary effect lives in *confidence on the band*, captured by soft B-IoU.
    margin = finals[5.0]["sbiou"] - finals[0.0]["sbiou"]
    print(f"\nsoft B-IoU margin (weight=5 over weight=0): {margin:+.4f}")
    assert margin > 0.005, (
        f"Expected weight=5 to beat weight=0 on soft Boundary-IoU by >0.005 within {steps} SGD steps. "
        f"weight=0: {finals[0.0]['sbiou']:.4f}, weight=5: {finals[5.0]['sbiou']:.4f}, margin={margin:+.4f}"
    )


if __name__ == "__main__":
    # Allow running the demo standalone: `python tests/test_boundary_loss.py`
    test_demo_higher_weight_improves_boundary_iou(steps=200)
