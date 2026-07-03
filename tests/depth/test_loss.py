from types import SimpleNamespace

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DepthLoss


def _args(**over):
    """Depth hyp namespace with the keys v8DepthLoss reads; override per test."""
    a = dict(
        silog=1.0,
        silog_grad=0.5,
        silog_lambda=0.5,
        silog_l1=0.0,
        dist_pw=0.0,
        silog_grad_scales=4,
        silog_grad_min_valid=0.5,
        silog_trim=0.0,
    )
    a.update(over)
    return SimpleNamespace(**a)


class _Model(torch.nn.Module):
    """Tiny stub exposing .parameters() and .args so the loss knobs can be varied per test."""

    def __init__(self, **over):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.args = _args(**over)


def _loss_for_scaled_pred(lam, scale, l1=0.0):
    crit = v8DepthLoss(_Model(silog_lambda=lam, silog_l1=l1, silog_grad=0.0))  # silog only
    gt = torch.rand(2, 1, 16, 16) * 5 + 1.0
    pred = (gt * scale).clone().requires_grad_(True)  # perfect structure, wrong global scale
    total, _ = crit({"depth": pred}, {"depth": gt})
    return float(total.detach())


def test_lower_lambda_penalizes_scale_error_more():
    """A globally scale-shifted prediction is ~free under scale-invariant silog (lambda=1) but
    must be heavily penalized as lambda drops (loss becomes scale-dependent)."""
    loss_invariant = _loss_for_scaled_pred(lam=1.0, scale=2.0)
    loss_anchored = _loss_for_scaled_pred(lam=0.15, scale=2.0)
    assert loss_invariant < 0.05
    assert loss_anchored > 5 * max(loss_invariant, 1e-6)


def test_l1_weight_adds_scale_penalty():
    """The scale-anchored L1 term penalizes a scale shift even when silog is scale-invariant."""
    no_l1 = _loss_for_scaled_pred(lam=1.0, scale=2.0, l1=0.0)
    with_l1 = _loss_for_scaled_pred(lam=1.0, scale=2.0, l1=1.0)
    assert with_l1 > no_l1 + 0.1


def test_grad_scales_1_matches_single_scale():
    """silog_grad_scales=1 must reproduce the original single-scale gradient loss byte-for-byte."""
    torch.manual_seed(0)
    gt = torch.rand(2, 1, 16, 16) * 5 + 1.0
    pred = (gt + 0.3 * torch.randn(2, 1, 16, 16)).clamp(min=0.1)
    crit = v8DepthLoss(_Model(silog=0.0, silog_grad=1.0, silog_grad_scales=1))
    total, _ = crit({"depth": pred}, {"depth": gt})

    pl, gl = torch.log(pred.clamp(min=0.001)), torch.log(gt.clamp(min=0.001))
    ref = F.l1_loss(pl[:, :, :, 1:] - pl[:, :, :, :-1], gl[:, :, :, 1:] - gl[:, :, :, :-1])
    ref = ref + F.l1_loss(pl[:, :, 1:, :] - pl[:, :, :-1, :], gl[:, :, 1:, :] - gl[:, :, :-1, :])
    assert abs(float(total) - float(ref) * pred.shape[0]) < 1e-5


def test_multiscale_grad_adds_coarse_levels():
    """silog_grad_scales>1 sums non-negative coarse-level terms, so it exceeds the single-scale
    loss whenever the prediction mismatches GT at coarse scales (generic random case)."""
    torch.manual_seed(1)
    gt = torch.rand(2, 1, 32, 32) * 5 + 1.0
    pred = (gt + 0.5 * torch.randn(2, 1, 32, 32)).clamp(min=0.1)
    single, _ = v8DepthLoss(_Model(silog=0.0, silog_grad=1.0, silog_grad_scales=1))({"depth": pred}, {"depth": gt})
    multi, _ = v8DepthLoss(_Model(silog=0.0, silog_grad=1.0, silog_grad_scales=4))({"depth": pred}, {"depth": gt})
    assert float(multi) > float(single) + 1e-4


def test_sparsity_guard_collapses_multiscale_on_sparse_gt():
    """On sparse GT (valid fraction < silog_grad_min_valid), multi-scale must self-disable and
    match single-scale — the guard blocks unreliable coarse gradients (the KITTI failure mode)."""
    torch.manual_seed(3)
    gt = torch.zeros(1, 1, 32, 32)
    mask = torch.rand(1, 1, 32, 32) < 0.15  # ~15% valid, below the 0.5 threshold
    gt[mask] = torch.rand(int(mask.sum())) * 5 + 1.0
    pred = torch.rand(1, 1, 32, 32) * 5 + 1.0
    single, _ = v8DepthLoss(_Model(silog=0.0, silog_grad=1.0, silog_grad_scales=1))({"depth": pred}, {"depth": gt})
    multi, _ = v8DepthLoss(_Model(silog=0.0, silog_grad=1.0, silog_grad_scales=4))({"depth": pred}, {"depth": gt})
    assert abs(float(single) - float(multi)) < 1e-6  # guard collapsed ms to single-scale


def test_trim_pct_0_is_no_op():
    """silog_trim=0.0 leaves the SILog term unchanged (regression against pre-trim behavior)."""
    torch.manual_seed(2)
    gt = torch.rand(1, 1, 16, 16) * 5 + 1.0
    pred = (gt + 0.2 * torch.randn(1, 1, 16, 16)).clamp(min=0.1)
    a, _ = v8DepthLoss(_Model(silog_grad=0.0, silog_trim=0.0))({"depth": pred}, {"depth": gt})
    b, _ = v8DepthLoss(_Model(silog_grad=0.0))({"depth": pred}, {"depth": gt})  # default trim 0.0
    assert abs(float(a) - float(b)) < 1e-6


def test_trimming_drops_outliers():
    """A handful of gross-error pixels blow up scale-invariant SILog; trimming removes them."""
    gt = torch.full((1, 1, 16, 16), 2.0)
    pred = gt.clone()
    pred[0, 0, 0, :5] = 20.0  # 5 / 256 ≈ 2% gross outliers
    no_trim, _ = v8DepthLoss(_Model(silog_grad=0.0, silog_trim=0.0))({"depth": pred}, {"depth": gt})
    with_trim, _ = v8DepthLoss(_Model(silog_grad=0.0, silog_trim=0.05))({"depth": pred}, {"depth": gt})
    assert float(no_trim) > 0.1
    assert float(with_trim) < 0.01
