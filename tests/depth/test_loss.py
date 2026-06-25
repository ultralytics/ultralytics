from types import SimpleNamespace

import torch

from ultralytics.utils.loss import v8DepthLoss


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))


def _loss_for_scaled_pred(lam, scale, l1=0.0):
    model = _Tiny()
    # grad weight 0 isolates the silog/scale terms
    model.args = SimpleNamespace(
        silog=1.0, silog_grad=0.0, silog_lambda=lam, silog_l1=l1, dist_pw=0.0
    )
    crit = v8DepthLoss(model)
    gt = torch.rand(2, 1, 16, 16) * 5 + 1.0
    pred = (gt * scale).clone().requires_grad_(True)  # perfect structure, wrong global scale
    total, _ = crit({"depth": pred}, {"depth": gt})
    return float(total)


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
