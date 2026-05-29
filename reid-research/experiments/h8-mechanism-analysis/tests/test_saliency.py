"""Unit tests for Integrated Gradients on a toy linear model.

For a *linear* function f(x) = w · x with zero baseline,
IG attribution at position i equals w_i * x_i — exactly recovered by IG.
This gives us a closed-form test target.
"""
import numpy as np
import pytest
import torch

from saliency import integrated_gradients


def test_ig_recovers_linear_attribution():
    torch.manual_seed(0)
    D = 16
    w = torch.randn(D)

    def f(x):
        return (w * x).sum(dim=-1)

    x = torch.randn(D)
    attribution = integrated_gradients(f, x, baseline=torch.zeros(D), steps=50)
    expected = (w * x).numpy()
    # 50 Riemann steps on a linear function should be exact up to numerical noise.
    np.testing.assert_allclose(attribution.numpy(), expected, atol=1e-4)


def test_ig_handles_batch_dim():
    torch.manual_seed(1)
    D = 8

    def f(x):
        return (x ** 2).sum(dim=-1)

    x = torch.randn(D)
    attribution = integrated_gradients(f, x, baseline=torch.zeros(D), steps=50)
    assert attribution.shape == x.shape


def test_ig_nan_detection_raises():
    """If the model returns NaN, IG should raise so the caller can skip-and-log."""

    def f(x):
        return torch.tensor(float("nan"))

    x = torch.randn(4)
    with pytest.raises(ValueError, match="NaN"):
        integrated_gradients(f, x, baseline=torch.zeros(4), steps=10)
