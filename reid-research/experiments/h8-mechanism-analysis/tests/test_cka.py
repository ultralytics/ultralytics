"""Unit tests for Linear-CKA implementation.

CKA properties tested:
- Identical features yield CKA = 1
- Orthogonal (uncorrelated random) features yield CKA ~ 0
- Symmetric: CKA(X, Y) == CKA(Y, X)
- Invariant to orthogonal transforms of either input
- Invariant to isotropic scaling
"""
import numpy as np
import pytest

from cka import linear_cka


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_identical_features_cka_is_one(rng):
    X = rng.standard_normal((100, 64))
    assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-6)


def test_independent_random_features_cka_near_zero(rng):
    X = rng.standard_normal((200, 64))
    Y = rng.standard_normal((200, 64))
    assert abs(linear_cka(X, Y)) < 0.3  # finite-sample noise floor


def test_symmetric(rng):
    X = rng.standard_normal((100, 64))
    Y = rng.standard_normal((100, 32))
    assert linear_cka(X, Y) == pytest.approx(linear_cka(Y, X), abs=1e-6)


def test_invariant_to_orthogonal_transform(rng):
    X = rng.standard_normal((100, 64))
    Y = rng.standard_normal((100, 64))
    # Random orthogonal matrix in 64d
    Q, _ = np.linalg.qr(rng.standard_normal((64, 64)))
    assert linear_cka(X, Y) == pytest.approx(linear_cka(X @ Q, Y), abs=1e-6)


def test_invariant_to_isotropic_scale(rng):
    X = rng.standard_normal((100, 64))
    Y = rng.standard_normal((100, 64))
    assert linear_cka(X, Y) == pytest.approx(linear_cka(3.0 * X, Y), abs=1e-6)
