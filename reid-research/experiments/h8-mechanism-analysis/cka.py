"""Linear Centered Kernel Alignment (CKA) — Kornblith et al. 2019.

Used in h8 to localise *where* champion and SOLIDER representations diverge.
NOTE: cross-architecture CKA between a CNN (champion P4/P5) and a Swin transformer
(SOLIDER block-3/4) is a coarse instrument — see s3_findings.md caveat.
"""

from __future__ import annotations

import numpy as np


def _center(X: np.ndarray) -> np.ndarray:
    """Mean-centre rows."""
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two feature matrices, each (N, D_x) and (N, D_y).

    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    Returns a scalar in [0, 1] (modulo finite-sample noise).
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"row count mismatch: {X.shape[0]} != {Y.shape[0]}")
    Xc = _center(X)
    Yc = _center(Y)
    num = np.linalg.norm(Yc.T @ Xc, ord="fro") ** 2
    den = np.linalg.norm(Xc.T @ Xc, ord="fro") * np.linalg.norm(Yc.T @ Yc, ord="fro")
    if den == 0:
        return 0.0
    return float(num / den)
