# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Multibin orientation encoding/decoding for stereo 3D detection.

Replaces the single sin/cos observation-angle (alpha) regression with a MultiBin
head (Mousavian et al., "3D Bounding Box Estimation Using Deep Learning and
Geometry"). The circle is split into overlapping bins; for each bin the head
predicts a confidence and a residual rotation (sin/cos of the offset from the
bin center). This lets the network commit to a heading bin and refine within it,
which improves orientation/AOS over a single sin/cos regressor.

Channel layout for N bins (here N=2): [conf_0..conf_{N-1}, sin_0, cos_0, ..., sin_{N-1}, cos_{N-1}].
For N=2 that is 6 channels: [c0, c1, s0, k0, s1, k1].

This module is the SINGLE SOURCE OF TRUTH for the bin layout: dataset.py encodes
targets with encode_orientation(), preprocess.py decodes with decode_orientation(),
and loss.py reads the same layout. Keeping encode+decode together guards against
the encode/decode mismatch class of bug.
"""

from __future__ import annotations

import math

# Bin centers (observation angle alpha, radians). Two bins at 0 and pi resolve the
# front/back (~180 deg) heading ambiguity that a single sin/cos regressor conflates.
ORIENT_BINS: tuple[float, ...] = (0.0, math.pi)
NUM_ORIENT_BINS: int = len(ORIENT_BINS)
ORIENT_CHANNELS: int = NUM_ORIENT_BINS * 3  # conf + (sin, cos) per bin


def _wrap(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def encode_orientation(alpha: float) -> list[float]:
    """Encode an observation angle alpha into the MultiBin target vector.

    Args:
        alpha: Observation angle in radians (rotation_y - ray_angle).

    Returns:
        A length-ORIENT_CHANNELS list: [conf_0..conf_{N-1}, sin_0, cos_0, ...],
        where conf is a one-hot of the nearest bin and (sin_i, cos_i) encode the
        residual offset of alpha from bin center i.
    """
    diffs = [_wrap(alpha - c) for c in ORIENT_BINS]
    nearest = min(range(NUM_ORIENT_BINS), key=lambda i: abs(diffs[i]))
    conf = [0.0] * NUM_ORIENT_BINS
    conf[nearest] = 1.0
    res: list[float] = []
    for d in diffs:
        res.extend((math.sin(d), math.cos(d)))
    return conf + res


def decode_orientation(pred: list[float]) -> float:
    """Decode a MultiBin prediction vector back to an observation angle alpha.

    Args:
        pred: Length-ORIENT_CHANNELS values [conf_0..conf_{N-1}, sin_0, cos_0, ...]. Confidences may be raw logits or
            probabilities (argmax is used).

    Returns:
        Observation angle alpha in radians, wrapped to [-pi, pi).
    """
    confs = pred[:NUM_ORIENT_BINS]
    b = max(range(NUM_ORIENT_BINS), key=lambda i: confs[i])
    s = pred[NUM_ORIENT_BINS + 2 * b]
    c = pred[NUM_ORIENT_BINS + 2 * b + 1]
    return _wrap(ORIENT_BINS[b] + math.atan2(s, c))
