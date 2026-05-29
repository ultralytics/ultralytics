"""Unit tests for occlusion score derivation from a person mask.

The segmentation model load is integration-tested in Stage 1's sanity gate;
here we test only the mask-to-score math.
"""
import numpy as np
import pytest

from segmentation import mask_to_occlusion_score


def test_full_person_mask_is_zero_occlusion():
    mask = np.ones((128, 64), dtype=bool)
    assert mask_to_occlusion_score(mask) == pytest.approx(0.0)


def test_empty_mask_is_full_occlusion():
    mask = np.zeros((128, 64), dtype=bool)
    assert mask_to_occlusion_score(mask) == pytest.approx(1.0)


def test_half_mask_is_half_occlusion():
    mask = np.zeros((128, 64), dtype=bool)
    mask[:64, :] = True
    assert mask_to_occlusion_score(mask) == pytest.approx(0.5)


def test_none_mask_returns_nan():
    """No person detected at all -> we return NaN, not 1.0, so callers can distinguish
    'pure occlusion' from 'segmenter failed'."""
    assert np.isnan(mask_to_occlusion_score(None))
