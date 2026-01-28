import torch
import pytest

from ultralytics.utils.ops import make_divisible


def test_make_divisible_basic():
    assert make_divisible(33, 8) == 40
    assert make_divisible(32, 8) == 32
    assert make_divisible(1, 8) == 8
    assert make_divisible(0, 8) == 0


def test_make_divisible_large_numbers():
    assert make_divisible(127, 16) == 128
    assert make_divisible(1025, 32) == 1056


def test_make_divisible_tensor_divisor():
    divisor = torch.tensor([8])
    assert make_divisible(64, divisor) == 64
    assert make_divisible(65, divisor) == 72


def test_make_divisible_tensor_multi_value():
    divisor = torch.tensor([4, 8, 16])
    # uses max() internally
    assert make_divisible(30, divisor) == 32
