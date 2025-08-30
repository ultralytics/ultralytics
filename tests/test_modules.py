# Scripts for modules

import pytest
import torch

from ultralytics.nn.modules import Permute, Reshape


@pytest.mark.parametrize(
    "shape_in, shape_out, permutation",
    [((3, 20, 20, 512), (3, 512, 20, 20), [0, 3, 1, 2]), ((4, 80, 90, 10), (10, 4, 80, 90), [3, 0, 1, 2])],
)
def test_permute(shape_in, shape_out, permutation):
    """Test for Permute module."""
    layer = Permute(permutation)
    ip = torch.randn(*shape_in)
    out = layer(ip)
    assert out.shape == shape_out, (
        f"Test failed for Permute module. Input shape: {shape_in}, Output shape: {out.shape}, Expected shape: {shape_out}"
    )
    torch.testing.assert_close(torch.sort(out.reshape(-1)).values, torch.sort(ip.reshape(-1)).values)


@pytest.mark.parametrize(
    "shape_in, shape_out", [((3, 20, 20, 512), (3, 512, 20, 20)), ((4, 80, 90, 10), (4, 7200, 10))]
)
def test_reshape(shape_in, shape_out):
    """Test for Reshape module."""
    layer = Reshape(shape_out)
    ip = torch.randn(*shape_in)
    out = layer(ip)
    assert out.shape == shape_out, (
        f"Test failed for Reshape module. Input shape: {shape_in}, Output shape: {out.shape}, Expected shape: {shape_out}"
    )
    torch.testing.assert_close(torch.sort(out.reshape(-1)).values, torch.sort(ip.reshape(-1)).values)
