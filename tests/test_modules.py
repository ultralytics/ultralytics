# Scripts for modules

import pytest
import torch
from ultralytics.nn.modules import Permute, Reshape

@pytest.mark.parametrize("shape_in, shape_out", [
    ((3, 20, 20, 512), (3, 512, 20, 20)),
    ((4, 80, 90, 10), (10, 4, 80, 90))    
])
def test_permute(shape_in, shape_out):
    layer = Permute(shape_out)
    ip = torch.Tensor(*shape_in)
    out = layer(ip)
    assert out.shape == shape_out, f"Test failed for Permute module. Input shape: {shape_in}, Output shape: {out.shape}, Expected shape: {shape_out}"
    assert sorted(out) == sorted(ip), f"Test failed for Permute module. Input values do not mach with the output values."


@pytest.mark.parametrize("shape_in, shape_out", [
    ((3, 20, 20, 512), (3, 512, 20, 20)),
    ((4, 80, 90, 10), (4, 7200, 10))    
])
def test_reshape(shape_in, shape_out):
    layer = Reshape(shape_out)
    ip = torch.Tensor(*shape_in)
    out = layer(ip)
    assert out.shape == shape_out, f"Test failed for Reshape module. Input shape: {shape_in}, Output shape: {out.shape}, Expected shape: {shape_out}"
    assert sorted(out) == sorted(ip), f"Test failed for Reshape module. Input values do not mach output values."