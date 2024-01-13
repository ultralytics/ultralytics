import pytest
import subprocess

import torch
import torch.nn as nn

from ultralytics.utils.torch_utils import fuse_conv_and_bn
from ultralytics.nn.modules.conv import Conv, ConvTranspose

from ultralytics.utils import ASSETS, WEIGHTS_DIR, checks

CUDA_IS_AVAILABLE = checks.cuda_is_available()
CUDA_DEVICE_COUNT = checks.cuda_device_count()


def max_diff(a, b):
    # Check if shapes are the same
    if a.shape != b.shape:
        raise ValueError("Input shapes must be the same")
    return torch.max(torch.abs(a - b)).item()


@pytest.fixture
def setup():
    conv = Conv(3, 64, 3, 1).eval()
    deconv = ConvTranspose(64, 3, 3, 1).eval()

    return {'conv': conv, 'deconv': deconv}


def test_fuse_instance(setup):
    # Test the function for 2D Convolution and BatchNormalization
    fused_conv_2d = fuse_conv_and_bn(setup['conv'].conv, setup['conv'].bn)
    assert isinstance(fused_conv_2d, nn.Conv2d)

    # Test the function for 2D Transposed Convolution and BatchNormalization
    fused_trans_conv_2d = fuse_conv_and_bn(setup['deconv'].conv_transpose, setup['deconv'].bn)
    assert isinstance(fused_trans_conv_2d, nn.ConvTranspose2d)


def test_fuse_conv_and_bn(setup):
    # Input a random batch
    x = torch.randn(10, 3, 224, 224)  # (batch_size, in_channels, height, width)
    output = setup['conv'](x)

    setup['conv'].conv = fuse_conv_and_bn(setup['conv'].conv, setup['conv'].bn)
    delattr(setup['conv'], "bn")  # remove batchnorm
    setup['conv'].forward = setup['conv'].forward_fuse  # update forward
    fused_output = setup['conv'](x)

    # Ensure the fused_conv output matches conv and bn output
    error = (output - fused_output).abs().mean().item()
    print(f"\nError  (2D Conv): {error}")
    print(f"Max diff = {max_diff(output, fused_output)}")
    assert torch.allclose(output, fused_output, atol=1e-5)


def test_fuse_deconv_and_bn(setup):
    # Input a random batch
    x = torch.randn(10, 64, 36, 36)  # (batch_size, in_channels, height, width)
    output = setup['deconv'](x)

    setup['deconv'].conv_transpose = fuse_conv_and_bn(setup['deconv'].conv_transpose, setup['deconv'].bn)
    delattr(setup['deconv'], "bn")  # remove batchnorm
    setup['deconv'].forward = setup['deconv'].forward_fuse  # update forward
    fused_output = setup['deconv'](x)

    # Ensure the fused_conv output matches conv and bn output
    error = (output - fused_output).abs().mean().item()
    print(f"\nError (2D Transpose): {error}")
    print(f"Max diff = {max_diff(output, fused_output)}")
    assert torch.allclose(output, fused_output, atol=1e-5)
