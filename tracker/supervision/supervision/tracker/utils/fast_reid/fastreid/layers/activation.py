# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'Mish',
    'Swish',
    'MemoryEfficientSwish',
    'GELU']


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
