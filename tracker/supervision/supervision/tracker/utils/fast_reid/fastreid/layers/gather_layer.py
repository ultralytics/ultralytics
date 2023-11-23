# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

# based on: https://github.com/open-mmlab/OpenSelfSup/blob/master/openselfsup/models/utils/gather_layer.py

import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
                  for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
