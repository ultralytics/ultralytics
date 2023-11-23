# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

from typing import Dict, List

import torch
from torch._six import container_abcs
from torch.cuda.amp import GradScaler


class _MultiDeviceReplicator(object):
    """
    Lazily serves copies of a tensor to requested devices.  Copies are cached per-device.
    """

    def __init__(self, master_tensor: torch.Tensor) -> None:
        assert master_tensor.is_cuda
        self.master = master_tensor
        self._per_device_tensors: Dict[torch.device, torch.Tensor] = {}

    def get(self, device) -> torch.Tensor:
        retval = self._per_device_tensors.get(device, None)
        if retval is None:
            retval = self.master.to(device=device, non_blocking=True, copy=True)
            self._per_device_tensors[device] = retval
        return retval


class MaxClipGradScaler(GradScaler):
    def __init__(self, init_scale, max_scale: float, growth_interval=100):
        super().__init__(init_scale=init_scale, growth_interval=growth_interval)
        self.max_scale = max_scale

    def scale_clip(self):
        if self.get_scale() == self.max_scale:
            self.set_growth_factor(1)
        elif self.get_scale() < self.max_scale:
            self.set_growth_factor(2)
        elif self.get_scale() > self.max_scale:
            self._scale.fill_(self.max_scale)
            self.set_growth_factor(1)

    def scale(self, outputs):
        """
        Multiplies ('scales') a tensor or list of tensors by the scale factor.
        Returns scaled outputs.  If this instance of :class:`GradScaler` is not enabled, outputs are returned
        unmodified.
        Arguments:
            outputs (Tensor or iterable of Tensors):  Outputs to scale.
        """
        if not self._enabled:
            return outputs
        self.scale_clip()
        # Short-circuit for the common case.
        if isinstance(outputs, torch.Tensor):
            assert outputs.is_cuda
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(outputs.device)
            assert self._scale is not None
            return outputs * self._scale.to(device=outputs.device, non_blocking=True)

        # Invoke the more complex machinery only if we're treating multiple outputs.
        stash: List[_MultiDeviceReplicator] = []  # holds a reference that can be overwritten by apply_scale

        def apply_scale(val):
            if isinstance(val, torch.Tensor):
                assert val.is_cuda
                if len(stash) == 0:
                    if self._scale is None:
                        self._lazy_init_scale_growth_tracker(val.device)
                    assert self._scale is not None
                    stash.append(_MultiDeviceReplicator(self._scale))
                return val * stash[0].get(val.device)
            elif isinstance(val, container_abcs.Iterable):
                iterable = map(apply_scale, val)
                if isinstance(val, list) or isinstance(val, tuple):
                    return type(val)(iterable)
                else:
                    return iterable
            else:
                raise ValueError("outputs must be a Tensor or an iterable of Tensors")

        return apply_scale(outputs)
