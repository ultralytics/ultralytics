# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

# based on: https://github.com/PhilJd/contiguous_pytorch_params/blob/master/contiguous_params/params.py

from collections import OrderedDict

import torch


class ContiguousParams:

    def __init__(self, parameters):
        # Create a list of the parameters to prevent emptying an iterator.
        self._parameters = parameters
        self._param_buffer = []
        self._grad_buffer = []
        self._group_dict = OrderedDict()
        self._name_buffer = []
        self._init_buffers()
        # Store the data pointers for each parameter into the buffer. These
        # can be used to check if an operation overwrites the gradient/data
        # tensor (invalidating the assumption of a contiguous buffer).
        self.data_pointers = []
        self.grad_pointers = []
        self.make_params_contiguous()

    def _init_buffers(self):
        dtype = self._parameters[0]["params"][0].dtype
        device = self._parameters[0]["params"][0].device
        if not all(p["params"][0].dtype == dtype for p in self._parameters):
            raise ValueError("All parameters must be of the same dtype.")
        if not all(p["params"][0].device == device for p in self._parameters):
            raise ValueError("All parameters must be on the same device.")

        # Group parameters by lr and weight decay
        for param_dict in self._parameters:
            freeze_status = param_dict["freeze_status"]
            param_key = freeze_status + '_' + str(param_dict["lr"]) + '_' + str(param_dict["weight_decay"])
            if param_key not in self._group_dict:
                self._group_dict[param_key] = []
            self._group_dict[param_key].append(param_dict)

        for key, params in self._group_dict.items():
            size = sum(p["params"][0].numel() for p in params)
            self._param_buffer.append(torch.zeros(size, dtype=dtype, device=device))
            self._grad_buffer.append(torch.zeros(size, dtype=dtype, device=device))
            self._name_buffer.append(key)

    def make_params_contiguous(self):
        """Create a buffer to hold all params and update the params to be views of the buffer.
        Args:
            parameters: An iterable of parameters.
        """
        for i, params in enumerate(self._group_dict.values()):
            index = 0
            for param_dict in params:
                p = param_dict["params"][0]
                size = p.numel()
                self._param_buffer[i][index:index + size] = p.data.view(-1)
                p.data = self._param_buffer[i][index:index + size].view(p.data.shape)
                p.grad = self._grad_buffer[i][index:index + size].view(p.data.shape)
                self.data_pointers.append(p.data.data_ptr)
                self.grad_pointers.append(p.grad.data.data_ptr)
                index += size
            # Bend the param_buffer to use grad_buffer to track its gradients.
            self._param_buffer[i].grad = self._grad_buffer[i]

    def contiguous(self):
        """Return all parameters as one contiguous buffer."""
        return [{
            "freeze_status": self._name_buffer[i].split('_')[0],
            "params": self._param_buffer[i],
            "lr": float(self._name_buffer[i].split('_')[1]),
            "weight_decay": float(self._name_buffer[i].split('_')[2]),
        } for i in range(len(self._param_buffer))]

    def original(self):
        """Return the non-flattened parameters."""
        return self._parameters

    def buffer_is_valid(self):
        """Verify that all parameters and gradients still use the buffer."""
        i = 0
        for params in self._group_dict.values():
            for param_dict in params:
                p = param_dict["params"][0]
                data_ptr = self.data_pointers[i]
                grad_ptr = self.grad_pointers[i]
                if (p.data.data_ptr() != data_ptr()) or (p.grad.data.data_ptr() != grad_ptr()):
                    return False
                i += 1
        return True

    def assert_buffer_is_valid(self):
        if not self.buffer_is_valid():
            raise ValueError(
                "The data or gradient buffer has been invalidated. Please make "
                "sure to use inplace operations only when updating parameters "
                "or gradients.")
