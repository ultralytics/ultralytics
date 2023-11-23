# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import ReLU, LeakyReLU
from torch.nn.parameter import Parameter


class TLU(nn.Module):
    def __init__(self, num_features):
        """max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau"""
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = Parameter(torch.Tensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau.view(1, self.num_features, 1, 1))


class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6, is_eps_leanable=False):
        """
        weight = gamma, bias = beta
        beta, gamma:
            Variables of shape [1, 1, 1, C]. if TensorFlow
            Variables of shape [1, C, 1, 1]. if PyTorch
        eps: A scalar constant or learnable variable.
        """
        super(FRN, self).__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable

        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        if is_eps_leanable:
            self.eps = Parameter(torch.Tensor(1))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)

    def forward(self, x):
        """
        0, 1, 2, 3 -> (B, H, W, C) in TensorFlow
        0, 1, 2, 3 -> (B, C, H, W) in PyTorch
        TensorFlow code
            nu2 = tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True)
            x = x * tf.rsqrt(nu2 + tf.abs(eps))
            # This Code include TLU function max(y, tau)
            return tf.maximum(gamma * x + beta, tau)
        """
        # Compute the mean norm of activations per channel.
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)

        # Perform FRN.
        x = x * torch.rsqrt(nu2 + self.eps.abs())

        # Scale and Bias
        x = self.weight.view(1, self.num_features, 1, 1) * x + self.bias.view(1, self.num_features, 1, 1)
        # x = self.weight * x + self.bias
        return x


def bnrelu_to_frn(module):
    """
    Convert 'BatchNorm2d + ReLU' to 'FRN + TLU'
    """
    mod = module
    before_name = None
    before_child = None
    is_before_bn = False

    for name, child in module.named_children():
        if is_before_bn and isinstance(child, (ReLU, LeakyReLU)):
            # Convert BN to FRN
            if isinstance(before_child, BatchNorm2d):
                mod.add_module(
                    before_name, FRN(num_features=before_child.num_features))
            else:
                raise NotImplementedError()

            # Convert ReLU to TLU
            mod.add_module(name, TLU(num_features=before_child.num_features))
        else:
            mod.add_module(name, bnrelu_to_frn(child))

        before_name = name
        before_child = child
        is_before_bn = isinstance(child, BatchNorm2d)
    return mod


def convert(module, flag_name):
    mod = module
    before_ch = None
    for name, child in module.named_children():
        if hasattr(child, flag_name) and getattr(child, flag_name):
            if isinstance(child, BatchNorm2d):
                before_ch = child.num_features
                mod.add_module(name, FRN(num_features=child.num_features))
            # TODO bn is no good...
            if isinstance(child, (ReLU, LeakyReLU)):
                mod.add_module(name, TLU(num_features=before_ch))
        else:
            mod.add_module(name, convert(child, flag_name))
    return mod


def remove_flags(module, flag_name):
    mod = module
    for name, child in module.named_children():
        if hasattr(child, 'is_convert_frn'):
            delattr(child, flag_name)
            mod.add_module(name, remove_flags(child, flag_name))
        else:
            mod.add_module(name, remove_flags(child, flag_name))
    return mod


def bnrelu_to_frn2(model, input_size=(3, 128, 128), batch_size=2, flag_name='is_convert_frn'):
    forard_hooks = list()
    backward_hooks = list()

    is_before_bn = [False]

    def register_forward_hook(module):
        def hook(self, input, output):
            if isinstance(module, (nn.Sequential, nn.ModuleList)) or (module == model):
                is_before_bn.append(False)
                return

            # input and output is required in hook def
            is_converted = is_before_bn[-1] and isinstance(self, (ReLU, LeakyReLU))
            if is_converted:
                setattr(self, flag_name, True)
            is_before_bn.append(isinstance(self, BatchNorm2d))

        forard_hooks.append(module.register_forward_hook(hook))

    is_before_relu = [False]

    def register_backward_hook(module):
        def hook(self, input, output):
            if isinstance(module, (nn.Sequential, nn.ModuleList)) or (module == model):
                is_before_relu.append(False)
                return
            is_converted = is_before_relu[-1] and isinstance(self, BatchNorm2d)
            if is_converted:
                setattr(self, flag_name, True)
            is_before_relu.append(isinstance(self, (ReLU, LeakyReLU)))

        backward_hooks.append(module.register_backward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(batch_size, *in_size) for in_size in input_size]

    # register hook
    model.apply(register_forward_hook)
    model.apply(register_backward_hook)

    # make a forward pass
    output = model(*x)
    output.sum().backward()  # Raw output is not enabled to use backward()

    # remove these hooks
    for h in forard_hooks:
        h.remove()
    for h in backward_hooks:
        h.remove()

    model = convert(model, flag_name=flag_name)
    model = remove_flags(model, flag_name=flag_name)
    return model
