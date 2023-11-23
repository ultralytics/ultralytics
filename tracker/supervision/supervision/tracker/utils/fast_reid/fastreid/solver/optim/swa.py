# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""
# based on:
# https://github.com/pytorch/contrib/blob/master/torchcontrib/optim/swa.py

import warnings
from collections import defaultdict

import torch
from torch.optim.optimizer import Optimizer


class SWA(Optimizer):
    def __init__(self, optimizer, swa_freq=None, swa_lr_factor=None):
        r"""Implements Stochastic Weight Averaging (SWA).
        Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
        Wider Optima and Better Generalization`_ by Pavel Izmailov, Dmitrii
        Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
        (UAI 2018).
        SWA is implemented as a wrapper class taking optimizer instance as input
        and applying SWA on top of that optimizer.
        SWA can be used in two modes: automatic and manual. In the automatic
        mode SWA running averages are automatically updated every
        :attr:`swa_freq` steps after :attr:`swa_start` steps of optimization. If
        :attr:`swa_lr` is provided, the learning rate of the optimizer is reset
        to :attr:`swa_lr` at every step starting from :attr:`swa_start`. To use
        SWA in automatic mode provide values for both :attr:`swa_start` and
        :attr:`swa_freq` arguments.
        Alternatively, in the manual mode, use :meth:`update_swa` or
        :meth:`update_swa_group` methods to update the SWA running averages.
        In the end of training use `swap_swa_sgd` method to set the optimized
        variables to the computed averages.
        Args:
            swa_freq (int): number of steps between subsequent updates of
                SWA running averages in automatic mode; if None, manual mode is
                selected (default: None)
            swa_lr (float): learning rate to use starting from step swa_start
                in automatic mode; if None, learning rate is not changed
                (default: None)
        Examples:
            >>> # automatic mode
            >>> base_opt = torch.optim.SGD(model.parameters(), lr=0.1)
            >>> opt = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)
            >>> for _ in range(100):
            >>>     opt.zero_grad()
            >>>     loss_fn(model(input), target).backward()
            >>>     opt.step()
            >>> opt.swap_swa_param()
            >>> # manual mode
            >>> opt = SWA(base_opt)
            >>> for i in range(100):
            >>>     opt.zero_grad()
            >>>     loss_fn(model(input), target).backward()
            >>>     opt.step()
            >>>     if i > 10 and i % 5 == 0:
            >>>         opt.update_swa()
            >>> opt.swap_swa_param()
        .. note::
            SWA does not support parameter-specific values of :attr:`swa_start`,
            :attr:`swa_freq` or :attr:`swa_lr`. In automatic mode SWA uses the
            same :attr:`swa_start`, :attr:`swa_freq` and :attr:`swa_lr` for all
            parameter groups. If needed, use manual mode with
            :meth:`update_swa_group` to use different update schedules for
            different parameter groups.
        .. note::
            Call :meth:`swap_swa_sgd` in the end of training to use the computed
            running averages.
        .. note::
            If you are using SWA to optimize the parameters of a Neural Network
            containing Batch Normalization layers, you need to update the
            :attr:`running_mean` and :attr:`running_var` statistics of the
            Batch Normalization module. You can do so by using
            `torchcontrib.optim.swa.bn_update` utility.
        .. note::
            See the blogpost
            https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
            for an extended description of this SWA implementation.
        .. note::
            The repo https://github.com/izmailovpavel/contrib_swa_examples
            contains examples of using this SWA implementation.
        .. _Averaging Weights Leads to Wider Optima and Better Generalization:
            https://arxiv.org/abs/1803.05407
        .. _Improving Consistency-Based Semi-Supervised Learning with Weight
            Averaging:
            https://arxiv.org/abs/1806.05594
        """
        self._auto_mode, (self.swa_freq,) = self._check_params(swa_freq)
        self.swa_lr_factor = swa_lr_factor

        if self._auto_mode:
            if swa_freq < 1:
                raise ValueError("Invalid swa_freq: {}".format(swa_freq))
        else:
            if self.swa_lr_factor is not None:
                warnings.warn(
                    "Swa_freq is None, ignoring swa_lr")
            # If not in auto mode make all swa parameters None
            self.swa_lr_factor = None
            self.swa_freq = None

        if self.swa_lr_factor is not None and self.swa_lr_factor < 0:
            raise ValueError("Invalid SWA learning rate factor: {}".format(swa_lr_factor))

        self.optimizer = optimizer

        self.defaults = self.optimizer.defaults
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.opt_state = self.optimizer.state
        for group in self.param_groups:
            group['n_avg'] = 0
            group['step_counter'] = 0

    @staticmethod
    def _check_params(swa_freq):
        params = [swa_freq]
        params_none = [param is None for param in params]
        if not all(params_none) and any(params_none):
            warnings.warn(
                "Some of swa_start, swa_freq is None, ignoring other")
        for i, param in enumerate(params):
            if param is not None and not isinstance(param, int):
                params[i] = int(param)
                warnings.warn("Casting swa_start, swa_freq to int")
        return not any(params_none), params

    def reset_lr_to_swa(self):
        for param_group in self.param_groups:
            param_group['initial_lr'] = self.swa_lr_factor * param_group['lr']

    def update_swa_group(self, group):
        r"""Updates the SWA running averages for the given parameter group.
        Arguments:
            group (dict): Specifies for what parameter group SWA running
                averages should be updated
        Examples:
            >>> # automatic mode
            >>> base_opt = torch.optim.SGD([{'params': [x]},
            >>>             {'params': [y], 'lr': 1e-3}], lr=1e-2, momentum=0.9)
            >>> opt = torchcontrib.optim.SWA(base_opt)
            >>> for i in range(100):
            >>>     opt.zero_grad()
            >>>     loss_fn(model(input), target).backward()
            >>>     opt.step()
            >>>     if i > 10 and i % 5 == 0:
            >>>         # Update SWA for the second parameter group
            >>>         opt.update_swa_group(opt.param_groups[1])
            >>> opt.swap_swa_param()
        """
        for p in group['params']:
            param_state = self.state[p]
            if 'swa_buffer' not in param_state:
                param_state['swa_buffer'] = torch.zeros_like(p.data)
            buf = param_state['swa_buffer']
            virtual_decay = 1 / float(group["n_avg"] + 1)
            diff = (p.data - buf) * virtual_decay
            buf.add_(diff)
        group["n_avg"] += 1

    def update_swa(self):
        r"""Updates the SWA running averages of all optimized parameters.
        """
        for group in self.param_groups:
            self.update_swa_group(group)

    def swap_swa_param(self):
        r"""Swaps the values of the optimized variables and swa buffers.
        It's meant to be called in the end of training to use the collected
        swa running averages. It can also be used to evaluate the running
        averages during training; to continue training `swap_swa_sgd`
        should be called again.
        """
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'swa_buffer' not in param_state:
                    # If swa wasn't applied we don't swap params
                    warnings.warn(
                        "SWA wasn't applied to param {}; skipping it".format(p))
                    continue
                buf = param_state['swa_buffer']
                tmp = torch.empty_like(p.data)
                tmp.copy_(p.data)
                p.data.copy_(buf)
                buf.copy_(tmp)

    def step(self, closure=None):
        r"""Performs a single optimization step.
        In automatic mode also updates SWA running averages.
        """
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            group["step_counter"] += 1
            steps = group["step_counter"]
            if self._auto_mode:
                if steps % self.swa_freq == 0:
                    self.update_swa_group(group)
        return loss

    def state_dict(self):
        r"""Returns the state of SWA as a :class:`dict`.
        It contains three entries:
            * opt_state - a dict holding current optimization state of the base
                optimizer. Its content differs between optimizer classes.
            * swa_state - a dict containing current state of SWA. For each
                optimized variable it contains swa_buffer keeping the running
                average of the variable
            * param_groups - a dict containing all parameter groups
        """
        opt_state_dict = self.optimizer.state_dict()
        swa_state = {(id(k) if isinstance(k, torch.Tensor) else k): v
                     for k, v in self.state.items()}
        opt_state = opt_state_dict["state"]
        param_groups = opt_state_dict["param_groups"]
        return {"opt_state": opt_state, "swa_state": swa_state,
                "param_groups": param_groups}

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.
        Args:
            state_dict (dict): SWA optimizer state. Should be an object returned
                from a call to `state_dict`.
        """
        swa_state_dict = {"state": state_dict["swa_state"],
                          "param_groups": state_dict["param_groups"]}
        opt_state_dict = {"state": state_dict["opt_state"],
                          "param_groups": state_dict["param_groups"]}
        super(SWA, self).load_state_dict(swa_state_dict)
        self.optimizer.load_state_dict(opt_state_dict)
        self.opt_state = self.optimizer.state

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.
        This can be useful when fine tuning a pre-trained network as frozen
        layers can be made trainable and added to the :class:`Optimizer` as
        training progresses.
        Args:
            param_group (dict): Specifies what Tensors should be optimized along
            with group specific optimization options.
        """
        param_group['n_avg'] = 0
        param_group['step_counter'] = 0
        self.optimizer.add_param_group(param_group)
