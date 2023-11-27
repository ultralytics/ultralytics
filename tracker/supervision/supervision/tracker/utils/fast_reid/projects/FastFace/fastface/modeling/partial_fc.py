# encoding: utf-8
# code based on:
# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/partial_fc.py

import logging
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from supervision.tracker.utils.fast_reid.fastreid.layers import any_softmax
from supervision.tracker.utils.fast_reid.fastreid.modeling.losses.utils import concat_all_gather
from supervision.tracker.utils.fast_reid.fastreid.utils import comm

logger = logging.getLogger('fastreid.partial_fc')


class PartialFC(nn.Module):
    """
    Author: {Xiang An, Yang Xiao, XuHan Zhu} in DeepGlint,
    Partial FC: Training 10 Million Identities on a Single Machine
    See the original paper:
    https://arxiv.org/abs/2010.05222
    """

    def __init__(
            self,
            embedding_size,
            num_classes,
            sample_rate,
            cls_type,
            scale,
            margin
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.sample_rate = sample_rate

        self.world_size = comm.get_world_size()
        self.rank = comm.get_rank()
        self.local_rank = comm.get_local_rank()
        self.device = torch.device(f'cuda:{self.local_rank}')

        self.num_local: int = self.num_classes // self.world_size + int(self.rank < self.num_classes % self.world_size)
        self.class_start: int = self.num_classes // self.world_size * self.rank + \
                                min(self.rank, self.num_classes % self.world_size)
        self.num_sample: int = int(self.sample_rate * self.num_local)

        self.cls_layer = getattr(any_softmax, cls_type)(num_classes, scale, margin)

        self.weight = torch.normal(0, 0.01, (self.num_local, self.embedding_size), device=self.device)
        self.weight_mom: torch.Tensor = torch.zeros_like(self.weight)
        logger.info("softmax weight init successfully!")
        logger.info("softmax weight mom init successfully!")
        self.stream: torch.cuda.Stream = torch.cuda.Stream(self.local_rank)

        self.index = None
        if int(self.sample_rate) == 1:
            self.update = lambda: 0
            self.sub_weight = nn.Parameter(self.weight)
            self.sub_weight_mom = self.weight_mom
        else:
            self.sub_weight = nn.Parameter(torch.empty((0, 0), device=self.device))

    def forward(self, total_features):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.cls_layer.__class__.__name__ == 'Linear':
            logits = F.linear(total_features, self.sub_weight)
        else:
            logits = F.linear(F.normalize(total_features), F.normalize(self.sub_weight))
        return logits

    def forward_backward(self, features, targets, optimizer):
        """
        Partial FC forward, which will sample positive weights and part of negative weights,
        then compute logits and get the grad of features.
        """
        total_targets = self.prepare(targets, optimizer)

        if self.world_size > 1:
            total_features = concat_all_gather(features)
        else:
            total_features = features.detach()

        total_features.requires_grad_(True)

        logits = self.forward(total_features)
        logits = self.cls_layer(logits, total_targets)

        # from ipdb import set_trace; set_trace()
        with torch.no_grad():
            max_fc = torch.max(logits, dim=1, keepdim=True)[0]
            if self.world_size > 1:
                dist.all_reduce(max_fc, dist.ReduceOp.MAX)

            # calculate exp(logits) and all-reduce
            logits_exp = torch.exp(logits - max_fc)
            logits_sum_exp = logits_exp.sum(dim=1, keepdim=True)

            if self.world_size > 1:
                dist.all_reduce(logits_sum_exp, dist.ReduceOp.SUM)

            # calculate prob
            logits_exp.div_(logits_sum_exp)

            # get one-hot
            grad = logits_exp
            index = torch.where(total_targets != -1)[0]
            one_hot = torch.zeros(size=[index.size()[0], grad.size()[1]], device=grad.device)
            one_hot.scatter_(1, total_targets[index, None], 1)

            # calculate loss
            loss = torch.zeros(grad.size()[0], 1, device=grad.device)
            loss[index] = grad[index].gather(1, total_targets[index, None])
            if self.world_size > 1:
                dist.all_reduce(loss, dist.ReduceOp.SUM)
            loss_v = loss.clamp_min_(1e-30).log_().mean() * (-1)

            # calculate grad
            grad[index] -= one_hot
            grad.div_(logits.size(0))

        logits.backward(grad)
        if total_features.grad is not None:
            total_features.grad.detach_()
        x_grad: torch.Tensor = torch.zeros_like(features)
        # feature gradient all-reduce
        if self.world_size > 1:
            dist.reduce_scatter(x_grad, list(total_features.grad.chunk(self.world_size, dim=0)))
        else:
            x_grad = total_features.grad
        x_grad = x_grad * self.world_size
        # backward backbone
        return x_grad, loss_v

    @torch.no_grad()
    def sample(self, total_targets):
        """
        Get sub_weights according to total targets gathered from all GPUs, due to each weights in different
        GPU contains different class centers.
        """
        index_positive = (self.class_start <= total_targets) & (total_targets < self.class_start + self.num_local)
        total_targets[~index_positive] = -1
        total_targets[index_positive] -= self.class_start
        if int(self.sample_rate) != 1:
            positive = torch.unique(total_targets[index_positive], sorted=True)
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(size=[self.num_local], device=self.weight.device)
                perm[positive] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1]
                index = index.sort()[0]
            else:
                index = positive
            self.index = index
            total_targets[index_positive] = torch.searchsorted(index, total_targets[index_positive])
            self.sub_weight = nn.Parameter(self.weight[index])
            self.sub_weight_mom = self.weight_mom[index]

    @torch.no_grad()
    def update(self):
        self.weight_mom[self.index] = self.sub_weight_mom
        self.weight[self.index] = self.sub_weight

    def prepare(self, targets, optimizer):
        with torch.cuda.stream(self.stream):
            if self.world_size > 1:
                total_targets = concat_all_gather(targets)
            else:
                total_targets = targets
            # update sub_weight
            self.sample(total_targets)
            optimizer.state.pop(optimizer.param_groups[-1]['params'][0], None)
            optimizer.param_groups[-1]['params'][0] = self.sub_weight
            optimizer.state[self.sub_weight]["momentum_buffer"] = self.sub_weight_mom
            return total_targets
