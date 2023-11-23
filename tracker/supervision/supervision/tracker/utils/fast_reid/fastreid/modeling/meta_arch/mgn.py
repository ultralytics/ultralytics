# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy

import torch
from torch import nn

from supervision.tracker.utils.fast_reid.fastreid.config import configurable
from supervision.tracker.utils.fast_reid.fastreid.layers import get_norm
from supervision.tracker.utils.fast_reid.fastreid.modeling.backbones import build_backbone
from supervision.tracker.utils.fast_reid.fastreid.modeling.backbones.resnet import Bottleneck
from supervision.tracker.utils.fast_reid.fastreid.modeling.heads import build_heads
from supervision.tracker.utils.fast_reid.fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class MGN(nn.Module):
    """
    Multiple Granularities Network architecture, which contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Multi-branch feature aggregation
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            neck1,
            neck2,
            neck3,
            b1_head,
            b2_head,
            b21_head,
            b22_head,
            b3_head,
            b31_head,
            b32_head,
            b33_head,
            pixel_mean,
            pixel_std,
            loss_kwargs=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            neck1:
            neck2:
            neck3:
            b1_head:
            b2_head:
            b21_head:
            b22_head:
            b3_head:
            b31_head:
            b32_head:
            b33_head:
            pixel_mean:
            pixel_std:
            loss_kwargs:
        """

        super().__init__()

        self.backbone = backbone

        # branch1
        self.b1 = neck1
        self.b1_head = b1_head

        # branch2
        self.b2 = neck2
        self.b2_head = b2_head
        self.b21_head = b21_head
        self.b22_head = b22_head

        # branch3
        self.b3 = neck3
        self.b3_head = b3_head
        self.b31_head = b31_head
        self.b32_head = b32_head
        self.b33_head = b33_head

        self.loss_kwargs = loss_kwargs
        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        bn_norm = cfg.MODEL.BACKBONE.NORM
        with_se = cfg.MODEL.BACKBONE.WITH_SE

        all_blocks = build_backbone(cfg)

        # backbone
        backbone = nn.Sequential(
            all_blocks.conv1,
            all_blocks.bn1,
            all_blocks.relu,
            all_blocks.maxpool,
            all_blocks.layer1,
            all_blocks.layer2,
            all_blocks.layer3[0]
        )
        res_conv4 = nn.Sequential(*all_blocks.layer3[1:])
        res_g_conv5 = all_blocks.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, bn_norm, False, with_se, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, 1, bias=False), get_norm(bn_norm, 2048))),
            Bottleneck(2048, 512, bn_norm, False, with_se),
            Bottleneck(2048, 512, bn_norm, False, with_se))
        res_p_conv5.load_state_dict(all_blocks.layer4.state_dict())

        # branch
        neck1 = nn.Sequential(
            copy.deepcopy(res_conv4),
            copy.deepcopy(res_g_conv5)
        )
        b1_head = build_heads(cfg)

        # branch2
        neck2 = nn.Sequential(
            copy.deepcopy(res_conv4),
            copy.deepcopy(res_p_conv5)
        )
        b2_head = build_heads(cfg)
        b21_head = build_heads(cfg)
        b22_head = build_heads(cfg)

        # branch3
        neck3 = nn.Sequential(
            copy.deepcopy(res_conv4),
            copy.deepcopy(res_p_conv5)
        )
        b3_head = build_heads(cfg)
        b31_head = build_heads(cfg)
        b32_head = build_heads(cfg)
        b33_head = build_heads(cfg)

        return {
            'backbone': backbone,
            'neck1': neck1,
            'neck2': neck2,
            'neck3': neck3,
            'b1_head': b1_head,
            'b2_head': b2_head,
            'b21_head': b21_head,
            'b22_head': b22_head,
            'b3_head': b3_head,
            'b31_head': b31_head,
            'b32_head': b32_head,
            'b33_head': b33_head,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    }
                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)  # (bs, 2048, 16, 8)

        # branch1
        b1_feat = self.b1(features)

        # branch2
        b2_feat = self.b2(features)
        b21_feat, b22_feat = torch.chunk(b2_feat, 2, dim=2)

        # branch3
        b3_feat = self.b3(features)
        b31_feat, b32_feat, b33_feat = torch.chunk(b3_feat, 3, dim=2)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            if targets.sum() < 0: targets.zero_()

            b1_outputs = self.b1_head(b1_feat, targets)
            b2_outputs = self.b2_head(b2_feat, targets)
            b21_outputs = self.b21_head(b21_feat, targets)
            b22_outputs = self.b22_head(b22_feat, targets)
            b3_outputs = self.b3_head(b3_feat, targets)
            b31_outputs = self.b31_head(b31_feat, targets)
            b32_outputs = self.b32_head(b32_feat, targets)
            b33_outputs = self.b33_head(b33_feat, targets)

            losses = self.losses(b1_outputs,
                                 b2_outputs, b21_outputs, b22_outputs,
                                 b3_outputs, b31_outputs, b32_outputs, b33_outputs,
                                 targets)
            return losses
        else:
            b1_pool_feat = self.b1_head(b1_feat)
            b2_pool_feat = self.b2_head(b2_feat)
            b21_pool_feat = self.b21_head(b21_feat)
            b22_pool_feat = self.b22_head(b22_feat)
            b3_pool_feat = self.b3_head(b3_feat)
            b31_pool_feat = self.b31_head(b31_feat)
            b32_pool_feat = self.b32_head(b32_feat)
            b33_pool_feat = self.b33_head(b33_feat)

            pred_feat = torch.cat([b1_pool_feat, b2_pool_feat, b3_pool_feat, b21_pool_feat,
                                   b22_pool_feat, b31_pool_feat, b32_pool_feat, b33_pool_feat], dim=1)
            return pred_feat

    def preprocess_image(self, batched_inputs):
        r"""
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self,
               b1_outputs,
               b2_outputs, b21_outputs, b22_outputs,
               b3_outputs, b31_outputs, b32_outputs, b33_outputs, gt_labels):
        # model predictions
        # fmt: off
        pred_class_logits = b1_outputs['pred_class_logits'].detach()
        b1_logits         = b1_outputs['cls_outputs']
        b2_logits         = b2_outputs['cls_outputs']
        b21_logits        = b21_outputs['cls_outputs']
        b22_logits        = b22_outputs['cls_outputs']
        b3_logits         = b3_outputs['cls_outputs']
        b31_logits        = b31_outputs['cls_outputs']
        b32_logits        = b32_outputs['cls_outputs']
        b33_logits        = b33_outputs['cls_outputs']
        b1_pool_feat      = b1_outputs['features']
        b2_pool_feat      = b2_outputs['features']
        b3_pool_feat      = b3_outputs['features']
        b21_pool_feat     = b21_outputs['features']
        b22_pool_feat     = b22_outputs['features']
        b31_pool_feat     = b31_outputs['features']
        b32_pool_feat     = b32_outputs['features']
        b33_pool_feat     = b33_outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        b22_pool_feat = torch.cat((b21_pool_feat, b22_pool_feat), dim=1)
        b33_pool_feat = torch.cat((b31_pool_feat, b32_pool_feat, b33_pool_feat), dim=1)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if "CrossEntropyLoss" in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls_b1'] = cross_entropy_loss(
                b1_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') * 0.125

            loss_dict['loss_cls_b2'] = cross_entropy_loss(
                b2_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') * 0.125

            loss_dict['loss_cls_b21'] = cross_entropy_loss(
                b21_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') * 0.125

            loss_dict['loss_cls_b22'] = cross_entropy_loss(
                b22_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') * 0.125

            loss_dict['loss_cls_b3'] = cross_entropy_loss(
                b3_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') * 0.125

            loss_dict['loss_cls_b31'] = cross_entropy_loss(
                b31_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') * 0.125

            loss_dict['loss_cls_b32'] = cross_entropy_loss(
                b32_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') * 0.125

            loss_dict['loss_cls_b33'] = cross_entropy_loss(
                b33_logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale') * 0.125

        if "TripletLoss" in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet_b1'] = triplet_loss(
                b1_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale') * 0.2

            loss_dict['loss_triplet_b2'] = triplet_loss(
                b2_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale') * 0.2

            loss_dict['loss_triplet_b3'] = triplet_loss(
                b3_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale') * 0.2

            loss_dict['loss_triplet_b22'] = triplet_loss(
                b22_pool_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale') * 0.2

            loss_dict['loss_triplet_b33'] = triplet_loss(
                b33_pool_feat,
                gt_labels,

                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale') * 0.2

        return loss_dict
