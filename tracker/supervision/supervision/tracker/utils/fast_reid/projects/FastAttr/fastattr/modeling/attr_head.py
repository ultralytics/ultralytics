# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from supervision.tracker.utils.fast_reid.fastreid.modeling.heads import EmbeddingHead
from supervision.tracker.utils.fast_reid.fastreid.modeling.heads.build import REID_HEADS_REGISTRY
from supervision.tracker.utils.fast_reid.fastreid.layers.weight_init import weights_init_kaiming


@REID_HEADS_REGISTRY.register()
class AttrHead(EmbeddingHead):
    def __init__(self, cfg):
        super().__init__(cfg)
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES

        self.bnneck = nn.BatchNorm1d(num_classes)
        self.bnneck.apply(weights_init_kaiming)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        pool_feat = self.pool_layer(features)
        neck_feat = self.bottleneck(pool_feat)
        neck_feat = neck_feat.view(neck_feat.size(0), -1)

        logits = F.linear(neck_feat, self.weight)
        logits = self.bnneck(logits)

        # Evaluation
        if not self.training:
            cls_outptus = torch.sigmoid(logits)
            return cls_outptus

        return {
            "cls_outputs": logits,
        }
