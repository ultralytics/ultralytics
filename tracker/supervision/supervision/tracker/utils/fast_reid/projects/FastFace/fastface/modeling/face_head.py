# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

from supervision.tracker.utils.fast_reid.fastreid.config import configurable
from supervision.tracker.utils.fast_reid.fastreid.modeling.heads import EmbeddingHead
from supervision.tracker.utils.fast_reid.fastreid.modeling.heads.build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class FaceHead(EmbeddingHead):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.pfc_enabled = False
        if cfg.MODEL.HEADS.PFC.ENABLED:
            # Delete pre-defined linear weights for partial fc sample
            del self.weight
            self.pfc_enabled = True

    def forward(self, features, targets=None):
        """
        Partial FC forward, which will sample positive weights and part of negative weights,
        then compute logits and get the grad of features.
        """
        if not self.pfc_enabled:
            return super().forward(features, targets)
        else:
            pool_feat = self.pool_layer(features)
            neck_feat = self.bottleneck(pool_feat)
            neck_feat = neck_feat[..., 0, 0]
            return neck_feat
