# encoding: utf-8
"""
@author:  lingxiao he
@contact: helingxiao3@jd.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from supervision.tracker.utils.fast_reid.fastreid.layers import *
from supervision.tracker.utils.fast_reid.fastreid.modeling.heads import EmbeddingHead
from supervision.tracker.utils.fast_reid.fastreid.modeling.heads.build import REID_HEADS_REGISTRY
from supervision.tracker.utils.fast_reid.fastreid.layers.weight_init import weights_init_kaiming


class OcclusionUnit(nn.Module):
    def __init__(self, in_planes=2048):
        super(OcclusionUnit, self).__init__()
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=6, stride=2, padding=0)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=8, stride=2, padding=0)
        self.mask_layer = nn.Linear(in_planes, 1, bias=True)

    def forward(self, x):
        SpaFeat1 = self.MaxPool1(x)  # shape: [n, c, h, w]
        SpaFeat2 = self.MaxPool2(x)
        SpaFeat3 = self.MaxPool3(x)
        SpaFeat4 = self.MaxPool4(x)

        Feat1 = SpaFeat1.view(SpaFeat1.size(0), SpaFeat1.size(1), SpaFeat1.size(2) * SpaFeat1.size(3))
        Feat2 = SpaFeat2.view(SpaFeat2.size(0), SpaFeat2.size(1), SpaFeat2.size(2) * SpaFeat2.size(3))
        Feat3 = SpaFeat3.view(SpaFeat3.size(0), SpaFeat3.size(1), SpaFeat3.size(2) * SpaFeat3.size(3))
        Feat4 = SpaFeat4.view(SpaFeat4.size(0), SpaFeat4.size(1), SpaFeat4.size(2) * SpaFeat4.size(3))
        SpatialFeatAll = torch.cat((Feat1, Feat2, Feat3, Feat4), 2)
        SpatialFeatAll = SpatialFeatAll.transpose(1, 2)  # shape: [n, c, m]
        y = self.mask_layer(SpatialFeatAll)
        mask_weight = torch.sigmoid(y[:, :, 0])
        # mask_score = torch.sigmoid(mask_weight[:, :48])
        feat_dim = SpaFeat1.size(2) * SpaFeat1.size(3)
        mask_score = F.normalize(mask_weight[:, :feat_dim], p=1, dim=1)
        #       mask_score_norm = mask_score
        # mask_weight_norm = torch.sigmoid(mask_weight)
        mask_weight_norm = F.normalize(mask_weight, p=1, dim=1)

        mask_score = mask_score.unsqueeze(1)

        SpaFeat1 = SpaFeat1.transpose(1, 2)
        SpaFeat1 = SpaFeat1.transpose(2, 3)  # shape: [n, h, w, c]
        SpaFeat1 = SpaFeat1.view((SpaFeat1.size(0), SpaFeat1.size(1) * SpaFeat1.size(2), -1))  # shape: [n, h*w, c]

        global_feats = mask_score.matmul(SpaFeat1).view(SpaFeat1.shape[0], -1, 1, 1)
        return global_feats, mask_weight, mask_weight_norm


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


@REID_HEADS_REGISTRY.register()
class DSRHead(EmbeddingHead):
    def __init__(self, cfg):
        super().__init__(cfg)

        feat_dim = cfg.MODEL.BACKBONE.FEAT_DIM
        with_bnneck = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type = cfg.MODEL.HEADS.NORM
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        self.occ_unit = OcclusionUnit(in_planes=feat_dim)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=6, stride=2, padding=0)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=8, stride=2, padding=0)

        occ_neck = []
        if embedding_dim > 0:
            occ_neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            occ_neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bnneck_occ = nn.Sequential(*occ_neck)
        self.bnneck_occ.apply(weights_init_kaiming)

        self.weight_occ = nn.Parameter(torch.normal(0, 0.01, (num_classes, feat_dim)))

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        SpaFeat1 = self.MaxPool1(features)  # shape: [n, c, h, w]
        SpaFeat2 = self.MaxPool2(features)
        SpaFeat3 = self.MaxPool3(features)
        SpaFeat4 = self.MaxPool4(features)

        Feat1 = SpaFeat1.view(SpaFeat1.size(0), SpaFeat1.size(1), SpaFeat1.size(2) * SpaFeat1.size(3))
        Feat2 = SpaFeat2.view(SpaFeat2.size(0), SpaFeat2.size(1), SpaFeat2.size(2) * SpaFeat2.size(3))
        Feat3 = SpaFeat3.view(SpaFeat3.size(0), SpaFeat3.size(1), SpaFeat3.size(2) * SpaFeat3.size(3))
        Feat4 = SpaFeat4.view(SpaFeat4.size(0), SpaFeat4.size(1), SpaFeat4.size(2) * SpaFeat4.size(3))
        SpatialFeatAll = torch.cat((Feat1, Feat2, Feat3, Feat4), dim=2)

        foreground_feat, mask_weight, mask_weight_norm = self.occ_unit(features)
        # print(time.time() - st)
        bn_foreground_feat = self.bnneck_occ(foreground_feat)
        bn_foreground_feat = bn_foreground_feat[..., 0, 0]

        # Evaluation
        if not self.training:
            return bn_foreground_feat, SpatialFeatAll, mask_weight_norm

        # Training
        global_feat = self.pool_layer(features)
        bn_feat = self.bottleneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]

        if self.cls_layer.__class__.__name__ == 'Linear':
            pred_class_logits = F.linear(bn_feat, self.weight)
            fore_pred_class_logits = F.linear(bn_foreground_feat, self.weight_occ)
        else:
            pred_class_logits = F.linear(F.normalize(bn_feat), F.normalize(self.weight))
            fore_pred_class_logits = F.linear(F.normalize(bn_foreground_feat), F.normalize(self.weight_occ))

        cls_outputs = self.cls_layer(pred_class_logits, targets)
        fore_cls_outputs = self.cls_layer(fore_pred_class_logits, targets)

        # pdb.set_trace()
        return {
            "cls_outputs": cls_outputs,
            "fore_cls_outputs": fore_cls_outputs,
            "pred_class_logits": pred_class_logits * self.cls_layer.s,
            "features": global_feat[..., 0, 0],
            "foreground_features": foreground_feat[..., 0, 0],
        }
