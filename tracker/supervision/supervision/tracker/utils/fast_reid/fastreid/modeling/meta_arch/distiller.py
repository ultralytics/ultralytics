# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn.functional as F

from supervision.tracker.utils.fast_reid.fastreid.config import get_cfg
from supervision.tracker.utils.fast_reid.fastreid.modeling.meta_arch import META_ARCH_REGISTRY, build_model, Baseline
from supervision.tracker.utils.fast_reid.fastreid.utils.checkpoint import Checkpointer

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class Distiller(Baseline):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Get teacher model config
        model_ts = []
        for i in range(len(cfg.KD.MODEL_CONFIG)):
            cfg_t = get_cfg()
            cfg_t.merge_from_file(cfg.KD.MODEL_CONFIG[i])
            cfg_t.defrost()
            cfg_t.MODEL.META_ARCHITECTURE = "Baseline"
            # Change syncBN to BN due to no DDP wrapper
            if cfg_t.MODEL.BACKBONE.NORM == "syncBN":
                cfg_t.MODEL.BACKBONE.NORM = "BN"
            if cfg_t.MODEL.HEADS.NORM == "syncBN":
                cfg_t.MODEL.HEADS.NORM = "BN"

            model_t = build_model(cfg_t)

            # No gradients for teacher model
            for param in model_t.parameters():
                param.requires_grad_(False)

            logger.info("Loading teacher model weights ...")
            Checkpointer(model_t).load(cfg.KD.MODEL_WEIGHTS[i])

            model_ts.append(model_t)

        self.ema_enabled = cfg.KD.EMA.ENABLED
        self.ema_momentum = cfg.KD.EMA.MOMENTUM
        if self.ema_enabled:
            cfg_self = cfg.clone()
            cfg_self.defrost()
            cfg_self.MODEL.META_ARCHITECTURE = "Baseline"
            if cfg_self.MODEL.BACKBONE.NORM == "syncBN":
                cfg_self.MODEL.BACKBONE.NORM = "BN"
            if cfg_self.MODEL.HEADS.NORM == "syncBN":
                cfg_self.MODEL.HEADS.NORM = "BN"
            model_self = build_model(cfg_self)
            # No gradients for self model
            for param in model_self.parameters():
                param.requires_grad_(False)

            if cfg_self.MODEL.WEIGHTS != '':
                logger.info("Loading self distillation model weights ...")
                Checkpointer(model_self).load(cfg_self.MODEL.WEIGHTS)
            else:
                # Make sure the initial state is same
                for param_q, param_k in zip(self.parameters(), model_self.parameters()):
                    param_k.data.copy_(param_q.data)

            model_ts.insert(0, model_self)

        # Not register teacher model as `nn.Module`, this is
        # make sure teacher model weights not saved
        self.model_ts = model_ts

    @torch.no_grad()
    def _momentum_update_key_encoder(self, m=0.999):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.parameters(), self.model_ts[0].parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    def forward(self, batched_inputs):
        if self.training:
            images = self.preprocess_image(batched_inputs)
            # student model forward
            s_feat = self.backbone(images)
            assert "targets" in batched_inputs, "Labels are missing in training!"
            targets = batched_inputs["targets"].to(self.device)

            if targets.sum() < 0: targets.zero_()

            s_outputs = self.heads(s_feat, targets)

            t_outputs = []
            # teacher model forward
            with torch.no_grad():
                if self.ema_enabled:
                    self._momentum_update_key_encoder(self.ema_momentum)  # update self distill model
                for model_t in self.model_ts:
                    t_feat = model_t.backbone(images)
                    t_output = model_t.heads(t_feat, targets)
                    t_outputs.append(t_output)

            losses = self.losses(s_outputs, t_outputs, targets)
            return losses

        # Eval mode, just conventional reid feature extraction
        else:
            return super().forward(batched_inputs)

    def losses(self, s_outputs, t_outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        loss_dict = super().losses(s_outputs, gt_labels)

        s_logits = s_outputs['pred_class_logits']
        loss_jsdiv = 0.
        for t_output in t_outputs:
            t_logits = t_output['pred_class_logits'].detach()
            loss_jsdiv += self.jsdiv_loss(s_logits, t_logits)

        loss_dict["loss_jsdiv"] = loss_jsdiv / len(t_outputs)

        return loss_dict

    @staticmethod
    def _kldiv(y_s, y_t, t):
        p_s = F.log_softmax(y_s / t, dim=1)
        p_t = F.softmax(y_t / t, dim=1)
        loss = F.kl_div(p_s, p_t, reduction="sum") * (t ** 2) / y_s.shape[0]
        return loss

    def jsdiv_loss(self, y_s, y_t, t=16):
        loss = (self._kldiv(y_s, y_t, t) + self._kldiv(y_t, y_s, t)) / 2
        return loss
