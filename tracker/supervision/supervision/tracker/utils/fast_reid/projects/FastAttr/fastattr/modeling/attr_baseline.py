# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from supervision.tracker.utils.fast_reid.fastreid.modeling.meta_arch.baseline import Baseline
from supervision.tracker.utils.fast_reid.fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY
from .bce_loss import cross_entropy_sigmoid_loss


@META_ARCH_REGISTRY.register()
class AttrBaseline(Baseline):

    @classmethod
    def from_config(cls, cfg):
        base_res = Baseline.from_config(cfg)
        base_res["loss_kwargs"].update({
            'bce': {
                'scale': cfg.MODEL.LOSSES.BCE.SCALE
            }
        })
        return base_res

    def losses(self, outputs, gt_labels):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        cls_outputs = outputs["cls_outputs"]

        loss_dict = {}
        loss_names = self.loss_kwargs["loss_names"]

        if "BinaryCrossEntropyLoss" in loss_names:
            bce_kwargs = self.loss_kwargs.get('bce')
            loss_dict["loss_bce"] = cross_entropy_sigmoid_loss(
                cls_outputs,
                gt_labels,
                self.sample_weights,
            ) * bce_kwargs.get('scale')

        return loss_dict
