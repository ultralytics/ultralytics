# encoding: utf-8
"""
@authorr:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from supervision.tracker.utils.fast_reid.fastreid.modeling.losses import *
from supervision.tracker.utils.fast_reid.fastreid.modeling.meta_arch import Baseline
from supervision.tracker.utils.fast_reid.fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class PartialBaseline(Baseline):

    def losses(self, outputs, gt_labels):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        loss_dict = super().losses(outputs, gt_labels)

        fore_cls_outputs = outputs["fore_cls_outputs"]
        fore_feat = outputs["foreground_features"]

        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_fore_cls'] = cross_entropy_loss(
                fore_cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_fore_triplet'] = triplet_loss(
                fore_feat,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

        return loss_dict
