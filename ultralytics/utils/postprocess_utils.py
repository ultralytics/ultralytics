import math

import torch

from ultralytics.nn.modules.block import DFL
from ultralytics.utils.tal import dist2bbox, make_anchors

def separate_outputs_decode(preds, task):
    mcv = float("-inf")
    lci = -1
    for idx, s in enumerate(preds):
        dim_1 = s.shape[1]
        if dim_1 > mcv:
            mcv = dim_1
            lci = idx
        if len(s.shape) == 4 and task == "segment":
            proto = s
            pidx = idx
    
    if task == "pose":
        return [item for index, item in enumerate(preds) if index not in [lci]], preds[lci]
    elif task == "segment":
        return [item for index, item in enumerate(preds) if index not in [pidx, lci]], preds[lci], proto.permute(0, 3, 1, 2)
    
def decode_bbox(preds, img_shape, device):
    num_classes = next((o.shape[2] for o in preds if o.shape[2] != 64), -1)
    assert num_classes != -1, 'cannot infer postprocessor inputs via output shape if there are 64 classes'
    pos = [
        i for i, _ in sorted(enumerate(preds),
                             key=lambda x: (x[1].shape[2] if num_classes > 64 else -x[1].shape[2], -x[1].shape[1]))]
    x = torch.permute(
        torch.cat([
            torch.cat([preds[i] for i in pos[:len(pos) // 2]], 1),
            torch.cat([preds[i] for i in pos[len(pos) // 2:]], 1)], 2), (0, 2, 1))
    reg_max = (x.shape[1] - num_classes) // 4
    dfl = DFL(reg_max) if reg_max > 1 else torch.nn.Identity()
    img_h, img_w = img_shape[-2], img_shape[-1]
    strides = [
        int(math.sqrt(img_shape[-2] * img_shape[-1] / preds[p].shape[1])) for p in pos if preds[p].shape[2] != 64]
    dims = [(img_h // s, img_w // s) for s in strides]
    fake_feats = [torch.zeros((1, 1, h, w), device=device) for h, w in dims]
    anchors, strides = (x.transpose(0, 1)
                        for x in make_anchors(fake_feats, strides, 0.5))  # generate anchors and strides
    dbox = dist2bbox(dfl(x[:, :-num_classes, :].cpu()).to(device), anchors.unsqueeze(0), xywh=True, dim=1) * strides
    return torch.cat((dbox, x[:, -num_classes:, :].sigmoid()), 1)


def decode_kpts(preds, img_shape, kpts, kpt_shape, device, bs=1):
    """Decodes keypoints."""
    num_classes = next((o.shape[2] for o in preds if o.shape[2] != 64), -1)
    assert num_classes != -1, 'cannot infer postprocessor inputs via output shape if there are 64 classes'
    pos = [
        i for i, _ in sorted(enumerate(preds),
                             key=lambda x: (x[1].shape[2] if num_classes > 64 else -x[1].shape[2], -x[1].shape[1]))]
    strides = [
        int(math.sqrt(img_shape[-2] * img_shape[-1] / preds[p].shape[1])) for p in pos if preds[p].shape[2] != 64]
    dims = [(img_shape[-2] // s, img_shape[-1] // s) for s in strides]
    fake_feats = [torch.zeros((1, 1, h, w), device=device) for h, w in dims]
    anchors, strides = (x.transpose(0, 1)
                        for x in make_anchors(fake_feats, strides, 0.5))  # generate anchors and strides
    ndim = kpt_shape[1]
    y = kpts.clone()
    if ndim == 3:
        y[:, 2::3].sigmoid_()  # inplace sigmoid
    y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (anchors[0] - 0.5)) * strides
    y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (anchors[1] - 0.5)) * strides
    return y
