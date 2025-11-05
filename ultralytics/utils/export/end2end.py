# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from types import MethodType

import torch

from ultralytics.nn.tasks import DetectionModel


def _forward_end2end_nopost(self, x: list[torch.Tensor]) -> dict | tuple:
    """Perform forward pass of the v10Detect module without postprocess.

    Args:
        x (list[torch.Tensor]): Input feature maps from different levels.

    Returns:
        outputs (dict | tuple): Returns decoded output of one2one head.
    """
    x_detach = [xi.detach() for xi in x]
    one2one = [
        torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
    ]
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    return self._inference(one2one)


def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80) -> torch.Tensor:
    """Post-process YOLO model predictions with support for extras.

    Args:
        preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + extras) with last dimension
            format [x, y, w, h, class_probs].
        max_det (int): Maximum detections per image.
        nc (int, optional): Number of classes.

    Returns:
        (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + extras) and last
            dimension format [x, y, w, h, max_class_prob, class_index, extras].
    """
    batch_size, anchors, np = preds.shape  # i.e. shape(16,8400,84)
    boxes, scores, extras = preds.split([4, nc, np - (4 + nc)], dim=-1)
    index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
    boxes = boxes.gather(dim=1, index=index.expand(-1, -1, 4))
    scores = scores.gather(dim=1, index=index.expand(-1, -1, nc))
    extras = extras.gather(dim=1, index=index.expand(-1, -1, extras.shape[-1]))
    scores, index = scores.flatten(1).topk(min(max_det, anchors))
    i, pi = torch.arange(batch_size)[..., None], index // nc  # batch indices, pred indices
    return torch.cat([boxes[i, pi], scores[..., None], (index % nc)[..., None].float(), extras[i, pi]], dim=-1)


def end2end_wrapper(model: DetectionModel) -> None:
    """Patch end2end forward pass to remove postprocessing.
    
    Args:
        model (DetectionModel): YOLO end2end model to patch.
    
    Returns:
        model (DetectionModel): Patched end2end model.
    """
    if getattr(model, "end2end"):
        model.model[-1].forward_end2end = MethodType(_forward_end2end_nopost, model.model[-1])
    return model
