from .torch_utils import copy_attr
from ultralytics.nn.modules import Detect
from ultralytics.utils.tal import make_anchors
import torch.nn as nn
import torch
import types


class FXModel(nn.Module):
    """
    A custom model class for torch.fx compatibility.

    This class extends `torch.nn.Module` and is designed to ensure compatibility with torch.fx for tracing and graph
    manipulation. It copies attributes from an existing model and explicitly sets the model attribute to ensure proper
    copying.

    Attributes:
        model (nn.Module): The original model's layers.
    """

    def __init__(self, model):
        """
        Initialize the FXModel.

        Args:
            model (nn.Module): The original model to wrap for torch.fx compatibility.
        """
        super().__init__()
        copy_attr(self, model)
        # Explicitly set `model` since `copy_attr` somehow does not copy it.
        self.model = model.model

    def forward(self, x):
        """
        Forward pass through the model.

        This method performs the forward pass through the model, handling the dependencies between layers and saving
        intermediate outputs.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            (torch.Tensor): The output tensor from the model.
        """
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                # from earlier layers
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if isinstance(m, Detect):
                m.inference = types.MethodType(_inference, m)  # bind method to Detect
                m.anchors, m.strides = (
                    x.transpose(0, 1)
                    for x in make_anchors(
                        torch.cat([s / m.stride.unsqueeze(-1) for s in self.imgsz], dim=1), m.stride, 0.5
                    )
                )

            x = m(x)  # run
            y.append(x)  # save output
        return x


def _inference(self, x: list[torch.Tensor]) -> torch.Tensor:
    """
    Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

    Args:
        x (list[torch.Tensor]): List of feature maps from different detection layers.

    Returns:
        (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
    """
    x_cat = torch.cat([xi.view(x[0].shape[0], self.no, -1) for xi in x], 2)
    box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
    return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
