# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch.nn as nn

from .anchor_free import AnchorFreeHead
from .head import Detect


class HybridHead(nn.Module):
    def __init__(self, nc, ch):
        super().__init__()

        self.anchor_head = Detect(nc, ch=ch)

        self.af_head = nn.ModuleList([AnchorFreeHead(c, nc) for c in ch])

    def forward(self, x):
        anchor_output = self.anchor_head(x)
        if self.training:
            # Training: return combined dict so DetectionModel.loss() can extract anchor preds
            af_output = []
            for i, f in enumerate(x):
                af_output.append(self.af_head[i](f))
            return {"anchor": anchor_output, "anchor_free": af_output}
        else:
            # Inference: return anchor output directly (same format as Detect: (decoded_tensor, preds_dict))
            # This ensures validator/NMS receives the tensor it expects
            return anchor_output

    # --- Properties that delegate to the inner Detect head ---
    # Required by DetectionModel stride init and v8DetectionLoss
    @property
    def stride(self):
        return self.anchor_head.stride

    @stride.setter
    def stride(self, value):
        self.anchor_head.stride = value

    @property
    def nc(self):
        return self.anchor_head.nc

    @property
    def reg_max(self):
        return self.anchor_head.reg_max

    @property
    def no(self):
        return self.anchor_head.no

    @property
    def inplace(self):
        return self.anchor_head.inplace

    @inplace.setter
    def inplace(self, value):
        self.anchor_head.inplace = value

    def bias_init(self):
        """Initialize biases of the inner Detect head."""
        self.anchor_head.bias_init()
