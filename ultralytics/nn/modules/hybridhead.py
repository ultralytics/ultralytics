from .head import Detect
import torch.nn as nn
from .anchor_free import AnchorFreeHead


class HybridHead(nn.Module):
    def __init__(self, nc, ch):
        super().__init__()

        self.anchor_head = Detect(nc, ch=ch)

        self.af_head = nn.ModuleList(
            [
                AnchorFreeHead(
                    c,
                    nc
                )
                for c in ch
            ]
        )
    
    def forward(self, x):
        anchor_output = self.anchor_head(x)
        af_output = []
        for i, f in enumerate(x):
            af_output.append(
                self.af_head[i](f)
            )
        return {
            "anchor": anchor_output,
            "anchor_free": af_output
        }