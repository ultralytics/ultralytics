from .head import Detect
import torch.nn as nn
import torch

class AnchorFreeHead(Detect):
    def __init__(self, ch, nc):
        super().__init__(nc, ch=(ch,))

        self.nc = nc

        self.bbox = nn.Conv2d(
            ch,
            4,
            kernel_size=1
        )

        self.cls = nn.Conv2d(
            ch,
            nc,
            kernel_size=1
        )

        self.center = nn.Conv2d(
            ch,
            1,
            kernel_size=1
        )
    def forward(self, x):
        bbox = self.bbox(x)
        cls = self.cls(x)
        center = self.center(x)

        return torch.cat(
            [
                bbox,
                cls,
                center
            ],
            dim=1
        )
