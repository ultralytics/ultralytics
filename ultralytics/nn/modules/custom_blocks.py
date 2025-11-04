import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordAttConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, reduction=32):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

        # Coordinate Attention
        mip = max(8, c2 // reduction)
        self.conv1 = nn.Conv2d(c2, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv_h = nn.Conv2d(mip, c2, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, c2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.act(self.bn(self.conv(x)))
        n, c, h, w = y.size()

        # Split spatial pooling along height and width
        fh = F.adaptive_avg_pool2d(y, (h, 1))
        fw = F.adaptive_avg_pool2d(y, (1, w)).permute(0, 1, 3, 2)

        # Encode coordinate information
        f = torch.cat([fh, fw], dim=2)
        f = self.act(self.bn1(self.conv1(f)))

        fh, fw = torch.split(f, [h, w], dim=2)
        fw = fw.permute(0, 1, 3, 2)

        # Apply attention
        sh = self.sigmoid(self.conv_h(fh))
        sw = self.sigmoid(self.conv_w(fw))
        return y * sh * sw

