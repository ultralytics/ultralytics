# https://www.researchgate.net/figure/YOLOv5-architecture-The-YOLO-network-consists-of-three-main-parts-Backbone-Neck-and_fig5_355962110
import torch
import torch.nn as nn

# performs a convolution, a batch_norm and then applies a SiLU activation function
class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CBL, self).__init__()

        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)

        self.cbl = nn.Sequential(
            conv,
            bn,
            # https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        #print(self.cbl(x).shape)
        return self.cbl(x)


# which is just a residual block
class Bottleneck(nn.Module):
    """
    Parameters:
        in_channels (int): number of channel of the input tensor
        out_channels (int): number of channel of the output tensor
        width_multiple (float): it controls the number of channels (and weights)
                                of all the convolutions beside the
                                first and last one. If closer to 0,
                                the simpler the modelIf closer to 1,
                                the model becomes more complex
    """
    def __init__(self, in_channels, out_channels, width_multiple=1):
        super(Bottleneck, self).__init__()
        c_ = int(width_multiple*in_channels)
        self.c1 = CBL(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c2 = CBL(c_, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.c2(self.c1(x)) + x


# kind of CSP backbone (https://arxiv.org/pdf/1911.11929v1.pdf)
class C3(nn.Module):
    """
    Parameters:
        in_channels (int): number of channel of the input tensor
        out_channels (int): number of channel of the output tensor
        width_multiple (float): it controls the number of channels (and weights)
                                of all the convolutions beside the
                                first and last one. If closer to 0,
                                the simpler the modelIf closer to 1,
                                the model becomes more complex
        depth (int): it controls the number of times the bottleneck (residual block)
                        is repeated within the C3 block
        backbone (bool): if True, self.seq will be composed by bottlenecks 1, if False
                            it will be composed by bottlenecks 2 (check in the image linked below)
        https://user-images.githubusercontent.com/31005897/172404576-c260dcf9-76bb-4bc8-b6a9-f2d987792583.png

    """
    def __init__(self, in_channels, out_channels, width_multiple=1, depth=1, backbone=True):
        super(C3, self).__init__()
        c_ = int(width_multiple*in_channels)

        self.c1 = CBL(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c_skipped = CBL(in_channels,  c_, kernel_size=1, stride=1, padding=0)
        if backbone:
            self.seq = nn.Sequential(
                *[Bottleneck(c_, c_, width_multiple=1) for _ in range(depth)]
            )
        else:
            self.seq = nn.Sequential(
                *[nn.Sequential(
                    CBL(c_, c_, 1, 1, 0),
                    CBL(c_, c_, 3, 1, 1)
                ) for _ in range(depth)]
            )
        self.c_out = CBL(c_ * 2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = torch.cat([self.seq(self.c1(x)), self.c_skipped(x)], dim=1)
        return self.c_out(x)


# Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPF, self).__init__()

        c_ = int(in_channels//2)

        self.c1 = CBL(in_channels, c_, 1, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.c_out = CBL(c_ * 4, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.c1(x)
        pool1 = self.pool(x)
        pool2 = self.pool(pool1)
        pool3 = self.pool(pool2)

        return self.c_out(torch.cat([x, pool1, pool2, pool3], dim=1))


# in the PANET the C3 block is different: no more CSP but a residual block composed
# a sequential branch of n SiLUs and a skipped branch with one SiLU
class C3_NECK(nn.Module):
    def __init__(self, in_channels, out_channels, width, depth):
        super(C3_NECK, self).__init__()
        c_ = int(in_channels*width)
        self.in_channels = in_channels
        self.c_ = c_
        self.out_channels = out_channels
        self.c_skipped = CBL(in_channels, c_, 1, 1, 0)
        self.c_out = CBL(c_*2, out_channels, 1, 1, 0)
        self.silu_block = self.make_silu_block(depth)

    def make_silu_block(self, depth):
        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(CBL(self.in_channels, self.c_, 1, 1, 0))
            elif i % 2 == 0:
                layers.append(CBL(self.c_, self.c_, 3, 1, 1))
            elif i % 2 != 0:
                layers.append(CBL(self.c_, self.c_, 1, 1, 0))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.c_out(torch.cat([self.silu_block(x), self.c_skipped(x)], dim=1))


class HEADS(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(HEADS, self).__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.naxs = len(anchors[0])

        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html command+f register_buffer
        # has the same result as self.anchors = anchors but, it's a way to register a buffer (make
        # a variable available in runtime) that should not be considered a model parameter
        self.stride = [8, 16, 32]

        # anchors are divided by the stride (anchors_for_head_1/8, anchors_for_head_1/16 etc.)
        anchors_ = torch.tensor(anchors).float().view(self.nl, -1, 2) / torch.tensor(self.stride).repeat(6, 1).T.reshape(3, 3, 2)
        self.register_buffer('anchors', anchors_)  # shape(nl,na,2)

        self.out_convs = nn.ModuleList()
        for in_channels in ch:
            self.out_convs += [
                nn.Conv2d(in_channels=in_channels, out_channels=(5+self.nc) * self.naxs, kernel_size=1)
            ]

    def forward(self, x):
        for i in range(self.nl):
            # performs out_convolution and stores the result in place
            x[i] = self.out_convs[i](x[i])

            bs, _, grid_y, grid_x = x[i].shape
            # reshaping output to be (bs, n_scale_predictions, n_grid_y, n_grid_x, 5 + num_classes)
            # why .permute? Here https://github.com/ultralytics/yolov5/issues/10524#issuecomment-1356822063
            x[i] = x[i].view(bs, self.naxs, (5+self.nc), grid_y, grid_x).permute(0, 1, 3, 4, 2).contiguous()

        return x
