# https://www.researchgate.net/figure/YOLOv5-architecture-The-YOLO-network-consists-of-three-main-parts-Backbone-Neck-and_fig5_355962110
import time
import torch
import torch.nn as nn
from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode
import config
from utils.utils import check_size, count_parameters
import torchsummary


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


class YOLOV5m(nn.Module):
    def __init__(self, first_out, nc=80, anchors=(),
                 ch=(), inference=False):
        super(YOLOV5m, self).__init__()
        self.inference = inference
        self.backbone = nn.ModuleList()
        self.backbone += [
            CBL(in_channels=3, out_channels=first_out, kernel_size=6, stride=2, padding=2),
            CBL(in_channels=first_out, out_channels=first_out*2, kernel_size=3, stride=2, padding=1),
            C3(in_channels=first_out*2, out_channels=first_out*2, width_multiple=0.5, depth=2),
            CBL(in_channels=first_out*2, out_channels=first_out*4, kernel_size=3, stride=2, padding=1),
            C3(in_channels=first_out*4, out_channels=first_out*4, width_multiple=0.5, depth=4),
            CBL(in_channels=first_out*4, out_channels=first_out*8, kernel_size=3, stride=2, padding=1),
            C3(in_channels=first_out*8, out_channels=first_out*8, width_multiple=0.5, depth=6),
            CBL(in_channels=first_out*8, out_channels=first_out*16, kernel_size=3, stride=2, padding=1),
            C3(in_channels=first_out*16, out_channels=first_out*16, width_multiple=0.5, depth=2),
            SPPF(in_channels=first_out*16, out_channels=first_out*16)
        ]

        self.neck = nn.ModuleList()
        self.neck += [
            CBL(in_channels=first_out*16, out_channels=first_out*8, kernel_size=1, stride=1, padding=0),
            C3(in_channels=first_out*16, out_channels=first_out*8, width_multiple=0.25, depth=2, backbone=False),
            CBL(in_channels=first_out*8, out_channels=first_out*4, kernel_size=1, stride=1, padding=0),
            C3(in_channels=first_out*8, out_channels=first_out*4, width_multiple=0.25, depth=2, backbone=False),
            CBL(in_channels=first_out*4, out_channels=first_out*4, kernel_size=3, stride=2, padding=1),
            C3(in_channels=first_out*8, out_channels=first_out*8, width_multiple=0.5, depth=2, backbone=False),
            CBL(in_channels=first_out*8, out_channels=first_out*8, kernel_size=3, stride=2, padding=1),
            C3(in_channels=first_out*16, out_channels=first_out*16, width_multiple=0.5, depth=2, backbone=False)
        ]
        self.head = HEADS(nc=nc, anchors=anchors, ch=ch)

    def forward(self, x):
        assert x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0, "Width and Height aren't divisible by 32!"
        backbone_connection = []
        neck_connection = []
        outputs = []
        for idx, layer in enumerate(self.backbone):
            # takes the out of the 2nd and 3rd C3 block and stores it
            x = layer(x)
            if idx in [4, 6]:
                backbone_connection.append(x)

        for idx, layer in enumerate(self.neck):
            if idx in [0, 2]:
                x = layer(x)
                neck_connection.append(x)
                x = Resize([x.shape[2]*2, x.shape[3]*2], interpolation=InterpolationMode.NEAREST)(x)
                x = torch.cat([x, backbone_connection.pop(-1)], dim=1)

            elif idx in [4, 6]:
                x = layer(x)
                x = torch.cat([x, neck_connection.pop(-1)], dim=1)

            elif (isinstance(layer, C3_NECK) and idx > 2) or (isinstance(layer, C3) and idx > 2):
                x = layer(x)
                outputs.append(x)

            else:
                x = layer(x)
                
        return self.head(outputs)


if __name__ == "__main__":
    batch_size = 2
    image_height = 640
    image_width = 640
    nc = 80
    anchors = config.ANCHORS
    x = torch.rand(batch_size, 3, image_height, image_width)
    first_out = 48

    model = YOLOV5m(first_out=first_out, nc=nc, anchors=anchors,
                    ch=(first_out*4, first_out*8, first_out*16), inference=False)

    start = time.time()
    out = model(x)
    end = time.time()

    assert out[0].shape == (batch_size, 3, image_height//8, image_width//8, nc + 5)
    assert out[1].shape == (batch_size, 3, image_height//16, image_width//16, nc + 5)
    assert out[2].shape == (batch_size, 3, image_height//32, image_width//32, nc + 5)

    print("Success!")
    print("feedforward took {:.2f} seconds".format(end - start))

    count_parameters(model)
    check_size(model)
    model.half()
    check_size(model)


