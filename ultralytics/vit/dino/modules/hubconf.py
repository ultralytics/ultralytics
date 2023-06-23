# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import vision_transformer as vits
from torchvision.models.resnet import resnet50

dependencies = ['torch', 'torchvision']


def dino_vits16(pretrained=True, **kwargs):
    """
    ViT-Small/16x16 pre-trained with DINO.
    Achieves 74.5% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vits.__dict__['vit_small'](patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth',
            map_location='cpu',
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_vits8(pretrained=True, **kwargs):
    """
    ViT-Small/8x8 pre-trained with DINO.
    Achieves 78.3% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vits.__dict__['vit_small'](patch_size=8, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth',
            map_location='cpu',
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_vitb16(pretrained=True, **kwargs):
    """
    ViT-Base/16x16 pre-trained with DINO.
    Achieves 76.1% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vits.__dict__['vit_base'](patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth',
            map_location='cpu',
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_vitb8(pretrained=True, **kwargs):
    """
    ViT-Base/8x8 pre-trained with DINO.
    Achieves 77.4% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vits.__dict__['vit_base'](patch_size=8, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth',
            map_location='cpu',
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_resnet50(pretrained=True, **kwargs):
    """
    ResNet-50 pre-trained with DINO.
    Achieves 75.3% top-1 accuracy on ImageNet linear evaluation benchmark (requires to train `fc`).
    """
    model = resnet50(pretrained=False, **kwargs)
    model.fc = torch.nn.Identity()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth',
            map_location='cpu',
        )
        model.load_state_dict(state_dict, strict=False)
    return model


def dino_xcit_small_12_p16(pretrained=True, **kwargs):
    """
    XCiT-Small-12/16 pre-trained with DINO.
    """
    model = torch.hub.load('facebookresearch/xcit:main', 'xcit_small_12_p16', num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=
            'https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth',
            map_location='cpu',
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_xcit_small_12_p8(pretrained=True, **kwargs):
    """
    XCiT-Small-12/8 pre-trained with DINO.
    """
    model = torch.hub.load('facebookresearch/xcit:main', 'xcit_small_12_p8', num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth',
            map_location='cpu',
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_xcit_medium_24_p16(pretrained=True, **kwargs):
    """
    XCiT-Medium-24/16 pre-trained with DINO.
    """
    model = torch.hub.load('facebookresearch/xcit:main', 'xcit_medium_24_p16', num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=
            'https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth',
            map_location='cpu',
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_xcit_medium_24_p8(pretrained=True, **kwargs):
    """
    XCiT-Medium-24/8 pre-trained with DINO.
    """
    model = torch.hub.load('facebookresearch/xcit:main', 'xcit_medium_24_p8', num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=
            'https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth',
            map_location='cpu',
        )
        model.load_state_dict(state_dict, strict=True)
    return model
