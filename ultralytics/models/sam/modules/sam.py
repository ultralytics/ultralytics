# Ultralytics YOLO 🚀, AGPL-3.0 license

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from torch import nn

from .decoders import MaskDecoder
from .encoders import ImageEncoderViT, PromptEncoder


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = 'RGB'

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = (123.675, 116.28, 103.53),
        pixel_std: List[float] = (58.395, 57.12, 57.375)
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Note:
            All forward() operations moved to SAMPredictor.

        Args:
          image_encoder (ImageEncoderViT): The backbone used to encode the image into image embeddings that allow for
            efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(-1, 1, 1), False)
