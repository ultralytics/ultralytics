# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.
"""

from dataclasses import dataclass
import torch


@dataclass
class Datapoint:
    img_batch: torch.Tensor
    img_ids: torch.Tensor
    text_ids: torch.Tensor
