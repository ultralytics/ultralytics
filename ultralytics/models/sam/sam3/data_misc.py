# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch

MyTensor = Union[torch.Tensor, List[Any]]


@dataclass
class FindStage:
    img_ids: MyTensor
    img_ids__type = torch.long
    text_ids: MyTensor
    text_ids__type = torch.long


@dataclass
class Datapoint:
    img_batch: torch.Tensor
    find_text_batch: List[str]
    find_inputs: FindStage
