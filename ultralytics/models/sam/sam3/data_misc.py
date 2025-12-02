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

    input_boxes: MyTensor
    input_boxes__type = torch.float
    input_boxes_mask: MyTensor
    input_boxes_mask__type = torch.bool
    input_boxes_label: MyTensor
    input_boxes_label__type = torch.long

    input_points: MyTensor
    input_points__type = torch.float
    input_points_mask: MyTensor
    input_points_mask__type = torch.bool

    # We track the object ids referred to by this query.
    # This is beneficial for tracking in videos without the need for pointers.
    object_ids: Optional[List[List]] = None  # List of objects per query


@dataclass
class Datapoint:
    img_batch: torch.Tensor
    find_text_batch: List[str]
    find_inputs: FindStage
