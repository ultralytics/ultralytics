# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
from torch import nn

from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.head_pruned import DetectPruned


def test_pruned_head_uses_official_detect_contract():
    for nc in (1, 2):
        head = DetectPruned(
            cv2x0_outs=[8, 12, 16],
            cv2x1_outs=[8, 12, 16],
            cv3x0_outs=[8, 12, 16],
            cv3x1_outs=[8, 12, 16],
            nc=nc,
            ch=[8, 16, 32],
        )
        head.f = [15, 18, 21]
        head.i = 22
        head.train()

        outputs = head(
            [
                torch.randn(2, 8, 8, 8),
                torch.randn(2, 16, 4, 4),
                torch.randn(2, 32, 2, 2),
            ]
        )

        assert isinstance(head, Detect)
        assert outputs["boxes"].shape == (2, 64, 84)
        assert outputs["scores"].shape == (2, nc, 84)
        assert [tuple(x.shape) for x in outputs["feats"]] == [
            (2, 8, 8, 8),
            (2, 16, 4, 4),
            (2, 32, 2, 2),
        ]

        dummy = nn.Module()
        dummy.model = nn.ModuleList([nn.Identity() for _ in range(22)] + [head])
        assert DistillationModel.get_distill_layers(dummy) == [15, 18, 21, 22]
