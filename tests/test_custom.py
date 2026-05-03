# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import DraxNet


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "ultralytics" / "cfg" / "models" / "ext" / "draxnet-yolo26.yaml"


def test_draxnet_yolo26_build_and_forward():
    """Build the custom DraxNet YOLO26 model and run a minimal forward pass."""
    model = YOLO(CFG)
    outputs = model.model(torch.randn(1, 3, 64, 64))

    assert model.model.model[0].__class__.__name__ == "DraxNet"
    assert isinstance(outputs, dict)
    assert {"one2many", "one2one"} <= set(outputs)


def test_draxnet_backbone_feature_shapes():
    """Validate the standalone DraxNet backbone P3/P4/P5 feature shapes."""
    backbone = DraxNet(3)
    features = backbone(torch.randn(1, 3, 64, 64))

    assert len(features) == 3
    assert [tuple(x.shape) for x in features] == [(1, 256, 8, 8), (1, 512, 4, 4), (1, 1024, 2, 2)]
