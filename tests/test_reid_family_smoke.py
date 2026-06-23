# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Smoke test: every YOLO family x scale builds a working ReID model (build/forward/loss/step)."""

from __future__ import annotations

import pytest
import torch

from ultralytics.nn.modules.head import ReID
from ultralytics.nn.tasks import ReidModel

# Bare config names — resolved via Ultralytics' cfg search path (cfg/models/{v8,11,12,26}/).
CFGS = {
    "v8": "yolov8-reid.yaml",
    "v11": "yolo11-reid.yaml",
    "v12": "yolo12-reid.yaml",
    "v26": "yolo26-reid.yaml",
}
SCALES = ["n", "s", "m", "l", "x"]
NC = 100


def _scaled_cfg(family: str, scale: str) -> str:
    """Splice the scale letter into the stem so the scaled cfg resolves (e.g. yolo12l-reid.yaml)."""
    name = CFGS[family]  # e.g. 'yolo12-reid.yaml'
    base, sep, rest = name.partition("-reid")  # ('yolo12', '-reid', '.yaml')
    return f"{base}{scale}{sep}{rest}"  # 'yolo12l-reid.yaml'


def _build(family: str, scale: str) -> ReidModel:
    return ReidModel(cfg=_scaled_cfg(family, scale), nc=NC, verbose=False)


@pytest.mark.parametrize("family", list(CFGS))
@pytest.mark.parametrize("scale", SCALES)
def test_family_scale_train_forward_loss_step(family, scale):
    """Each variant: train-mode forward -> 3-tuple, ReIDLoss computes, one optimizer step succeeds."""
    model = _build(family, scale).train()
    imgs = torch.randn(8, 3, 256, 128)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])  # P=4, K=2 so triplet has positives
    preds = model(imgs)
    assert isinstance(preds, (list, tuple)) and len(preds) == 3, "train-mode head must return triple"
    logits, _bn_feat, _raw_feat = preds
    assert logits.shape == (8, NC)

    crit = model.init_criterion()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    opt.zero_grad()
    loss, _items = crit(preds, {"cls": labels})
    assert torch.isfinite(loss), "loss must be finite"
    loss.backward()
    opt.step()


@pytest.mark.parametrize("family", list(CFGS))
@pytest.mark.parametrize("scale", SCALES)
def test_family_scale_eval_embedding_is_unit_norm(family, scale):
    """Eval-mode forward returns an L2-normalized embedding of embed_dim per sample."""
    model = _build(family, scale).eval()
    head = model.model[-1]
    assert isinstance(head, ReID)
    imgs = torch.randn(4, 3, 256, 128)
    with torch.no_grad():
        out = model(imgs)
    emb = out[0] if isinstance(out, (list, tuple)) else out
    assert emb.shape == (4, head.embed_dim)
    assert torch.allclose(emb.norm(dim=1), torch.ones(4), atol=1e-4)


@pytest.mark.parametrize("family", list(CFGS))
def test_v_family_imgsz448_forward(family):
    """Imgsz=448 square forward works for the l scale (v12 A2C2f stride/area-attention sanity)."""
    model = _build(family, "l").eval()
    imgs = torch.randn(2, 3, 448, 448)
    with torch.no_grad():
        out = model(imgs)
    emb = out[0] if isinstance(out, (list, tuple)) else out
    assert emb.shape[0] == 2
