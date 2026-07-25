# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the dense photometric left-right consistency loss (training.photometric_loss)."""

import math

import torch


def _stereo_pair_with_disparity(d_px: int, h: int = 64, w: int = 128):
    """Build a 6-ch stereo image where the right view is the left shifted by d_px (uniform disparity).

    Convention: u_R = u_L − d, so right content sits d px to the LEFT of its left-image position, i.e. right =
    roll(left, −d) along width.
    """
    g = torch.Generator().manual_seed(0)
    left = torch.rand(1, 3, h, w, generator=g)
    left = torch.nn.functional.avg_pool2d(left, 5, 1, 2)  # smooth → meaningful photometric gradients
    right = torch.roll(left, shifts=-d_px, dims=3)
    return torch.cat([left, right], dim=1)


def _lr_map(disp_px: float, w: int, h8: int, w8: int):
    """Dense lr_distance map (log normalized disparity) for the P3 grid, HW_total = P3 only."""
    return torch.full((1, 1, h8 * w8), math.log(max(disp_px / w, 1e-6)))


def test_photometric_loss_prefers_true_disparity():
    """Loss at the true disparity must beat clearly wrong disparities on a synthetic pair."""
    from ultralytics.models.yolo.s3d.loss import photometric_lr_loss

    d = 16
    imgs = _stereo_pair_with_disparity(d)
    h8, w8 = imgs.shape[2] // 8, imgs.shape[3] // 8
    losses = {
        cand: float(photometric_lr_loss(_lr_map(cand, imgs.shape[3], h8, w8), imgs)) for cand in (d, d // 2, d * 2, 1)
    }
    assert losses[d] < losses[d // 2]
    assert losses[d] < losses[d * 2]
    assert losses[d] < losses[1]


def test_photometric_loss_finite_at_extremes():
    """Huge disparity pushes samples out of bounds — masked, loss stays finite with grad."""
    from ultralytics.models.yolo.s3d.loss import photometric_lr_loss

    imgs = _stereo_pair_with_disparity(8)
    h8, w8 = imgs.shape[2] // 8, imgs.shape[3] // 8
    lr = _lr_map(imgs.shape[3] * 0.9, imgs.shape[3], h8, w8).requires_grad_(True)
    loss = photometric_lr_loss(lr, imgs)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(lr.grad).all()
    # Runaway positive log-disparity must stay finite too (upper clamp bounds exp, incl. fp16 range)
    assert torch.isfinite(photometric_lr_loss(torch.full((1, 1, h8 * w8), 20.0), imgs))


def test_photometric_gated_in_loss(tmp_path):
    """training.photometric_loss adds a live 'photo' loss item; zero when off (the default)."""
    import yaml as _yaml

    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.s3d.orientation import ORIENT_CHANNELS
    from ultralytics.utils import ROOT

    cfg = _yaml.safe_load((ROOT / "cfg/models/26/yolo26-s3d.yaml").read_text())
    cfg.setdefault("training", {})["photometric_loss"] = True
    p = tmp_path / "yolo26n-s3d-photo.yaml"
    p.write_text(_yaml.safe_dump(cfg))

    batch = {
        "img": torch.rand(1, 6, 64, 64),
        "batch_idx": torch.zeros(1),
        "cls": torch.zeros(1, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "aux_targets": {
            "lr_distance": torch.full((1, 1, 1), 2.0),
            "depth": torch.full((1, 1, 1), math.log(20.0)),
            "dimensions": torch.zeros(1, 1, 3),
            "orientation": torch.zeros(1, 1, ORIENT_CHANNELS),
            "proj_offset": torch.zeros(1, 1, 2),
        },
    }
    for yaml_path, expect_on in ((str(p), True), ("yolo26n-s3d.yaml", False)):
        model = YOLO(yaml_path)
        core = model.model
        core.args = get_cfg()
        criterion = core.init_criterion()
        core.train()
        preds = core(batch["img"])
        total, items = criterion.loss(preds, batch)
        assert "photo" in items
        if expect_on:
            assert float(items["photo"]) > 0.0 and total.requires_grad
        else:
            assert float(items["photo"]) == 0.0


def test_kitti_recipe_yaml_builds():
    """The opt-in KITTI recipe variant builds with photometric on and the finer cost volume."""
    from ultralytics import YOLO
    from ultralytics.nn.modules.block import StereoCostVolume

    model = YOLO("yolo26n-s3d-kitti.yaml")
    core = model.model
    cv = next(m for m in core.model if isinstance(m, StereoCostVolume))
    assert len(cv.disparities) == 48
    assert core.init_criterion().photometric_loss is True
