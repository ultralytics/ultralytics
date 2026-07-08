import torch


def _bare_validator():
    """Validator without __init__ (no dataloader/args needed for metric accumulation)."""
    from ultralytics.models.yolo.depth.val import DepthValidator

    v = DepthValidator.__new__(DepthValidator)
    v.calibrating = False  # normally set by __init__
    v.init_metrics(model=None)
    return v


def test_depth_validator_uses_metrics_object():
    v = _bare_validator()
    gt = torch.rand(1, 1, 16, 16) * 5 + 0.5
    v.update_metrics({"depth": gt.clone()}, {"depth": gt.clone()})
    stats = v.get_stats()
    assert stats["metrics/delta1"] > 0.99
    assert stats["fitness"] == stats["metrics/delta1"]


def test_depth_validator_imperfect_and_resized():
    v = _bare_validator()
    gt = torch.rand(1, 1, 32, 32) * 5 + 0.5
    pred = (gt * 1.5)[:, :, ::2, ::2]  # scaled error + half resolution (exercises interpolate)
    v.update_metrics(pred, {"depth": gt.clone()})
    stats = v.get_stats()
    assert 0.0 <= stats["metrics/delta1"] <= 1.0
    assert stats["metrics/abs_rel"] > 0.0  # imperfect prediction has nonzero error
