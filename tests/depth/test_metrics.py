import torch

from ultralytics.utils.metrics import DepthMetrics


def test_depth_metrics_perfect_prediction():
    m = DepthMetrics()
    gt = torch.rand(2, 1, 16, 16) * 5 + 0.5  # depths in [0.5, 5.5]
    m.update_stats(gt.clone(), gt.clone())  # perfect prediction
    m.process()
    res = m.results_dict
    assert res["metrics/delta1"] > 0.99
    assert res["metrics/abs_rel"] < 1e-4
    assert m.fitness == res["metrics/delta1"]
    assert m.keys[0] == "metrics/delta1"
    assert m.delta1 == res["metrics/delta1"]
    assert m.abs_rel == res["metrics/abs_rel"]


def test_median_alignment_recovers_scaled_prediction():
    """A globally mis-scaled prediction (correct structure) scores well after median alignment."""
    m = DepthMetrics()  # median alignment is the default eval protocol
    gt = torch.rand(2, 1, 16, 16) * 5 + 0.5
    pred = gt * 3.0  # wrong absolute scale, perfect relative structure
    m.update_stats(pred, gt)
    m.process()
    res = m.results_dict
    assert res["metrics/delta1"] > 0.99
    assert res["metrics/abs_rel"] < 1e-4


def test_median_alignment_is_per_image():
    """Each image is aligned with its own median scale, not one batch-wide factor."""
    m = DepthMetrics()
    gt = torch.rand(2, 1, 16, 16) * 5 + 0.5
    pred = gt.clone()
    pred[0] *= 2.0  # image 0 mis-scaled 2x
    pred[1] *= 10.0  # image 1 mis-scaled 10x
    m.update_stats(pred, gt)
    m.process()
    assert m.results_dict["metrics/delta1"] > 0.99


def test_process_does_not_invoke_collective(monkeypatch):
    """Ultralytics validates on rank 0 only — a collective while finalizing depth metrics would
    deadlock the other ranks (they never validate).
    """
    import torch.distributed as dist

    m = DepthMetrics()
    gt = torch.rand(1, 1, 8, 8) * 5 + 0.5
    m.update_stats(gt.clone(), gt.clone())
    calls = {"n": 0}
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))
    m.process()  # metrics finalize from rank-0 accumulators alone
    assert calls["n"] == 0
    assert m.results_dict["metrics/delta1"] > 0.99


def test_median_alignment_can_be_disabled():
    """With alignment off, a mis-scaled prediction scores poorly (legacy raw behavior)."""
    m = DepthMetrics(align="none")
    gt = torch.rand(2, 1, 16, 16) * 5 + 0.5
    m.update_stats(gt * 3.0, gt)
    m.process()
    assert m.results_dict["metrics/delta1"] < 0.01
