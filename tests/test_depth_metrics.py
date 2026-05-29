import torch

from ultralytics.utils.metrics import DepthMetrics


def test_depth_metrics_perfect_prediction():
    m = DepthMetrics()
    gt = torch.rand(2, 1, 16, 16) * 5 + 0.5   # depths in [0.5, 5.5]
    m.update_stats(gt.clone(), gt.clone())     # perfect prediction
    m.process()
    res = m.results_dict
    assert res["metrics/delta1"] > 0.99
    assert res["metrics/abs_rel"] < 1e-4
    assert m.fitness == res["metrics/delta1"]
    assert m.keys[0] == "metrics/delta1"
    assert m.delta1 == res["metrics/delta1"]
    assert m.abs_rel == res["metrics/abs_rel"]
