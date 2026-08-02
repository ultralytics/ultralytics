# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Configuration contracts for the public Detect3D training entry points."""

from ultralytics.cfg import DEFAULT_CFG, TASK2METRIC, TASK2MODEL


def test_detect3d_uses_generic_3d_map50_for_checkpoint_fitness():
    """Training fitness must use generic 3D mAP50, not a KITTI-only percentage metric."""
    assert TASK2METRIC["detect3d"] == "metrics/mAP50(3D)"
    assert TASK2MODEL["detect3d"] == "yolo26n-3d.yaml"
    assert DEFAULT_CFG.kitti_eval == "off"


def test_detect3d_release_defaults_do_not_change_global_checkpoint_policy():
    """Detect3D defaults should retain the release losses without changing global checkpoint behavior."""
    assert DEFAULT_CFG.epochs == 100
    assert DEFAULT_CFG.save_period == -1
    assert DEFAULT_CFG.d3_geometry_gain == 2.0
    assert DEFAULT_CFG.depth_z == 0.1
    assert DEFAULT_CFG.depth_z_tau == 2.0
    assert DEFAULT_CFG.quality3d_power == 0.5
    assert DEFAULT_CFG.calib is None
