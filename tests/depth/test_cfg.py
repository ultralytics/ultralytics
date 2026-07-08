from ultralytics.cfg import TASKS, TASK2CALIBRATIONDATA, TASK2DATA, TASK2METRIC, TASK2MODEL


def test_depth_registered_in_tasks():
    assert "depth" in TASKS
    assert TASK2DATA["depth"] == "nyu-depth.yaml"
    assert TASK2CALIBRATIONDATA["depth"] == "nyu-depth.yaml"
    assert TASK2METRIC["depth"] == "metrics/delta1"
    assert TASK2MODEL["depth"] == "yolo26n-depth.pt"


def test_depth_hyperparameters_in_default_cfg():
    """Depth loss/calibration knobs are real cfg args with the documented defaults."""
    from ultralytics.cfg import get_cfg

    args = get_cfg()
    assert args.silog == 1.0
    assert args.silog_grad == 0.5
    assert args.silog_lambda == 0.5
    assert args.silog_l1 == 0.0
    assert args.dist_pw == 0.0
    assert args.cal_dist_pw == 0.0
    assert args.auto_calibrate is True


def test_depth_hyp_recipe_yaml_reproduces_release_training():
    """depth-hyp.yaml resolves by name, contains only valid cfg args, and carries the release recipe."""
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import YAML, checks

    recipe = YAML.load(checks.check_yaml("depth-hyp.yaml"))
    args = get_cfg(overrides=recipe)  # raises on any key not in default.yaml
    assert (args.degrees, args.translate, args.scale, args.shear) == (8.0, 0.08, 0.15, 2.0)
    assert (args.fliplr, args.flipud, args.perspective) == (0.5, 0.0, 0.0)
    assert (args.hsv_h, args.hsv_s, args.hsv_v) == (0.05, 0.3, 0.3)
