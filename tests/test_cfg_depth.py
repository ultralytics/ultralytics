from ultralytics.cfg import TASKS, TASK2CALIBRATIONDATA, TASK2DATA, TASK2METRIC, TASK2MODEL


def test_depth_registered_in_tasks():
    assert "depth" in TASKS
    assert TASK2DATA["depth"] == "nyu-depth.yaml"
    assert TASK2CALIBRATIONDATA["depth"] == "nyu-depth.yaml"
    assert TASK2METRIC["depth"] == "metrics/delta1"
    assert TASK2MODEL["depth"] == "yolo26n-depth.pt"
