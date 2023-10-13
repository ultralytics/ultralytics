# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib

from ultralytics import YOLO
from ultralytics.utils import SETTINGS, checks


def test_model_ray_tune():
    """Tune YOLO model with Ray optimization library."""
    with contextlib.suppress(RuntimeError):  # RuntimeError may be caused by out-of-memory
        YOLO('yolov8n-cls.yaml').tune(use_ray=True,
                                      data='imagenet10',
                                      grace_period=1,
                                      iterations=1,
                                      imgsz=32,
                                      epochs=1,
                                      plots=False,
                                      device='cpu')


def test_mlflow():
    """Test training with MLflow tracking enabled."""
    checks.check_requirements('mlflow')
    current_setting = SETTINGS['mlflow']
    SETTINGS['mlflow'] = True
    YOLO('yolov8n-cls.yaml').train(data='imagenet10', imgsz=32, epochs=3, plots=False, device='cpu')
    SETTINGS['mlflow'] = current_setting
