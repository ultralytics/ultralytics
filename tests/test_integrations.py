# Ultralytics YOLO 🚀, AGPL-3.0 license

import pytest

from ultralytics import YOLO
from ultralytics.utils import SETTINGS, checks


@pytest.mark.skipif(not checks.check_requirements('ray', install=False), reason='RayTune not installed')
def test_model_ray_tune():
    """Tune YOLO model with Ray optimization library."""
    YOLO('yolov8n-cls.yaml').tune(use_ray=True,
                                  data='imagenet10',
                                  grace_period=1,
                                  iterations=1,
                                  imgsz=32,
                                  epochs=1,
                                  plots=False,
                                  device='cpu')


@pytest.mark.skipif(not checks.check_requirements('mlflow', install=False), reason='MLflow not installed')
def test_mlflow():
    """Test training with MLflow tracking enabled."""
    SETTINGS['mlflow'] = True
    YOLO('yolov8n-cls.yaml').train(data='imagenet10', imgsz=32, epochs=3, plots=False, device='cpu')
