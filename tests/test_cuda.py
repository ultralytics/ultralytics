# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

CUDA_IS_AVAILABLE = torch.cuda.is_available()
CUDA_DEVICE_COUNT = torch.cuda.device_count()

WEIGHTS_DIR = Path(SETTINGS['weights_dir'])
MODEL = WEIGHTS_DIR / 'path with spaces' / 'yolov8n.pt'  # test spaces in path
DATA = 'coco8.yaml'


def test_checks():
    from ultralytics.utils.checks import cuda_device_count, cuda_is_available

    assert cuda_device_count() == CUDA_DEVICE_COUNT
    assert cuda_is_available() == CUDA_IS_AVAILABLE


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason='CUDA is not available')
def test_train():
    model = YOLO(MODEL)
    model(data=DATA, imgsz=32, epochs=1, batch=-1, device=0)  # also test AutoBatch


@pytest.mark.skipif(CUDA_DEVICE_COUNT < 2, reason=f'DDP is not available, {CUDA_DEVICE_COUNT} device(s) found')
def test_train_ddp():
    model = YOLO(MODEL)
    model(data=DATA, imgsz=32, epochs=1, device=[0, 1])


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason='CUDA is not available')
def test_utils_benchmarks():
    from ultralytics.utils.benchmarks import ProfileModels

    YOLO(MODEL).export(format='engine', imgsz=32, dynamic=True, batch=1)  # pre-export engine model
    ProfileModels([MODEL], imgsz=32, half=False, min_time=1, num_timed_runs=3, num_warmup_runs=1).profile()
