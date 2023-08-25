# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import pytest

from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from ultralytics.utils.checks import cuda_device_count, cuda_is_available

CUDA_IS_AVAILABLE = cuda_is_available()
CUDA_DEVICE_COUNT = cuda_device_count()

WEIGHTS_DIR = Path(SETTINGS['weights_dir'])
MODEL = WEIGHTS_DIR / 'path with spaces' / 'yolov8n.pt'  # test spaces in path
DATA = 'coco8.yaml'


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

    ProfileModels(['yolov8n.yaml'], imgsz=32, min_time=1, num_timed_runs=3, num_warmup_runs=1).profile()
