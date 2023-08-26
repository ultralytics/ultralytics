# Ultralytics YOLO 🚀, AGPL-3.0 license
import subprocess
from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.utils import ASSETS, SETTINGS

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
    YOLO(MODEL).train(data=DATA, imgsz=64, epochs=1, batch=-1, device=0)  # also test AutoBatch, requires imgsz>=64


@pytest.mark.skipif(CUDA_DEVICE_COUNT < 2, reason=f'DDP is not available, {CUDA_DEVICE_COUNT} device(s) found')
def test_train_ddp():
    YOLO(MODEL).train(data=DATA, imgsz=64, epochs=1, device=[0, 1])  # requires imgsz>=64


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason='CUDA is not available')
def test_utils_benchmarks():
    from ultralytics.utils.benchmarks import ProfileModels

    YOLO(MODEL).export(format='engine', imgsz=32, dynamic=True, batch=1)  # pre-export engine model, auto-device
    ProfileModels([MODEL], imgsz=32, half=False, min_time=1, num_timed_runs=3, num_warmup_runs=1).profile()


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason='CUDA is not available')
def test_predict_sam():
    from ultralytics import SAM

    # Load a model
    model = SAM(WEIGHTS_DIR / 'sam_b.pt')

    # Display model information (optional)
    model.info()

    # Run inference
    model(ASSETS / 'bus.jpg', device=0)

    # Run inference with bboxes prompt
    model(ASSETS / 'zidane.jpg', bboxes=[439, 437, 524, 709], device=0)

    # Run inference with points prompt
    model(ASSETS / 'zidane.jpg', points=[900, 370], labels=[1], device=0)


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason='CUDA is not available')
def test_model_tune():
    subprocess.run('pip install ray[tune]'.split(), check=True)
    YOLO('yolov8n-cls.yaml').tune(data='imagenet10',
                                  grace_period=1,
                                  max_samples=1,
                                  imgsz=32,
                                  epochs=1,
                                  plots=False,
                                  device='cpu')
