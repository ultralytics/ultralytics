# Ultralytics YOLO 🚀, AGPL-3.0 license

import pytest
import torch

from ultralytics import YOLO
from ultralytics.utils import ASSETS, WEIGHTS_DIR, checks

CUDA_IS_AVAILABLE = checks.cuda_is_available()
CUDA_DEVICE_COUNT = checks.cuda_device_count()

MODEL = WEIGHTS_DIR / 'path with spaces' / 'yolov8n.pt'  # test spaces in path
DATA = 'coco8.yaml'
BUS = ASSETS / 'bus.jpg'


def test_checks():
    """Validate CUDA settings against torch CUDA functions."""
    assert torch.cuda.is_available() == CUDA_IS_AVAILABLE
    assert torch.cuda.device_count() == CUDA_DEVICE_COUNT


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason='CUDA is not available')
def test_train():
    """Test model training on a minimal dataset."""
    device = 0 if CUDA_DEVICE_COUNT == 1 else [0, 1]
    YOLO(MODEL).train(data=DATA, imgsz=64, epochs=1, device=device)  # requires imgsz>=64


@pytest.mark.slow
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason='CUDA is not available')
def test_predict_multiple_devices():
    """Validate model prediction on multiple devices."""
    model = YOLO('yolov8n.pt')
    model = model.cpu()
    assert str(model.device) == 'cpu'
    _ = model(BUS)  # CPU inference
    assert str(model.device) == 'cpu'

    model = model.to('cuda:0')
    assert str(model.device) == 'cuda:0'
    _ = model(BUS)  # CUDA inference
    assert str(model.device) == 'cuda:0'

    model = model.cpu()
    assert str(model.device) == 'cpu'
    _ = model(BUS)  # CPU inference
    assert str(model.device) == 'cpu'

    model = model.cuda()
    assert str(model.device) == 'cuda:0'
    _ = model(BUS)  # CUDA inference
    assert str(model.device) == 'cuda:0'


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason='CUDA is not available')
def test_autobatch():
    """Check batch size for YOLO model using autobatch."""
    from ultralytics.utils.autobatch import check_train_batch_size

    check_train_batch_size(YOLO(MODEL).model.cuda(), imgsz=128, amp=True)


@pytest.mark.slow
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason='CUDA is not available')
def test_utils_benchmarks():
    """Profile YOLO models for performance benchmarks."""
    from ultralytics.utils.benchmarks import ProfileModels

    # Pre-export a dynamic engine model to use dynamic inference
    YOLO(MODEL).export(format='engine', imgsz=32, dynamic=True, batch=1)
    ProfileModels([MODEL], imgsz=32, half=False, min_time=1, num_timed_runs=3, num_warmup_runs=1).profile()


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason='CUDA is not available')
def test_predict_sam():
    """Test SAM model prediction with various prompts."""
    from ultralytics import SAM
    from ultralytics.models.sam import Predictor as SAMPredictor

    # Load a model
    model = SAM(WEIGHTS_DIR / 'sam_b.pt')

    # Display model information (optional)
    model.info()

    # Run inference
    model(BUS, device=0)

    # Run inference with bboxes prompt
    model(BUS, bboxes=[439, 437, 524, 709], device=0)

    # Run inference with points prompt
    model(ASSETS / 'zidane.jpg', points=[900, 370], labels=[1], device=0)

    # Create SAMPredictor
    overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model=WEIGHTS_DIR / 'mobile_sam.pt')
    predictor = SAMPredictor(overrides=overrides)

    # Set image
    predictor.set_image(ASSETS / 'zidane.jpg')  # set with image file
    # predictor(bboxes=[439, 437, 524, 709])
    # predictor(points=[900, 370], labels=[1])

    # Reset image
    predictor.reset_image()
