# Ultralytics YOLO 🚀, AGPL-3.0 license

from itertools import product
from pathlib import Path

import pytest
import torch

from tests import CUDA_DEVICE_COUNT, CUDA_IS_AVAILABLE, MODEL, SOURCE
from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS
from ultralytics.utils import ASSETS, WEIGHTS_DIR
from ultralytics.utils.checks import check_amp


def test_checks():
    """Validate CUDA settings against torch CUDA functions."""
    assert torch.cuda.is_available() == CUDA_IS_AVAILABLE
    assert torch.cuda.device_count() == CUDA_DEVICE_COUNT


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
def test_amp():
    """Test AMP training checks."""
    model = YOLO("yolo11n.pt").model.cuda()
    assert check_amp(model)


@pytest.mark.slow
@pytest.mark.skipif(True, reason="CUDA export tests disabled pending additional Ultralytics GPU server availability")
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch",
    [  # generate all combinations but exclude those where both int8 and half are True
        (task, dynamic, int8, half, batch)
        # Note: tests reduced below pending compute availability expansion as GPU CI runner utilization is high
        # for task, dynamic, int8, half, batch in product(TASKS, [True, False], [True, False], [True, False], [1, 2])
        for task, dynamic, int8, half, batch in product(TASKS, [True], [True], [False], [2])
        if not (int8 and half)  # exclude cases where both int8 and half are True
    ],
)
def test_export_engine_matrix(task, dynamic, int8, half, batch):
    """Test YOLO model export to TensorRT format for various configurations and run inference."""
    file = YOLO(TASK2MODEL[task]).export(
        format="engine",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
        data=TASK2DATA[task],
        workspace=1,  # reduce workspace GB for less resource utilization during testing
        simplify=True,  # use 'onnxslim'
    )
    YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32)  # exported model inference
    Path(file).unlink()  # cleanup
    Path(file).with_suffix(".cache").unlink() if int8 else None  # cleanup INT8 cache


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
def test_train():
    """Test model training on a minimal dataset using available CUDA devices."""
    device = 0 if CUDA_DEVICE_COUNT == 1 else [0, 1]
    YOLO(MODEL).train(data="coco8.yaml", imgsz=64, epochs=1, device=device)  # requires imgsz>=64


@pytest.mark.slow
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
def test_predict_multiple_devices():
    """Validate model prediction consistency across CPU and CUDA devices."""
    model = YOLO("yolo11n.pt")
    model = model.cpu()
    assert str(model.device) == "cpu"
    _ = model(SOURCE)  # CPU inference
    assert str(model.device) == "cpu"

    model = model.to("cuda:0")
    assert str(model.device) == "cuda:0"
    _ = model(SOURCE)  # CUDA inference
    assert str(model.device) == "cuda:0"

    model = model.cpu()
    assert str(model.device) == "cpu"
    _ = model(SOURCE)  # CPU inference
    assert str(model.device) == "cpu"

    model = model.cuda()
    assert str(model.device) == "cuda:0"
    _ = model(SOURCE)  # CUDA inference
    assert str(model.device) == "cuda:0"


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
def test_autobatch():
    """Check optimal batch size for YOLO model training using autobatch utility."""
    from ultralytics.utils.autobatch import check_train_batch_size

    check_train_batch_size(YOLO(MODEL).model.cuda(), imgsz=128, amp=True)


@pytest.mark.slow
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
def test_utils_benchmarks():
    """Profile YOLO models for performance benchmarks."""
    from ultralytics.utils.benchmarks import ProfileModels

    # Pre-export a dynamic engine model to use dynamic inference
    YOLO(MODEL).export(format="engine", imgsz=32, dynamic=True, batch=1)
    ProfileModels([MODEL], imgsz=32, half=False, min_time=1, num_timed_runs=3, num_warmup_runs=1).profile()


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
def test_predict_sam():
    """Test SAM model predictions using different prompts, including bounding boxes and point annotations."""
    from ultralytics import SAM
    from ultralytics.models.sam import Predictor as SAMPredictor

    # Load a model
    model = SAM(WEIGHTS_DIR / "sam_b.pt")

    # Display model information (optional)
    model.info()

    # Run inference
    model(SOURCE, device=0)

    # Run inference with bboxes prompt
    model(SOURCE, bboxes=[439, 437, 524, 709], device=0)

    # Run inference with points prompt
    model(ASSETS / "zidane.jpg", points=[900, 370], labels=[1], device=0)

    # Create SAMPredictor
    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model=WEIGHTS_DIR / "mobile_sam.pt")
    predictor = SAMPredictor(overrides=overrides)

    # Set image
    predictor.set_image(ASSETS / "zidane.jpg")  # set with image file
    # predictor(bboxes=[439, 437, 524, 709])
    # predictor(points=[900, 370], labels=[1])

    # Reset image
    predictor.reset_image()
