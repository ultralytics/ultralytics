# Ultralytics YOLO 🚀, AGPL-3.0 license
import contextlib
from pathlib import Path

import pytest
import torch

from ultralytics import YOLO, download
from ultralytics.utils import ASSETS, SETTINGS

CUDA_IS_AVAILABLE = torch.cuda.is_available()
CUDA_DEVICE_COUNT = torch.cuda.device_count()

DATASETS_DIR = Path(SETTINGS['datasets_dir'])
WEIGHTS_DIR = Path(SETTINGS['weights_dir'])
MODEL = WEIGHTS_DIR / 'path with spaces' / 'yolov8n.pt'  # test spaces in path
DATA = 'coco8.yaml'


def test_checks():
    from ultralytics.utils.checks import cuda_device_count, cuda_is_available

    assert cuda_device_count() == CUDA_DEVICE_COUNT
    assert cuda_is_available() == CUDA_IS_AVAILABLE


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason='CUDA is not available')
def test_train():
    device = 0 if CUDA_DEVICE_COUNT == 1 else [0, 1]
    YOLO(MODEL).train(data=DATA, imgsz=64, epochs=1, device=device)  # requires imgsz>=64


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason='CUDA is not available')
def test_autobatch():
    from ultralytics.utils.autobatch import check_train_batch_size

    check_train_batch_size(YOLO(MODEL).model.cuda(), imgsz=128, amp=True)


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason='CUDA is not available')
def test_utils_benchmarks():
    from ultralytics.utils.benchmarks import ProfileModels

    # Pre-export a dynamic engine model to use dynamic inference
    YOLO(MODEL).export(format='engine', imgsz=32, dynamic=True, batch=1)
    ProfileModels([MODEL], imgsz=32, half=False, min_time=1, num_timed_runs=3, num_warmup_runs=1).profile()


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason='CUDA is not available')
def test_predict_sam():
    from ultralytics import SAM
    from ultralytics.models.sam import Predictor as SAMPredictor

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

    # Create SAMPredictor
    overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model='mobile_sam.pt')
    predictor = SAMPredictor(overrides=overrides)

    # Set image
    predictor.set_image(ASSETS / 'zidane.jpg')  # set with image file
    # predictor(bboxes=[439, 437, 524, 709])
    # predictor(points=[900, 370], labels=[1])

    # Reset image
    predictor.reset_image()


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason='CUDA is not available')
def test_model_ray_tune():
    with contextlib.suppress(RuntimeError):  # RuntimeError may be caused by out-of-memory
        YOLO('yolov8n-cls.yaml').tune(use_ray=True,
                                      data='imagenet10',
                                      grace_period=1,
                                      iterations=1,
                                      imgsz=32,
                                      epochs=1,
                                      plots=False,
                                      device='cpu')


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason='CUDA is not available')
def test_model_tune():
    YOLO('yolov8n-pose.pt').tune(data='coco8-pose.yaml', plots=False, imgsz=32, epochs=1, iterations=2, device='cpu')
    YOLO('yolov8n-cls.pt').tune(data='imagenet10', plots=False, imgsz=32, epochs=1, iterations=2, device='cpu')


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason='CUDA is not available')
def test_pycocotools():
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.models.yolo.pose import PoseValidator
    from ultralytics.models.yolo.segment import SegmentationValidator

    # Download annotations after each dataset downloads first
    url = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/'

    args = {'model': 'yolov8n.pt', 'data': 'coco8.yaml', 'save_json': True, 'imgsz': 64}
    validator = DetectionValidator(args=args)
    validator()
    validator.is_coco = True
    download(f'{url}instances_val2017.json', dir=DATASETS_DIR / 'coco8/annotations')
    _ = validator.eval_json(validator.stats)

    args = {'model': 'yolov8n-seg.pt', 'data': 'coco8-seg.yaml', 'save_json': True, 'imgsz': 64}
    validator = SegmentationValidator(args=args)
    validator()
    validator.is_coco = True
    download(f'{url}instances_val2017.json', dir=DATASETS_DIR / 'coco8-seg/annotations')
    _ = validator.eval_json(validator.stats)

    args = {'model': 'yolov8n-pose.pt', 'data': 'coco8-pose.yaml', 'save_json': True, 'imgsz': 64}
    validator = PoseValidator(args=args)
    validator()
    validator.is_coco = True
    download(f'{url}person_keypoints_val2017.json', dir=DATASETS_DIR / 'coco8-pose/annotations')
    _ = validator.eval_json(validator.stats)
