# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import pytest
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import classify_transforms
from ultralytics.engine.exporter import Exporter
from ultralytics.models.yolo import classify, detect, segment
from ultralytics.utils import ASSETS, DEFAULT_CFG, WEIGHTS_DIR

CFG_DET = 'yolov8n.yaml'
CFG_SEG = 'yolov8n-seg.yaml'
CFG_CLS = 'yolov8n-cls.yaml'  # or 'squeezenet1_0'
CFG = get_cfg(DEFAULT_CFG)
MODEL = WEIGHTS_DIR / 'yolov8n'


def test_func(*args):  # noqa
    """Test function callback."""
    print('callback test passed')


def test_export():
    """Test model exporting functionality."""
    exporter = Exporter()
    exporter.add_callback('on_export_start', test_func)
    assert test_func in exporter.callbacks['on_export_start'], 'callback test failed'
    f = exporter(model=YOLO(CFG_DET).model)
    YOLO(f)(ASSETS)  # exported model inference


def test_detect():
    """Test object detection functionality."""
    overrides = {'data': 'coco8.yaml', 'model': CFG_DET, 'imgsz': 32, 'epochs': 1, 'save': False}
    CFG.data = 'coco8.yaml'
    CFG.imgsz = 32

    # Trainer
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback('on_train_start', test_func)
    assert test_func in trainer.callbacks['on_train_start'], 'callback test failed'
    trainer.train()

    # Validator
    val = detect.DetectionValidator(args=CFG)
    val.add_callback('on_val_start', test_func)
    assert test_func in val.callbacks['on_val_start'], 'callback test failed'
    val(model=trainer.best)  # validate best.pt

    # Predictor
    pred = detect.DetectionPredictor(overrides={'imgsz': [64, 64]})
    pred.add_callback('on_predict_start', test_func)
    assert test_func in pred.callbacks['on_predict_start'], 'callback test failed'
    result = pred(source=ASSETS, model=f'{MODEL}.pt')
    assert len(result), 'predictor test failed'

    overrides['resume'] = trainer.last
    trainer = detect.DetectionTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f'Expected exception caught: {e}')
        return

    Exception('Resume test failed!')


def test_segment():
    """Test image segmentation functionality."""
    overrides = {'data': 'coco8-seg.yaml', 'model': CFG_SEG, 'imgsz': 32, 'epochs': 1, 'save': False}
    CFG.data = 'coco8-seg.yaml'
    CFG.imgsz = 32
    # YOLO(CFG_SEG).train(**overrides)  # works

    # trainer
    trainer = segment.SegmentationTrainer(overrides=overrides)
    trainer.add_callback('on_train_start', test_func)
    assert test_func in trainer.callbacks['on_train_start'], 'callback test failed'
    trainer.train()

    # Validator
    val = segment.SegmentationValidator(args=CFG)
    val.add_callback('on_val_start', test_func)
    assert test_func in val.callbacks['on_val_start'], 'callback test failed'
    val(model=trainer.best)  # validate best.pt

    # Predictor
    pred = segment.SegmentationPredictor(overrides={'imgsz': [64, 64]})
    pred.add_callback('on_predict_start', test_func)
    assert test_func in pred.callbacks['on_predict_start'], 'callback test failed'
    result = pred(source=ASSETS, model=f'{MODEL}-seg.pt')
    assert len(result), 'predictor test failed'

    # Test resume
    overrides['resume'] = trainer.last
    trainer = segment.SegmentationTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f'Expected exception caught: {e}')
        return

    Exception('Resume test failed!')


def test_classify():
    """Test image classification functionality."""
    overrides = {'data': 'imagenet10', 'model': CFG_CLS, 'imgsz': 32, 'epochs': 1, 'save': False}
    CFG.data = 'imagenet10'
    CFG.imgsz = 32
    # YOLO(CFG_SEG).train(**overrides)  # works

    # Trainer
    trainer = classify.ClassificationTrainer(overrides=overrides)
    trainer.add_callback('on_train_start', test_func)
    assert test_func in trainer.callbacks['on_train_start'], 'callback test failed'
    trainer.train()

    # Validator
    val = classify.ClassificationValidator(args=CFG)
    val.add_callback('on_val_start', test_func)
    assert test_func in val.callbacks['on_val_start'], 'callback test failed'
    val(model=trainer.best)

    # Predictor
    pred = classify.ClassificationPredictor(overrides={'imgsz': [64, 64]})
    pred.add_callback('on_predict_start', test_func)
    assert test_func in pred.callbacks['on_predict_start'], 'callback test failed'
    result = pred(source=ASSETS, model=trainer.best)
    assert len(result), 'predictor test failed'


@pytest.mark.parametrize('image_size,transform_size', [((640, 480), 500), ((480, 640), 500), ((640, 480), 244),
                                                       ((640, 480), 1024), ((500, 500), 500)])
def test_preprocessing_classify(image_size, transform_size):
    STRIDE = 32
    for rect in [False, True]:
        for letterbox in [False, True]:
            augment = classify_transforms(size=transform_size, rect=rect, letterbox=letterbox)
            image = np.random.randint(0, 255, (image_size[0], image_size[1], 3), dtype=np.uint8)
            preprocessed_image = augment(image)
            assert preprocessed_image.shape[0] == 3, 'preprocessed channelno not as expected'
            assert preprocessed_image.dtype == torch.float32, 'preprocessed type not as expected'
            assert preprocessed_image.min() >= 0 and preprocessed_image.max() <= 1, 'normalization not as expected'
            if rect:
                assert preprocessed_image.shape[1] % STRIDE == 0 and preprocessed_image.shape[
                    2] % STRIDE == 0, 'preprocessed size not as expected'
            else:
                assert preprocessed_image.shape[1] == transform_size and preprocessed_image.shape[
                    2] == transform_size, 'preprocessed size not as expected'
