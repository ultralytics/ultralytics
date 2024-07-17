# Ultralytics YOLO 🚀, AGPL-3.0 license

import sys
from unittest import mock

from tests import MODEL
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.models.yolo import classify, detect, segment
from ultralytics.utils import ASSETS, DEFAULT_CFG, WEIGHTS_DIR


def test_func(*args):  # noqa
    """Test function callback for evaluating YOLO model performance metrics."""
    print("callback test passed")


def test_export():
    """Tests the model exporting function by adding a callback and asserting its execution."""
    exporter = Exporter()
    exporter.add_callback("on_export_start", test_func)
    assert test_func in exporter.callbacks["on_export_start"], "callback test failed"
    f = exporter(model=YOLO("yolov8n.yaml").model)
    YOLO(f)(ASSETS)  # exported model inference


def test_detect():
    """Test YOLO object detection training, validation, and prediction functionality."""
    overrides = {"data": "coco8.yaml", "model": "yolov8n.yaml", "imgsz": 32, "epochs": 1, "save": False}
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "coco8.yaml"
    cfg.imgsz = 32

    # Trainer
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    # Validator
    val = detect.DetectionValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)  # validate best.pt

    # Predictor
    pred = detect.DetectionPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    # Confirm there is no issue with sys.argv being empty.
    with mock.patch.object(sys, "argv", []):
        result = pred(source=ASSETS, model=MODEL)
        assert len(result), "predictor test failed"

    overrides["resume"] = trainer.last
    trainer = detect.DetectionTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return

    Exception("Resume test failed!")


def test_segment():
    """Tests image segmentation training, validation, and prediction pipelines using YOLO models."""
    overrides = {"data": "coco8-seg.yaml", "model": "yolov8n-seg.yaml", "imgsz": 32, "epochs": 1, "save": False}
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "coco8-seg.yaml"
    cfg.imgsz = 32
    # YOLO(CFG_SEG).train(**overrides)  # works

    # Trainer
    trainer = segment.SegmentationTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    # Validator
    val = segment.SegmentationValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)  # validate best.pt

    # Predictor
    pred = segment.SegmentationPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    result = pred(source=ASSETS, model=WEIGHTS_DIR / "yolov8n-seg.pt")
    assert len(result), "predictor test failed"

    # Test resume
    overrides["resume"] = trainer.last
    trainer = segment.SegmentationTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return

    Exception("Resume test failed!")


def test_classify():
    """Test image classification including training, validation, and prediction phases."""
    overrides = {"data": "imagenet10", "model": "yolov8n-cls.yaml", "imgsz": 32, "epochs": 1, "save": False}
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "imagenet10"
    cfg.imgsz = 32
    # YOLO(CFG_SEG).train(**overrides)  # works

    # Trainer
    trainer = classify.ClassificationTrainer(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "callback test failed"
    trainer.train()

    # Validator
    val = classify.ClassificationValidator(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "callback test failed"
    val(model=trainer.best)

    # Predictor
    pred = classify.ClassificationPredictor(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "callback test failed"
    result = pred(source=ASSETS, model=trainer.best)
    assert len(result), "predictor test failed"
