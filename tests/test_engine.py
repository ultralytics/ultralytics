# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
from unittest import mock

import numpy as np
import pytest
import torch

from tests import MODEL, SOURCE
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.models.yolo import classify, detect, segment
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import ASSETS, DEFAULT_CFG, WEIGHTS_DIR
from ultralytics.utils.metrics import DetMetrics, compute_detect_fitness, normalize_detect_fitness_weights


def test_func(*args, **kwargs):
    """Test function used as a callback stub to verify callback registration."""
    print("callback test passed")


def test_export():
    """Test model exporting functionality by adding a callback and verifying its execution."""
    exporter = Exporter()
    exporter.add_callback("on_export_start", test_func)
    assert test_func in exporter.callbacks["on_export_start"], "callback test failed"
    f = exporter(model=YOLO("yolo26n.yaml").model)
    YOLO(f)(SOURCE)  # exported model inference


def test_detect():
    """Test YOLO object detection training, validation, and prediction functionality."""
    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 1, "save": False}
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
    # Confirm there is no issue with sys.argv being empty
    with mock.patch.object(sys, "argv", []):
        result = pred(source=ASSETS, model=MODEL)
        assert len(result), "predictor test failed"

    # Test resume functionality
    overrides["resume"] = trainer.last
    trainer = detect.DetectionTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return

    raise Exception("Resume test failed!")


def test_segment():
    """Test image segmentation training, validation, and prediction pipelines using YOLO models."""
    overrides = {
        "data": "coco8-seg.yaml",
        "model": "yolo26n-seg.yaml",
        "imgsz": 32,
        "epochs": 1,
        "save": False,
        "mask_ratio": 1,
        "overlap_mask": False,
    }
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "coco8-seg.yaml"
    cfg.imgsz = 32

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
    result = pred(source=ASSETS, model=WEIGHTS_DIR / "yolo26n-seg.pt")
    assert len(result), "predictor test failed"

    # Test resume functionality
    overrides["resume"] = trainer.last
    trainer = segment.SegmentationTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return

    raise Exception("Resume test failed!")


def test_classify():
    """Test image classification including training, validation, and prediction phases."""
    overrides = {"data": "imagenet10", "model": "yolo26n-cls.yaml", "imgsz": 32, "epochs": 1, "save": False}
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = "imagenet10"
    cfg.imgsz = 32

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


def test_nan_recovery():
    """Test NaN loss detection and recovery during training."""
    nan_injected = [False]

    def inject_nan(trainer):
        """Inject NaN into loss during batch processing to test recovery mechanism."""
        if trainer.epoch == 1 and trainer.tloss is not None and not nan_injected[0]:
            trainer.tloss *= torch.tensor(float("nan"))
            nan_injected[0] = True

    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 3}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_batch_end", inject_nan)
    trainer.train()
    assert nan_injected[0], "NaN injection failed"


def test_fitness_weights_cfg_validation():
    """Validate detect fitness_weights config parsing and normalization."""
    cfg = get_cfg(DEFAULT_CFG, overrides={"fitness_weights": (0, 0, 1, 0)})
    assert cfg.fitness_weights == [0.0, 0.0, 1.0, 0.0]

    cfg = get_cfg(DEFAULT_CFG, overrides={"fitness_weights": [0, 1, 0, 0]})
    assert cfg.fitness_weights == [0.0, 1.0, 0.0, 0.0]

    with pytest.raises(TypeError, match="fitness_weights"):
        get_cfg(DEFAULT_CFG, overrides={"fitness_weights": "0,0,1,0"})

    with pytest.raises(TypeError, match="fitness_weights"):
        get_cfg(DEFAULT_CFG, overrides={"fitness_weights": [False, 0, 1, 0]})

    with pytest.raises(ValueError, match="fitness_weights"):
        get_cfg(DEFAULT_CFG, overrides={"fitness_weights": [0, 0, 1]})

    with pytest.raises(ValueError, match="fitness_weights"):
        get_cfg(DEFAULT_CFG, overrides={"fitness_weights": [0, -1, 1, 0]})

    with pytest.raises(ValueError, match="fitness_weights"):
        get_cfg(DEFAULT_CFG, overrides={"fitness_weights": [0, float("nan"), 1, 0]})

    with pytest.raises(ValueError, match="fitness_weights"):
        get_cfg(DEFAULT_CFG, overrides={"fitness_weights": [0, float("inf"), 1, 0]})

    with pytest.raises(TypeError, match="fitness_weights"):
        get_cfg(DEFAULT_CFG, overrides={"fitness_weights": [0, "bad", 1, 0]})


def test_detect_fitness_helpers_default_and_validation():
    """Validate detect fitness helper defaults and error handling."""
    assert normalize_detect_fitness_weights() == [0.0, 0.0, 0.0, 1.0]
    assert compute_detect_fitness([0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.4)

    with pytest.raises(ValueError, match="fitness_weights"):
        normalize_detect_fitness_weights([0.0, 1.0, 0.0])

    with pytest.raises(TypeError, match="fitness_weights"):
        normalize_detect_fitness_weights([True, 0.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="fitness_weights"):
        normalize_detect_fitness_weights([0.0, float("nan"), 0.0, 1.0])

    with pytest.raises(ValueError, match="fitness_weights"):
        normalize_detect_fitness_weights([0.0, 0.0, -1.0, 1.0])

    with pytest.raises(ValueError, match="metrics must contain exactly 4 values"):
        compute_detect_fitness([0.1, 0.2, 0.3])


def test_det_metrics_custom_fitness_weights():
    """Compute detect fitness with custom weights."""
    metrics = DetMetrics(fitness_weights=[0.0, 0.0, 1.0, 0.0])
    metrics.box.p = np.array([0.2, 0.4])
    metrics.box.r = np.array([0.3, 0.5])
    metrics.box.all_ap = np.array([[0.8] + [0.4] * 9, [0.6] + [0.2] * 9], dtype=float)
    metrics.box.ap_class_index = np.array([0, 1])
    metrics.box.nc = 2

    assert metrics.fitness == pytest.approx(0.7)
    assert metrics.results_dict["fitness"] == pytest.approx(0.7)


def test_detection_validator_uses_custom_fitness_weights():
    """Pass train args into the detect validator metrics."""
    validator = detect.DetectionValidator(args={"data": "coco8.yaml", "model": "yolo26n.pt", "fitness_weights": [0, 1, 0, 0]})
    assert validator.metrics.fitness_weights == [0.0, 1.0, 0.0, 0.0]


def test_detection_trainer_logs_custom_fitness_weights():
    """Log when detect training uses explicit custom fitness weights."""
    with mock.patch("ultralytics.models.yolo.detect.train.LOGGER.info") as info_mock:
        detect.DetectionTrainer(
            overrides={
                "data": "coco8.yaml",
                "model": "yolo26n.yaml",
                "imgsz": 32,
                "epochs": 1,
                "save": False,
                "fitness_weights": [0.0, 0.0, 1.0, 0.0],
            }
        )

    assert any(
        call.args == ("Using custom fitness_weights=[0.0, 0.0, 1.0, 0.0] for best.pt selection",)
        for call in info_mock.call_args_list
    )


def test_detection_trainer_resume_preserves_fitness_weights(tmp_path):
    """Persist fitness_weights into args.yaml and restore them on resume."""
    overrides = {
        "data": "coco8.yaml",
        "model": "yolo26n.yaml",
        "imgsz": 32,
        "epochs": 1,
        "save": False,
        "fitness_weights": [0.0, 0.0, 1.0, 0.0],
    }
    trainer = detect.DetectionTrainer(overrides=overrides)
    assert trainer.args.fitness_weights == [0.0, 0.0, 1.0, 0.0]

    ckpt_path = tmp_path / "resume.pt"
    torch.save(
        {
            "model": DetectionModel("yolo26n.yaml", nc=80, ch=3, verbose=False),
            "train_args": {**overrides, "save_dir": str(trainer.save_dir)},
        },
        ckpt_path,
    )

    resumed = detect.DetectionTrainer(overrides={"resume": str(ckpt_path), "data": "coco8.yaml"})
    assert resumed.args.fitness_weights == [0.0, 0.0, 1.0, 0.0]


def test_detection_validator_coco_evaluate_uses_custom_fitness_weights():
    """Apply custom fitness weights to COCO eval overrides."""
    validator = detect.DetectionValidator(args={"data": "coco8.yaml", "model": "yolo26n.pt", "fitness_weights": [0, 0, 1, 0]})
    validator.args.save_json = True
    validator.is_coco = True
    validator.is_lvis = False
    validator.jdict = [{"image_id": 1}]
    validator.dataloader = mock.Mock()
    validator.dataloader.dataset.im_files = ["1.jpg"]

    pred_json = mock.Mock()
    pred_json.is_file.return_value = True
    anno_json = mock.Mock()
    anno_json.is_file.return_value = True

    coco_eval = mock.Mock()
    coco_eval.stats_as_dict = {"AP_50": 0.61, "AP_all": 0.27, "AP_small": 0.1, "AP_medium": 0.2, "AP_large": 0.3}

    fake_module = mock.Mock()
    fake_module.COCO.return_value.loadRes.return_value = mock.Mock()
    fake_module.COCOeval_faster.return_value = coco_eval

    stats = {"metrics/precision(B)": 0.11, "metrics/recall(B)": 0.22, "metrics/mAP50(B)": 0.33, "metrics/mAP50-95(B)": 0.44}
    with mock.patch("ultralytics.models.yolo.detect.val.check_requirements"), mock.patch.dict(
        sys.modules, {"faster_coco_eval": fake_module}
    ):
        results = validator.coco_evaluate(stats, pred_json, anno_json)

    assert results["metrics/mAP50(B)"] == pytest.approx(0.61)
    assert results["metrics/mAP50-95(B)"] == pytest.approx(0.27)
    assert results["fitness"] == pytest.approx(0.61)


def test_detection_validator_coco_evaluate_preserves_legacy_default_fitness():
    """Preserve the historical COCO fitness mix unless custom weights are explicitly set."""
    validator = detect.DetectionValidator(args={"data": "coco8.yaml", "model": "yolo26n.pt"})
    validator.args.save_json = True
    validator.is_coco = True
    validator.is_lvis = False
    validator.jdict = [{"image_id": 1}]
    validator.dataloader = mock.Mock()
    validator.dataloader.dataset.im_files = ["1.jpg"]

    pred_json = mock.Mock()
    pred_json.is_file.return_value = True
    anno_json = mock.Mock()
    anno_json.is_file.return_value = True

    coco_eval = mock.Mock()
    coco_eval.stats_as_dict = {"AP_50": 0.61, "AP_all": 0.27, "AP_small": 0.1, "AP_medium": 0.2, "AP_large": 0.3}

    fake_module = mock.Mock()
    fake_module.COCO.return_value.loadRes.return_value = mock.Mock()
    fake_module.COCOeval_faster.return_value = coco_eval

    stats = {"metrics/precision(B)": 0.11, "metrics/recall(B)": 0.22, "metrics/mAP50(B)": 0.33, "metrics/mAP50-95(B)": 0.44}
    with mock.patch("ultralytics.models.yolo.detect.val.check_requirements"), mock.patch.dict(
        sys.modules, {"faster_coco_eval": fake_module}
    ):
        results = validator.coco_evaluate(stats, pred_json, anno_json)

    assert results["metrics/mAP50(B)"] == pytest.approx(0.61)
    assert results["metrics/mAP50-95(B)"] == pytest.approx(0.27)
    assert results["fitness"] == pytest.approx(0.1 * 0.61 + 0.9 * 0.27)
