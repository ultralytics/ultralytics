# Ultralytics YOLO 🚀, GPL-3.0 license

from ultralytics import YOLO
from ultralytics.yolo.configs import get_config
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT
from ultralytics.yolo.v8 import classify, detect, segment

CFG_DET = 'yolov8n.yaml'
CFG_SEG = 'yolov8n-seg.yaml'
CFG_CLS = 'squeezenet1_0'
CFG = get_config(DEFAULT_CONFIG)
SOURCE = ROOT / "assets"


def test_detect():
    overrides = {"data": "coco128.yaml", "model": CFG_DET, "imgsz": 32, "epochs": 1, "save": False}
    CFG.data = "coco128.yaml"
    # trainer
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.train()
    trained_model = trainer.best

    # Validator
    val = detect.DetectionValidator(args=CFG)
    val(model=trained_model)

    # predictor
    pred = detect.DetectionPredictor(overrides={"imgsz": [640, 640]})
    i = 0
    for _ in pred(source=SOURCE, model="yolov8n.pt"):
        i += 1
    assert i == 2, "predictor test failed"

    overrides["resume"] = trainer.last
    trainer = detect.DetectionTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return

    Exception("Resume test failed!")


def test_segment():
    overrides = {"data": "coco128-seg.yaml", "model": CFG_SEG, "imgsz": 32, "epochs": 1, "save": False}
    CFG.data = "coco128-seg.yaml"
    CFG.v5loader = False

    # YOLO(CFG_SEG).train(**overrides) # This works
    # trainer
    trainer = segment.SegmentationTrainer(overrides=overrides)
    trainer.train()
    trained_model = trainer.best

    # Validator
    val = segment.SegmentationValidator(args=CFG)
    val(model=trained_model)

    # predictor
    pred = segment.SegmentationPredictor(overrides={"imgsz": [640, 640]})
    i = 0
    for _ in pred(source=SOURCE, model="yolov8n-seg.pt"):
        i += 1
    assert i == 2, "predictor test failed"

    # test resume
    overrides["resume"] = trainer.last
    trainer = segment.SegmentationTrainer(overrides=overrides)
    try:
        trainer.train()
    except Exception as e:
        print(f"Expected exception caught: {e}")
        return

    Exception("Resume test failed!")


def test_classify():
    overrides = {"data": "mnist160", "model": "yolov8n-cls.yaml", "imgsz": 32, "epochs": 1, "batch": 64, "save": False}
    CFG.data = "mnist160"
    CFG.imgsz = 32
    CFG.batch = 64
    # YOLO(CFG_SEG).train(**overrides) # This works
    # trainer
    trainer = classify.ClassificationTrainer(overrides=overrides)
    trainer.train()
    trained_model = trainer.best

    # Validator
    val = classify.ClassificationValidator(args=CFG)
    val(model=trained_model)

    # predictor
    pred = classify.ClassificationPredictor(overrides={"imgsz": [640, 640]})
    i = 0
    for _ in pred(source=SOURCE, model=trained_model):
        i += 1
    assert i == 2, "predictor test failed"
