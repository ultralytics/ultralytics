# Ultralytics YOLO 🚀, GPL-3.0 license

from pathlib import Path

from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, SETTINGS
from ultralytics.yolo.v8 import classify, detect, segment

CFG_DET = 'yolov8n.yaml'
CFG_SEG = 'yolov8n-seg.yaml'
CFG_CLS = 'squeezenet1_0'
CFG = get_cfg(DEFAULT_CFG)
TRANSFORM = [
    ["ultralytics.yolo.data.augment.LetterBox", dict(new_shape=(640, 640), scaleup=False)],
    [
        "ultralytics.yolo.data.augment.Format",
        dict(
            bbox_format="xywh",
            normalize=True,
            return_mask=False,
            return_keypoint=False,
            batch_idx=True,
            mask_ratio=4,
            mask_overlap=True
        )
    ]
]
MODEL = Path(SETTINGS['weights_dir']) / 'yolov8n'
SOURCE = ROOT / "assets"


def test_detect():
    overrides = {"data": "coco8.yaml", "model": CFG_DET, "imgsz": 32, "epochs": 1, "save": False}
    CFG.data = "coco8.yaml"

    # Trainer
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.train()
    
    # Trainer with transform
    trainer_transform = detect.DetectionTrainer(
        overrides=dict({"train_transform": TRANSFORM}, **overrides)
    )
    trainer_transform.train()

    # Validator
    val = detect.DetectionValidator(args=CFG)
    val(model=trainer.best)  # validate best.pt

    # Predictor
    pred = detect.DetectionPredictor(overrides={"imgsz": [64, 64]})
    result = pred(source=SOURCE, model=f"{MODEL}.pt")
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
    overrides = {"data": "coco8-seg.yaml", "model": CFG_SEG, "imgsz": 32, "epochs": 1, "save": False}
    CFG.data = "coco8-seg.yaml"
    CFG.v5loader = False
    # YOLO(CFG_SEG).train(**overrides)  # works

    # Trainer
    trainer = segment.SegmentationTrainer(overrides=overrides)
    trainer.train()
    
    # Trainer with transform
    TRANSFORM[1][1]["return_mask"] = True
    trainer_transform = segment.SegmentationTrainer(
        overrides=dict({"train_transform": TRANSFORM}, **overrides)
    )
    trainer_transform.train()

    # Validator
    val = segment.SegmentationValidator(args=CFG)
    val(model=trainer.best)  # validate best.pt

    # Predictor
    pred = segment.SegmentationPredictor(overrides={"imgsz": [64, 64]})
    result = pred(source=SOURCE, model=f"{MODEL}-seg.pt")
    assert len(result) == 2, "predictor test failed"

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
    overrides = {"data": "mnist160", "model": "yolov8n-cls.yaml", "imgsz": 32, "epochs": 1, "batch": 64, "save": False}
    CFG.data = "mnist160"
    CFG.imgsz = 32
    CFG.batch = 64
    # YOLO(CFG_SEG).train(**overrides)  # works

    # Trainer
    trainer = classify.ClassificationTrainer(overrides=overrides)
    trainer.train()

    # Validator
    val = classify.ClassificationValidator(args=CFG)
    val(model=trainer.best)

    # Predictor
    pred = classify.ClassificationPredictor(overrides={"imgsz": [64, 64]})
    result = pred(source=SOURCE, model=trainer.best)
    assert len(result) == 2, "predictor test failed"
