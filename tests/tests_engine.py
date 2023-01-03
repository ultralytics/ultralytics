from ultralytics.yolo.v8 import detect, segment, classify
from ultralytics.yolo.configs import get_config
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT

CFG_DET = 'yolov8n.yaml'
CFG_SEG = 'yolov8n-seg.yaml'
CFG_CLS = 'squeezenet1_0'
CFG = get_config(DEFAULT_CONFIG)
SOURCE = ROOT/ "assets"

def test_detect():
    overrides = {"data":"coco128.yaml", "model":CFG_DET, "imgsz":32, "epochs":1, "save":False}
    CFG.data = "coco128.yaml"
    # trainer
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.train()
    trained_model = trainer.best

    # Validator
    val = detect.DetectionValidator(args=CFG)
    val(model=trained_model)

    # predictor
    pred = detect.DetectionPredictor(overrides={"imgsz":[640,640]})
    pred(source=SOURCE, model=trained_model)

def test_segment():
    overrides = {"data":"coco128-seg.yaml", "model":CFG_SEG, "imgsz":32, "epochs":1, "save":False}
    CFG.data = "coco128-seg.yaml"
    # trainer
    # trainer
    trainer = segment.SegmentationTrainer(overrides=overrides)
    trainer.train()
    trained_model = trainer.best

    # Validator
    val = segment.SegmentationPredictor(args=CFG)
    val(model=trained_model)

    # predictor
    pred = segment.SegmentationPredictor(overrides={"imgsz":[640,640]})
    pred(source=SOURCE, model=trained_model)
    

