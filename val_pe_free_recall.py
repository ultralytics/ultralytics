from ultralytics import YOLOE
from ultralytics.models.yolo.detect import DetectionValidator

model = YOLOE("pretrain/yolov8l-vl-seg-pf.pt")

filename = "ultralytics/cfg/datasets/lvis.yaml"

model.val(data=filename, batch=1, split="minival", rect=False, validator=DetectionValidator, single_cls=True)
