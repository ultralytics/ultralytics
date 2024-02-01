from ultralytics.utils.tlc.detect.model import TLCYOLO

model = TLCYOLO("yolov8n.pt")  # initialize

for split in ("train", "val"):
    results = model.val(data="coco128.yaml", split=split, batch=32, imgsz=320, device=0, workers=0)
