print("Running")

from ultralytics.utils.tlc.detect.model import TLCYOLO  # noqa: E402

model = TLCYOLO("yolov8n.pt")  # initialize

results = model.val(data="coco128.yaml", split="train", batch=32, imgsz=320, device=0, workers=0)
