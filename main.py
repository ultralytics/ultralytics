print("Running")

from ultralytics.utils.tlc.detect.model import TLCYOLO  # noqa: E402

model = TLCYOLO('yolov8n.pt')  # initialize

results = model.train(data='coco128.yaml', model='yolov8s.pt', epochs=5, batch=32, imgsz=320, workers=0)
