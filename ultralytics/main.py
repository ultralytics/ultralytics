print("Running")

from ultralytics.utils.tlc import TLCYOLO  # noqa: E402

model = TLCYOLO('yolov8n.pt')  # initialize

results = model.train(data='coco128.yaml', model='yolov8n.pt', epochs=2, batch=64, imgsz=320)
