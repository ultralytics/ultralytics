print("Running")

from ultralytics import YOLO  # noqa: E402

model = YOLO('yolov8n.pt')  # initialize

results = model.train(data='coco128.yaml', model='yolov8n.pt', epochs=2, batch=64, imgsz=320, workers=0)
