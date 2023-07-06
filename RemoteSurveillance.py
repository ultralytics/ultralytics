from ultralytics.yolo.engine.model import YOLO

model = YOLO('surveillance.pt')
metrics_t = model.predict(0, save=True, show=True)
