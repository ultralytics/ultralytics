from ultralytics.yolo.engine.model import YOLO



model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
results = model("https://ultralytics.com/images/bus.jpg", stream=False, save=True)  