from ultralytics import YOLO
from ultralytics.models import yolo

data_cfg = "/home/shu/Documents/PROTECH/ultralytics/ultralytics/cfg/datasets/coco8.yaml" 

model = YOLO('yolo11n.yaml').load('yolo11n.pt')  # load a model

# TODO: define trainer here
results = model.train(data=data_cfg, epochs=3, imgsz=640, trainer=yolo.detect.DetectionTrainer)  # train the model


