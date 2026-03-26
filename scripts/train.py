from ultralytics import YOLO

model = YOLO('yolo26n.yaml')
results = model.train(data='scripts/cfg/config.yaml', amp=False, epochs=10)