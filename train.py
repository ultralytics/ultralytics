from ultralytics import YOLO

model = YOLO('yolov8n.pt') # pass any model type
model.train(data="dataset.yaml", epochs=100, batch=32)  # train the model